"""Focused deployment checks: history time, mapping, geometry and command ownership."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from tests.test_hereditary_geometry_model import _model
from src.control.hereditary_fast import FrozenHereditary
from src.control.hereditary_feedback import correct_state,physical_shape
from real_validation.runtime.hereditary_deployment import (ChannelMapping,HereditaryDeployment,advance,similarity_alignment,transform,resample_curve,edge_state_update)
from real_validation.perception.partial_edges import extract_edges_vectorized,project_camera
from real_validation.execution.executor import CommandReceipt,MockCommandTransport
from real_validation.execution.hereditary_executor import HereditaryExecutor


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.limits=threadpool_limits(1)
        self.tmp=tempfile.TemporaryDirectory()
        self.model=_model(residual_mode='none').eval();self.model.requires_grad_(False)
        self.engine=FrozenHereditary(self.model)
        self.now=100.
        self.meta=dict(dt=float(self.model.dt),expansion6=[0,1,1,2,2,3],action_unit_to_kpa=[150.]*4,
                       lower_kpa=[0.]*4,upper_kpa=[150.]*4,rate_kpa_s=[50.]*4,max_horizon=80,radius_mm=8.,checkpoint_sha256='test')
        self.r=HereditaryDeployment(self.engine,self.meta,Path(self.tmp.name)/'run',clock=lambda:self.now)
        self.r.initialize([0.]*6,100.)
    def tearDown(self):
        self.r.close();self.tmp.cleanup();self.limits.restore_original_limits()
    def receipt(self,t,p,ident='1'):
        six=tuple(self.r.mapping.expand(np.full(4,p)))
        return CommandReceipt(ident,six,six,t,t+.01,'ack')
    def test_nonzero_initialization_is_bounded_equilibrium_prior(self):
        self.r.initialize([30.]*6,100.)
        drive,_=self.engine.drive(np.full(4,.2))
        np.testing.assert_allclose(self.r.state,np.r_[np.repeat(drive,self.engine.n_play),np.repeat(drive,self.engine.n_maxwell)])
        self.assertFalse(self.r.ready)
        self.assertEqual(len(self.r.history),1)
        with self.assertRaises(ValueError):self.r.initialize([151.]*6)

    def test_limit_intersection_is_atomic_and_cannot_exclude_hold(self):
        self.r.initialize([30.]*6,100.)
        before=self.r.bounds;version=self.r.version
        with self.assertRaises(ValueError):self.r.configure_limits([0.]*6,[20.]*6,[50.]*6,[50.]*6)
        self.assertIs(self.r.bounds,before);self.assertEqual(self.r.version,version)
        upper=[100.,60.,80.,90.,100.,120.]
        result=self.r.configure_limits([0.]*6,upper,[50.]*6,[20.]*6,commit=False)
        np.testing.assert_allclose(result.upper*150,[100,60,90,120])
        self.assertIs(self.r.bounds,before)
        with self.assertRaises(ValueError):self.r.configure_limits([0,70,0,0,0,0],upper,[50.]*6,[50.]*6)

    def test_full_shape_fit_reduces_error_without_changing_camera(self):
        self.r.initialize([30.]*6,100.)
        target_state=self.r.state.copy();target_state[self.engine.n_p:]+=.05
        target=self.engine.observe(target_state,self.r.action)
        self.r.matrix=np.eye(3);before=np.linalg.norm(self.engine.observe(self.r.state,self.r.action)[1:]-target[1:],axis=1).mean()
        error=self.r.fit_full_state(target,100.)
        self.assertLessEqual(error,before+1e-6)
        np.testing.assert_array_equal(self.r.matrix,np.eye(3))

    def test_auto_horizon_gates_failure_and_copies_snapshot(self):
        goal=self.engine.observe(self.r.state,self.r.action)
        with self.assertRaises(ValueError):self.r.plan_to_tolerance(goal)
        self.r.ready=True
        before=self.r.state.copy();version=self.r.version
        plan=self.r.plan_to_tolerance(goal,1.,3.,20)
        self.assertTrue(plan['qualified']);self.assertEqual(len(plan['actions']),10)
        np.testing.assert_array_equal(self.r.state,before);self.assertEqual(self.r.version,version)
        self.assertEqual(plan['planning_exit'],'qualified')
        self.assertEqual(plan['planning_config']['iterations'],24)
        self.assertGreaterEqual(plan['planning_ms'],0.)
        tuned=self.r.plan_to_tolerance(goal,1.,3.,20,iterations=1,shooting_nfev=2,horizon_step=4)
        self.assertEqual(len(tuned['actions']),4);self.assertEqual(tuned['planning_config']['iterations'],1)
        self.assertEqual(tuned['attempts'][0]['b_iterations'],1)
        with self.assertRaises(ValueError):self.r.plan_to_tolerance(goal,iterations=0)
        distant=goal.copy();distant[1:]+=1000
        failed=self.r.plan_to_tolerance(distant,.01,.02,10)
        self.assertFalse(failed['qualified']);self.assertEqual(len(failed['attempts']),1)

    def test_partial_target_ignores_unconstrained_nodes_through_planning_and_execution(self):
        self.r.ready=True
        goal=self.engine.observe(self.r.state,self.r.action)
        goal[1:-1]+=1000  # Intentionally impossible placeholders, not constraints.
        ids=[self.engine.n_nodes-1]
        plan=self.r.plan_to_tolerance(goal,1.,3.,10,iterations=1,node_indices=ids)
        self.assertTrue(plan['qualified']);self.assertLess(plan['mean_error'],1.)
        np.testing.assert_array_equal(plan['node_indices'],ids)
        executor=HereditaryExecutor(self.r,MockCommandTransport(),lambda:None,clock=lambda:self.now)
        executor.hold()  # Exercise execution preflight and archive without motion.
        self.assertEqual(executor.execute(plan),[])
        with np.load(executor.last_execution_dir/'initial_plan.npz') as data:
            np.testing.assert_array_equal(data['node_indices'],ids)
        np.testing.assert_array_equal(self.r.control_node_indices,ids)

    def test_partial_goal_mapping_and_strict_contract(self):
        self.r.matrix=np.eye(3);self.r.alignment_confirmed=True
        tip=self.engine.n_nodes-1
        goal,ids=self.r.partial_goal([[30,40]],[tip])
        np.testing.assert_array_equal(goal[ids],[[30,40]])
        goal,ids=self.r.partial_goal([[10,20],[20,30]],[tip-2,tip-1,tip])
        np.testing.assert_allclose(goal[ids],[[10,20],[15,25],[20,30]])
        for invalid in ([],[0],[tip+1],[2,1],[1,3],[1.,2.]):
            with self.assertRaises(ValueError):self.r.partial_goal([[10,20],[20,30]],invalid)

    def test_automatic_segment_selects_interval_and_reverse_direction(self):
        self.r.ready=True
        shape=self.engine.observe(self.r.state,self.r.action)
        state=self.r.state.copy();version=self.r.version
        plan=self.r.plan_any_segment(shape[3:7][::-1],.05,.1,3,budget_s=.5,iterations=1,shooting_nfev=2)
        self.assertTrue(plan['qualified']);self.assertTrue(plan['matching_reversed'])
        np.testing.assert_array_equal(plan['node_indices'],[3,4,5,6])
        self.assertEqual(plan['target_matrix'].shape,(32,self.engine.n_nodes))
        self.assertEqual(plan['reference'].shape[1:],(32,2))
        np.testing.assert_array_equal(self.r.state,state);self.assertEqual(self.r.version,version)

    def test_dense_curve_cannot_pass_by_matching_only_two_endpoints(self):
        from real_validation.runtime.shape_target import segment_projection,target_distances
        shape=self.engine.observe(self.r.state,self.r.action);ids=[3,4]
        matrix=segment_projection(self.engine.n_nodes,ids)
        samples=matrix@shape
        samples[:,0]+=20*np.sin(np.linspace(0,np.pi,32))
        self.assertEqual(target_distances(shape,shape,ids).max(),0.)
        self.assertGreater(target_distances(shape,shape,ids,matrix,samples).mean(),10.)

    def test_dense_analytic_b_reduces_curve_error_and_retains_pressure_bounds(self):
        from real_validation.runtime.shape_target import segment_projection
        from real_validation.runtime.hereditary_math import fast_suffix_b
        self.r.initialize([30.]*6,100.)
        ids=[3,4,5,6];matrix=segment_projection(self.engine.n_nodes,ids)
        old=np.tile(self.r.action,(4,1));prediction=self.engine.rollout(self.r.state,old)
        ref=np.einsum('sn,tnc->tsc',matrix,prediction);ref[:,:,0]+=.5
        actions,info=fast_suffix_b(self.engine,self.r.state,old,self.r.action,ref,self.r.bounds,node_indices=ids,target_matrix=matrix)
        self.assertAlmostEqual(info['mse_before_mm2'],.125)
        self.assertLessEqual(info['mse_after_mm2'],info['mse_before_mm2'])
        self.assertTrue(self.r.bounds.valid(actions,self.r.action))

    def test_unqualified_trial_requires_explicit_consent_and_records_failure(self):
        import json
        self.r.ready=True
        plan=self._flat_plan(3);prediction=self.engine.rollout(self.r.state,plan['actions'])
        plan['goal']=plan['goal'].copy();plan['goal'][1:]+=100
        plan.update(qualified=False,tolerance=1.,max_node=3.,prediction=prediction,mean_error=141.4,max_error=141.4)
        transport=MockCommandTransport();executor=HereditaryExecutor(self.r,transport,lambda:None,clock=lambda:self.now)
        with self.assertRaisesRegex(ValueError,'明确允许'):executor.execute(plan)
        self.assertEqual(transport.commands,[])
        executor.hold();executor.execute(plan,allow_unqualified=True)
        metadata=json.loads((executor.last_execution_dir/'metadata.json').read_text())
        self.assertTrue(metadata['unqualified_trial']);self.assertFalse(metadata['planned_qualified'])
        with np.load(executor.last_execution_dir/'initial_plan.npz') as saved:
            self.assertTrue(saved['unqualified_trial']);self.assertFalse(saved['planned_qualified'])
        self.assertFalse(plan['qualified'])

    def test_trial_does_not_bypass_stale_preview_or_pressure_safety(self):
        self.r.ready=True
        plan=self._flat_plan(3);prediction=self.engine.rollout(self.r.state,plan['actions'])
        plan.update(qualified=False,tolerance=1.,max_node=3.,prediction=prediction+10)
        transport=MockCommandTransport();executor=HereditaryExecutor(self.r,transport,lambda:None,clock=lambda:self.now)
        with self.assertRaisesRegex(ValueError,'预览'):executor.execute(plan,allow_unqualified=True)
        plan['prediction']=prediction;plan['actions'][0]=1000
        with self.assertRaisesRegex(ValueError,'压力'):executor.execute(plan,allow_unqualified=True)
        self.assertEqual(transport.commands,[])

    def test_open_loop_records_blank_images_but_never_calls_correction(self):
        import json
        self.r.clock=time.monotonic;self.r.initialize([0.]*6)
        plan=self._flat_plan(4);plan['actions'][:]=.01
        transport=MockCommandTransport()
        # Black frames deliberately contain no usable robot evidence. In the
        # control ablation they are recorded, not used to gate edge coverage.
        frames=lambda:(np.zeros((30,30,3),np.uint8),time.monotonic())
        executor=HereditaryExecutor(self.r,transport,frames,use_correction=False,max_missing=1,max_skipped=1)
        with patch('real_validation.execution.hereditary_executor.deadline_feedback',side_effect=AssertionError('correction called')):
            with patch.object(self.r,'feedback',side_effect=AssertionError('observer called')):
                receipts=executor.execute(plan)
        self.assertEqual(len(receipts),4)
        np.testing.assert_allclose(transport.commands,self.r.mapping.expand(plan['actions']),atol=1e-9)
        folder=executor.last_execution_dir
        self.assertEqual(len(list((folder/'raw/cam0').glob('*.png'))),4)
        self.assertEqual(len(list((folder/'feedback_jobs').glob('*.json'))),0)
        rows=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
        self.assertTrue(all(row['revision_status']=='disabled' and not row['state_committed'] for row in rows))
        self.assertTrue(all(row['consecutive_skipped']==0 for row in rows))
        metadata=json.loads((folder/'metadata.json').read_text())
        self.assertEqual(metadata['control_mode'],'open_loop');self.assertFalse(metadata['correction_enabled'])

    def test_analytic_b_partial_cost_is_independent_of_other_nodes(self):
        from real_validation.runtime.hereditary_math import fast_suffix_b
        self.r.initialize([30.]*6,100.)
        old=np.tile(self.r.action,(5,1))
        ref=self.engine.rollout(self.r.state,old);ref[:,-1,0]+=2
        other=ref.copy();other[:,1:-1]+=1000
        ids=[self.engine.n_nodes-1]
        a,info=fast_suffix_b(self.engine,self.r.state,old,self.r.action,ref,self.r.bounds,node_indices=ids)
        b,altered=fast_suffix_b(self.engine,self.r.state,old,self.r.action,other,self.r.bounds,node_indices=ids)
        np.testing.assert_allclose(a,b,atol=1e-10)
        self.assertEqual(info['mse_before_mm2'],altered['mse_before_mm2'])
        self.assertTrue(self.r.bounds.valid(a,self.r.action))

    def test_warmup_requires_unique_quality_frames_and_timeout(self):
        from real_validation.runtime.deployment_quality import WarmupGate,WarmupCriteria
        gate=WarmupGate(WarmupCriteria(minimum_s=.2,timeout_s=1.,consecutive=2),100.)
        info=lambda:dict(prediction_px=np.zeros((15,2)),observer=dict(count=20,after=1.),visibility=dict(coverage=1.))
        self.assertFalse(gate.update(info(),100.,100.))
        self.assertFalse(gate.update(info(),100.1,100.1))
        self.assertFalse(gate.update(info(),100.1,100.2))
        self.assertFalse(gate.update(info(),100.3,100.3))
        self.assertTrue(gate.update(info(),100.4,100.4))
        bad=info();bad['visibility']['coverage']=.2
        self.assertFalse(gate.update(bad,100.5,100.5))
        with self.assertRaises(ValueError):gate.update(info(),101.1,101.1)

    def test_mapping_requires_all_inputs_and_equal_followers(self):
        m=ChannelMapping((3,2,2,1,1,0),(150.,)*4)
        u=np.array([.1,.2,.3,.4]);np.testing.assert_allclose(m.reduce(m.expand(u)),u)
        with self.assertRaises(ValueError):m.reduce([0,0,1,0,0,0])
        with self.assertRaises(ValueError):ChannelMapping((0,0,0,0,0,0),(150.,)*4)
    def test_negative_pressure_cannot_initialize_positive_model(self):
        with self.assertRaises(ValueError):self.r.initialize([-100]*6)
    def test_long_idle_relaxes_maxwell_without_erasing_play(self):
        self.r.acknowledge(self.receipt(101.,.5))
        self.r.acknowledge(self.receipt(102.,0.,'2'))
        before=self.r.state.copy();self.now=202.
        after,_=self.r.state_at(self.now)
        np.testing.assert_array_equal(after[:self.engine.n_p],before[:self.engine.n_p])
        self.assertLess(np.linalg.norm(after[self.engine.n_p:]),np.linalg.norm(before[self.engine.n_p:]))
        self.assertGreater(np.linalg.norm(after[:self.engine.n_p]),0.)
    def test_new_epoch_preserves_known_memory_and_ack_deduplication(self):
        self.r.acknowledge(self.receipt(101.,.5))
        zero=self.receipt(102.,0.,'2');self.r.acknowledge(zero)
        self.now=110.;expected,action=self.r.state_at(self.now)
        epoch=self.r.history_epoch
        self.r.begin_history_epoch()
        self.assertEqual(self.r.history_epoch,epoch+1)
        np.testing.assert_allclose(self.r.state,expected)
        np.testing.assert_array_equal(self.r.action,action)
        self.assertGreater(np.linalg.norm(self.r.state[:self.engine.n_p]),0)
        with self.assertRaises(ValueError):self.r.state_at(109.)
        version=self.r.version;self.r.acknowledge(zero)
        self.assertEqual(self.r.version,version)
        events=(self.r.run_dir/'events.jsonl').read_text()
        self.assertIn('command_receipt',events)
        self.assertIn('history_epoch_started',events)
        self.assertEqual(self.r.phase,'initial_hold')

    def test_new_epoch_requires_confirmed_history(self):
        self.r.pending.add('in_flight')
        with self.assertRaises(ValueError):self.r.begin_history_epoch()
        self.r.pending.clear();self.r.fault='failed ACK'
        with self.assertRaises(ValueError):self.r.begin_history_epoch()

    def test_cold_initialize_invalidates_alignment_and_old_images(self):
        self.r.matrix=np.eye(3);self.r.alignment_confirmed=True;self.r.last_frame=101.
        self.now=102.;self.r.initialize([0.]*6)
        self.assertIsNone(self.r.matrix)
        self.assertFalse(self.r.alignment_confirmed)
        self.assertEqual(self.r.last_frame,-float('inf'))
        with self.assertRaises(ValueError):self.r.state_at(101.)

    def test_split_holds_equal_single_hold(self):
        z=self.r.state;u=np.full(4,.3)
        a=advance(self.engine,z,u,.7,self.r.dt)
        b=advance(self.engine,advance(self.engine,z,u,.2,self.r.dt),u,.5,self.r.dt)
        np.testing.assert_allclose(a,b,atol=1e-12)
    def test_ack_id_is_not_consumed_twice_and_failures_invalidate(self):
        receipt=self.receipt(101.,.1);self.r.acknowledge(receipt);version=self.r.version
        self.r.acknowledge(receipt);self.assertEqual(version,self.r.version)
        bad=CommandReceipt('bad',(0,)*6,(0,)*6,102.,None,'timeout')
        with self.assertRaises(ValueError):self.r.acknowledge(bad)
        self.assertIsNotNone(self.r.fault)
    def test_similarity_recovers_base_rotation_scale(self):
        p=np.column_stack([np.linspace(0,100,15),np.zeros(15)])
        theta=.6;m=np.eye(3);m[:2,:2]=1.7*np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);m[:2,2]=[130,75]
        estimated,error=similarity_alignment(p,transform(p,m))
        np.testing.assert_allclose(estimated,m,atol=1e-10);self.assertLess(error,1e-10)
    def test_full_brush_resampling_and_base_gate(self):
        self.r.matrix=np.eye(3);self.r.alignment_confirmed=True
        with self.assertRaises(ValueError):self.r.goal([[999,999],[1010,1020]])
        curve=resample_curve([[0,0],[10,0],[10,10]],15)
        self.assertEqual(curve.shape,(15,2));np.testing.assert_array_equal(curve[-1],[10,10])
    def test_late_image_replays_commands_and_duplicate_rejected(self):
        self.r.matrix=np.eye(3);self.r.alignment_confirmed=True
        self.r.acknowledge(self.receipt(100.1,.2));self.r.acknowledge(self.receipt(100.2,.3,'2'))
        self.now=100.25;expected,_=self.r.state_at(self.now)
        image=np.zeros((480,640,3),np.uint8)
        old=np.tile(self.r.action,(3,1));ref=self.engine.rollout(expected,old)
        self.r.feedback(image,100.15,old,ref)
        actual,_=self.r.state_at(self.now);np.testing.assert_allclose(actual,expected,atol=1e-10)
        _,info=self.r.feedback(image,100.15,old,ref);self.assertEqual(info['reason'],'stale_or_duplicate_frame')
    def test_numpy_observer_matches_torch_on_fixed_edges(self):
        u=np.full(4,.3,dtype=np.float32)
        z=self.model.init_z_from_action(torch.tensor(u[None,None]))[0].detach().numpy()
        m=np.eye(3,dtype=np.float32);m[:2,:2]*=10;m[:2,2]=[150,150]
        shape=self.engine.observe(z,u);pixels=transform(shape,m)
        image=np.full((500,500,3),25,np.uint8)
        cv2.polylines(image,[np.rint(pixels+[0,2]).astype(np.int32)],False,(225,225,225),16)
        evidence=extract_edges_vectorized(image,pixels,radius=8)
        self.assertGreater(len(evidence.pixels),0)
        np_z,info=edge_state_update(self.engine,z,u,evidence,m,8,self.r.bounds)
        action=torch.tensor(u);state=torch.tensor(z);matrix=torch.tensor(m)
        residual=lambda zz:evidence.residual(project_camera(physical_shape(self.model,zz,action),matrix),8)
        tz,ti=correct_state(self.model,state,action,residual,torch.zeros(4),torch.ones(4))
        self.assertEqual(info['accepted'],ti['accepted']);np.testing.assert_allclose(np_z,tz.numpy(),atol=2e-5)
    def test_executor_rejects_plan_changed_by_later_command(self):
        self.now=100.1;goal=self.engine.observe(self.r.state,self.r.action)
        plan=self.r.plan(goal,3);self.r.acknowledge(self.receipt(100.2,.01))
        executor=HereditaryExecutor(self.r,None,lambda:None)
        with self.assertRaisesRegex(ValueError,'过期'):executor.execute(plan)

    def test_changed_preview_is_rejected_before_any_dispatch(self):
        plan=self._flat_plan();prediction=self.engine.rollout(self.r.state,plan['actions'])
        plan.update(qualified=True,tolerance=1.,max_node=3.,prediction=prediction+10.)
        self.r.ready=True;transport=MockCommandTransport()
        with self.assertRaisesRegex(ValueError,'预览'):
            HereditaryExecutor(self.r,transport,lambda:None,clock=lambda:self.now).execute(plan)
        self.assertEqual(transport.commands,[])

    def test_normal_hold_cancels_without_sending_zero(self):
        transport=MockCommandTransport();executor=HereditaryExecutor(self.r,transport,lambda:None)
        executor.hold();rows=executor.execute(self._flat_plan())
        self.assertEqual(rows,[]);self.assertEqual(transport.commands,[])
        self.assertEqual(self.r.phase,'final_hold')
        self.assertIn('execute_held',(self.r.run_dir/'events.jsonl').read_text())

    def test_limits_intersect_every_shared_physical_chamber(self):
        self.r.configure_limits([0]*6,[150,50,10,150,150,150],[50,100,20,50,50,50],[50]*6)
        self.assertAlmostEqual(self.r.bounds.upper[1]*150,10)
        self.assertAlmostEqual(self.r.bounds.rise[1]*150/self.r.dt,20)
        with self.assertRaises(ValueError):self.r.configure_limits([-1]*6,[150]*6,[50]*6,[50]*6)

    def _flat_plan(self,steps=3):
        return dict(version=self.r.version,actions=np.zeros((steps,4)),reference=np.tile(self.engine.observe(self.r.state,self.r.action),(steps,1,1)),goal=self.engine.observe(self.r.state,self.r.action))

    def test_slow_feedback_does_not_round_up_another_full_slot(self):
        self.r.clock=time.monotonic;self.r.initialize([0.]*6)
        transport=MockCommandTransport();times=[]
        def feedback(image,stamp,tail,reference):
            times.append(time.monotonic());time.sleep(.065)
            return tail,dict(observer=dict(count=1),compute_ms=65.)
        frames=lambda:(np.zeros((10,10,3),np.uint8),time.monotonic())
        with patch.object(self.r,'feedback',side_effect=feedback):
            rows=HereditaryExecutor(self.r,transport,frames).execute(self._flat_plan(3))
        intervals=np.diff([row.t_command for row in rows])
        self.assertTrue(np.all(intervals>=self.r.dt-.001))
        # A 65 ms correction must not round dispatch to another 100 ms slot.
        self.assertLess(float(np.median(intervals)),.18)

    def test_deadline_discards_state_and_plan_without_queueing_or_delaying_commands(self):
        import threading,json
        self.r.clock=time.monotonic;self.r.initialize([0.]*6)
        release=threading.Event();entered=[]
        def slow(snapshot,image,stamp,tail,reference):
            entered.append(snapshot)
            release.wait(2.)
            snapshot.state[:]=99.;snapshot.version+=1
            return tail+.01,dict(observer=dict(count=1),control=dict(accepted=True))
        frames=lambda:(np.zeros((10,10,3),np.uint8),time.monotonic())
        transport=MockCommandTransport()
        executor=HereditaryExecutor(self.r,transport,frames)
        try:
            with patch.object(HereditaryDeployment,'feedback',slow):
                rows=executor.execute(self._flat_plan(3))
                self.assertEqual(len(entered),1)  # one stuck job, no queued work
                self.assertIsNot(entered[0].lock,self.r.lock)
                self.assertLess(max(np.diff([row.t_command for row in rows])),.18)
                before=self.r.state.copy();version=self.r.version
                release.set();self.r._feedback_job.thread.join(1.)
                np.testing.assert_array_equal(self.r.state,before)
                self.assertEqual(self.r.version,version)
                self.assertFalse(self.r.ready)
            steps=[json.loads(line) for line in (executor.last_execution_dir/'steps.jsonl').read_text().splitlines()]
            self.assertEqual([x['revision_status'] for x in steps],['deadline_expired','worker_busy','worker_busy'])
            for row in steps:
                self.assertFalse(row['state_committed']);self.assertFalse(row['suffix_changed'])
            proposal=json.loads((executor.last_execution_dir/'feedback_jobs/00000.json').read_text())
            self.assertGreater(proposal['wall_ms'],100.)
            self.assertTrue(proposal['diagnostics']['control']['accepted']) # proposed, never applied
        finally:
            release.set()
            if hasattr(self.r,'_feedback_job'):self.r._feedback_job.thread.join(1.)

    def test_feedback_transaction_accepts_only_current_version_before_deadline(self):
        self.r.clock=time.monotonic
        snapshot,version=self.r.feedback_snapshot()
        snapshot.state[:]=.123;snapshot.version+=1
        ok,reason=self.r.commit_feedback(snapshot,version,time.monotonic()-.01)
        self.assertFalse(ok);self.assertEqual(reason,'deadline_expired')
        self.r.version+=1
        ok,reason=self.r.commit_feedback(snapshot,version,time.monotonic()+1.)
        self.assertFalse(ok);self.assertEqual(reason,'snapshot_changed')
        snapshot,version=self.r.feedback_snapshot();snapshot.state[:]=.123;snapshot.version+=1
        ok,reason=self.r.commit_feedback(snapshot,version,time.monotonic()+1.)
        self.assertTrue(ok);np.testing.assert_array_equal(self.r.state,snapshot.state)

    def test_repeated_deadline_skips_stop_at_configured_limit(self):
        import threading,json
        self.r.clock=time.monotonic;self.r.initialize([0.]*6)
        release=threading.Event()
        def slow(snapshot,image,stamp,tail,reference):
            release.wait(2.);return tail,dict(observer=dict(count=1))
        executor=HereditaryExecutor(self.r,MockCommandTransport(),lambda:(np.zeros((10,10,3),np.uint8),time.monotonic()),max_skipped=2)
        try:
            with patch.object(HereditaryDeployment,'feedback',slow):
                with self.assertRaisesRegex(RuntimeError,'及时提交反馈'):executor.execute(self._flat_plan(5))
            self.assertEqual(len(executor.transport.commands),3)
            self.assertEqual(executor.transport.commands[-1],(0.,)*6)
            self.assertEqual(len((executor.last_execution_dir/'steps.jsonl').read_text().splitlines()),2)
        finally:
            release.set();self.r._feedback_job.thread.join(1.)

    def test_applied_slew_clipping_rebases_tail_before_feedback(self):
        from dataclasses import replace
        class ClippedTransport(MockCommandTransport):
            def send(self,action6,required_groups,timeout_s):
                receipt=super().send(action6,required_groups,timeout_s)
                return replace(receipt,applied6=tuple(.5*np.asarray(receipt.applied6)))
        self.r.clock=time.monotonic;self.r.initialize([0.]*6)
        rise=self.r.bounds.rise
        actions=np.array([.8*rise,1.8*rise])
        reference=self.engine.rollout(self.r.state,actions)
        plan=dict(version=self.r.version,actions=actions,reference=reference,goal=reference[-1])
        observed=[]
        def feedback(image,stamp,tail,ref):
            self.assertTrue(self.r.bounds.valid(tail,self.r.action))
            observed.append(tail.copy())
            return tail,dict(observer=dict(count=1),compute_ms=0.)
        transport=ClippedTransport()
        frame=lambda:(np.zeros((10,10,3),np.uint8),time.monotonic())
        with patch.object(self.r,'feedback',side_effect=feedback):
            receipts=HereditaryExecutor(self.r,transport,frame).execute(plan)
        self.assertEqual(len(receipts),2)
        self.assertLess(observed[0][0,0],actions[1,0])
        self.assertIn('applied_suffix_rebase',(self.r.run_dir/'events.jsonl').read_text())

    def test_command_failure_sends_zero_and_keeps_history_fault(self):
        transport=MockCommandTransport(fail_at=1)
        executor=HereditaryExecutor(self.r,transport,lambda:None)
        with self.assertRaises(ValueError):executor.execute(self._flat_plan())
        self.assertEqual(transport.commands[-1],(0.,)*6)
        self.assertIsNotNone(self.r.fault)

    def test_camera_loss_stops_after_three_missing_frames(self):
        transport=MockCommandTransport()
        executor=HereditaryExecutor(self.r,transport,lambda:None)
        with self.assertRaisesRegex(RuntimeError,'没有新图像'):executor.execute(self._flat_plan(5))
        self.assertEqual(len(transport.commands),4)
        self.assertEqual(transport.commands[-1],(0.,)*6)

    def test_operator_abort_does_not_issue_the_plan(self):
        transport=MockCommandTransport();executor=HereditaryExecutor(self.r,transport,lambda:None)
        executor.abort()
        with self.assertRaisesRegex(RuntimeError,'operator_abort'):executor.execute(self._flat_plan())
        self.assertEqual(transport.commands,[(0.,)*6])

    def test_short_blind_plan_cannot_leave_deployment_ready(self):
        self.r.clock=time.monotonic;self.r.matrix=np.eye(3);self.r.ready=True
        transport=MockCommandTransport()
        frame=lambda:(np.zeros((480,640,3),np.uint8),time.monotonic())
        executor=HereditaryExecutor(self.r,transport,frame)
        executor.execute(self._flat_plan(2))
        self.assertFalse(self.r.ready)
        import json
        metadata=json.loads((executor.last_execution_dir/'metadata.json').read_text())
        self.assertEqual(metadata['status'],'completed_unobserved')

    def test_fresh_but_fully_blind_images_also_stop(self):
        self.r.clock=time.monotonic;self.r.matrix=np.eye(3)
        transport=MockCommandTransport()
        frame=lambda:(np.zeros((480,640,3),np.uint8),time.monotonic())
        executor=HereditaryExecutor(self.r,transport,frame)
        with self.assertRaisesRegex(RuntimeError,'没有有效图像边缘'):executor.execute(self._flat_plan(5))
        self.assertEqual(len(transport.commands),4)
        self.assertEqual(transport.commands[-1],(0.,)*6)
        self.assertEqual(len(list((executor.last_execution_dir/'raw/cam0').glob('*.png'))),3)
        self.assertFalse(self.r.ready)

class InitialShapeTests(unittest.TestCase):
    def image(self):
        image=np.full((300,300,3),25,np.uint8)
        curve=np.column_stack([150+20*np.sin(np.linspace(0,3,80)),np.linspace(45,250,80)])
        cv2.polylines(image,[np.rint(curve).astype(np.int32)],False,(225,225,225),16)
        return image,curve

    def test_pixel_extraction_both_polarities_without_model_input(self):
        from real_validation.perception.initial_shape import extract_initial_shape
        image,reference=self.image()
        for pixels,polarity in ((image,'bright'),(255-image,'dark')):
            curve,mask,info=extract_initial_shape(pixels,15,polarity)
            self.assertEqual(curve.shape,(15,2))
            # cv2 polylines have round caps extending ~8 px past these centers.
            # The physical endpoints are the outer cap, not the drawing centers.
            for point,center in zip(curve[[0,-1]],reference[[0,-1]]):
                self.assertLess(np.linalg.norm(point-center),10)
                xy=np.rint(point).astype(int)
                self.assertLessEqual(cv2.distanceTransform(mask,cv2.DIST_L2,5)[xy[1],xy[0]],2.)
            self.assertTrue(info['base_endpoint_fix_applied'])
            self.assertTrue(info['tip_endpoint_fix_applied'])
            self.assertGreater(info['arc_length_px'],190)
            self.assertGreater(mask.sum(),0)

    def test_occluded_fragments_are_not_initialized_as_whole_arm(self):
        from real_validation.perception.initial_shape import extract_initial_shape
        image,_=self.image();image[130:175,120:200]=25
        with self.assertRaisesRegex(ValueError,'分离区域'):extract_initial_shape(image,15)

    def test_blank_or_cropped_image_requires_new_capture(self):
        from real_validation.perception.initial_shape import extract_initial_shape
        with self.assertRaises(ValueError):extract_initial_shape(np.zeros((300,300,3),np.uint8),15)
        image,_=self.image()
        with self.assertRaisesRegex(ValueError,'边界'):extract_initial_shape(image[100:],15)

    def test_flat_short_edges_and_guided_refinement_both_directions(self):
        from real_validation.perception.initial_shape import extract_initial_shape,refine_initial_shape
        # Compare known physical short-edge centers on upright and rotated arms.
        for angle in (0.,35.):
            mask=np.zeros((320,320),np.uint8)
            rectangle=np.array([[146,55],[174,55],[174,265],[146,265]],float)
            theta=np.deg2rad(angle);rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            poly=(rectangle-160)@rot.T+160
            caps=(np.array([[160,55],[160,265]])-160)@rot.T+160
            cv2.fillPoly(mask,[np.rint(poly).astype(np.int32)],255)
            image=np.repeat(np.where(mask,225,25)[...,None],3,axis=2).astype(np.uint8)
            for pixels,polarity in ((image,'bright'),(255-image,'dark')):
                curve,_,_=extract_initial_shape(pixels,15,polarity)
                np.testing.assert_allclose(curve[[0,-1]],caps,atol=2.)
                guide=np.linspace(caps[0],caps[1],15)+np.array([5.,0])@rot.T
                for draft in (guide,guide[::-1]):
                    refined,_,info=refine_initial_shape(pixels,draft,polarity)
                    expected=caps if draft is guide else caps[::-1]
                    np.testing.assert_allclose(refined[[0,-1]],expected,atol=2.)
                    normal=rot[:,0]
                    error=np.abs((refined[2:-2]-160)@normal).mean()
                    self.assertLess(error,1.)
                    self.assertEqual(info['tip_endpoint_fix_reason'],'applied')

    def test_guided_refinement_rejects_blank_or_occluded_image_without_mutating_draft(self):
        from real_validation.perception.initial_shape import refine_initial_shape
        image,curve=self.image();draft=resample_curve(curve,15);original=draft.copy()
        hidden=image.copy();hidden[130:175]=25
        for pixels in (np.zeros_like(image),hidden):
            with self.assertRaises(ValueError):refine_initial_shape(pixels,draft)
            np.testing.assert_array_equal(draft,original)


if __name__=='__main__':unittest.main()
