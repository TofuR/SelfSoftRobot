"""Finite previewed control margin and reuse of known deployment geometry."""
import json
import time
import unittest
from unittest.mock import patch
import numpy as np

from tests import test_hereditary_deployment as fixtures
from real_validation.execution.hereditary_executor import HereditaryExecutor
from real_validation.execution.executor import MockCommandTransport
from real_validation.runtime.shape_target import segment_projection
from real_validation.runtime.hereditary_math import fast_suffix_b


class ReserveTests(unittest.TestCase):
    setUp=fixtures.DeploymentTests.setUp
    tearDown=fixtures.DeploymentTests.tearDown

    def test_reserve_is_previewed_feasible_and_targets_requested_shape(self):
        self.r.ready=True
        goal=self.engine.observe(self.r.state,self.r.action);goal[1:]+=100
        plan=self.r.plan_to_tolerance(goal,.01,.02,10,iterations=1,shooting_nfev=2,reserve_steps=4)
        self.assertEqual((plan['primary_steps'],plan['reserve_steps']),(10,4))
        self.assertEqual(len(plan['actions']),14)
        self.assertTrue(self.r.bounds.valid(plan['actions'],self.r.action))
        np.testing.assert_array_equal(plan['reference'][-4:],np.repeat(goal[None],4,axis=0))
        np.testing.assert_allclose(plan['prediction'],self.engine.rollout(plan['state'],plan['actions']))
        self.assertFalse(plan['qualified'])
        self.assertEqual(plan['planning_config']['max_total_steps'],14)

    def test_projected_target_reserve_has_sample_reference(self):
        self.r.ready=True
        goal=self.engine.observe(self.r.state,self.r.action);ids=np.array([5,6,7])
        matrix=segment_projection(self.engine.n_nodes,ids);samples=matrix@goal
        plan=self.r.plan_to_tolerance(goal,1.,3.,10,iterations=1,shooting_nfev=2,
            node_indices=ids,target_matrix=matrix,target_samples=samples,reserve_steps=3)
        np.testing.assert_allclose(plan['reference'][-3:],np.repeat(samples[None],3,axis=0))

    def test_short_reserve_can_be_revised_with_same_b_and_pressure_limits(self):
        self.r.initialize([30.]*6)
        held=np.tile(self.r.action,(6,1))
        reachable=self.r.bounds.project(np.tile([.45,.2,.35,.2],(6,1)),self.r.action)
        target=self.engine.rollout(self.r.state,reachable)[-1]
        reference=np.repeat(target[None],6,axis=0)
        candidate,info=fast_suffix_b(self.engine,self.r.state,held,self.r.action,reference,self.r.bounds)
        self.assertTrue(info['accepted']);self.assertTrue(self.r.bounds.valid(candidate,self.r.action))
        self.assertLess(info['mse_after_mm2'],info['mse_before_mm2'])

    def test_margin_limits_are_explicit_and_zero_default_preserved(self):
        self.r.ready=True;goal=self.engine.observe(self.r.state,self.r.action)
        plan=self.r.plan_to_tolerance(goal,iterations=1,max_horizon=10)
        self.assertEqual(len(plan['actions']),10);self.assertEqual(plan['reserve_steps'],0)
        for value in [-1,1.5,True,101,float('nan')]:
            with self.assertRaises(ValueError):self.r.plan_to_tolerance(goal,reserve_steps=value)
        self.r.meta['max_horizon']=200
        with self.assertRaisesRegex(ValueError,'200'):self.r.plan_to_tolerance(goal,max_horizon=200,reserve_steps=1)

    def test_open_loop_executes_exact_preview_and_archives_estimate(self):
        self.r.clock=time.monotonic;self.r.initialize([0.]*6);self.r.ready=True
        goal=self.engine.observe(self.r.state,self.r.action)
        plan=self.r.plan_to_tolerance(goal,1.,3.,2,iterations=1,reserve_steps=2)
        self.r.dt=.02
        transport=MockCommandTransport()
        frames=lambda:(np.zeros((12,12,3),np.uint8),time.monotonic())
        ex=HereditaryExecutor(self.r,transport,frames,use_correction=False)
        with patch('real_validation.execution.hereditary_executor.deadline_feedback',side_effect=AssertionError('OL cannot correct')):
            rows=ex.execute(plan)
        self.assertEqual(len(rows),4);self.assertEqual(self.r.phase,'final_hold')
        np.testing.assert_allclose([r.requested6 for r in rows],self.r.mapping.expand(plan['actions']))
        metadata=json.loads((ex.last_execution_dir/'metadata.json').read_text())
        self.assertEqual(metadata['reserve_steps'],2)
        self.assertFalse(metadata['completion_assessment']['physical_arrival_verified'])
        self.assertFalse(metadata['completion_assessment']['latest_image_supported'])
        steps=[json.loads(v) for v in (ex.last_execution_dir/'steps.jsonl').read_text().splitlines()]
        self.assertEqual([v['plan_phase'] for v in steps],['primary','primary','reserve','reserve'])

    def test_goal_miss_does_not_invalidate_geometry_or_force_zero(self):
        self.r.clock=time.monotonic;self.r.initialize([0.]*6);self.r.ready=True
        self.r.matrix=np.eye(3);self.r.alignment_confirmed=True;self.r.dt=.02
        current=self.engine.observe(self.r.state,self.r.action);goal=current.copy();goal[1:]+=100
        plan=dict(actions=np.zeros((2,4)),reference=np.repeat(goal[None],2,axis=0),
            goal=goal,version=self.r.version,tolerance=.1,max_node=.2)
        ex=HereditaryExecutor(self.r,MockCommandTransport(),lambda:(np.zeros((12,12,3),np.uint8),time.monotonic()),use_correction=False)
        ex.execute(plan)
        self.assertTrue(self.r.ready);self.assertTrue(self.r.alignment_confirmed)
        self.assertFalse(ex.completion_assessment['estimate_within_tolerance'])
        self.assertEqual(ex.completion_assessment['reason'],'finite_plan_exhausted')

    def test_ack_zero_and_rewarmup_keep_known_geometry_but_require_quality(self):
        self.r.clock=time.monotonic;self.r.initialize([0.]*6);self.r.ready=True
        self.r.matrix=np.eye(3);self.r.alignment_confirmed=True
        ex=HereditaryExecutor(self.r,MockCommandTransport(),lambda:None)
        ex.zero();self.assertFalse(self.r.ready)
        history=list(self.r.history);matrix=self.r.matrix.copy()
        self.r.prepare_rewarmup()
        self.assertFalse(self.r.ready);self.assertEqual(len(self.r.history),len(history))
        np.testing.assert_array_equal(self.r.matrix,matrix)
        self.r.fault='unknown ACK'
        with self.assertRaisesRegex(ValueError,'unknown ACK'):self.r.prepare_rewarmup()


if __name__=='__main__':unittest.main()
