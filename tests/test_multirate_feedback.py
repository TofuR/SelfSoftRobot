"""Focused changes only: scheduled commit, actual ACK replay, planning reserve."""
from collections import deque
from types import SimpleNamespace
import json
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import tests.test_hereditary_deployment as fixtures
from real_validation.execution.multirate_feedback import MultirateFeedback
from real_validation.execution.hereditary_executor import HereditaryExecutor
from real_validation.execution.executor import MockCommandTransport
from real_validation.runtime.hereditary_deployment import HereditaryDeployment,advance


class MultirateTest(unittest.TestCase):
    setUp=fixtures.DeploymentTests.setUp
    tearDown=fixtures.DeploymentTests.tearDown
    receipt=fixtures.DeploymentTests.receipt
    _flat_plan=fixtures.DeploymentTests._flat_plan

    def pending(self,*,blind=False,done=True):
        self.r.matrix=np.eye(3)
        snapshot,version=self.r.feedback_snapshot()
        posterior=snapshot.state.copy();posterior[-1]=.01
        snapshot.state=posterior;snapshot.at=100.01
        snapshot.history.append((100.01,posterior.copy(),snapshot.action.copy()))
        snapshot.last_frame=100.01;snapshot.version+=1
        old=np.zeros((2,4));ready=threading.Event()
        if done:ready.set()
        manager=MultirateFeedback(self.r,2,self.r.run_dir)
        manager.job=SimpleNamespace(source_step=0,apply_step=2,stamp=100.01,command_time=100.,
            deadline=100.197,started=100.01,finished=100.15,nominal=old.copy(),version=version,
            snapshot=snapshot,receipts=[],consumed=False,done=ready,error=None,
            result=(old.copy(),dict(observer=dict(count=0 if blind else 3),compute_ms=140.,control=dict(accepted=False))),
            path=self.r.run_dir/'proposal.json')
        ack=self.receipt(100.1,.01,'middle')
        self.r.acknowledge(ack);manager.acknowledge(1,ack);self.now=100.18
        reference=self.engine.rollout(self.r.state,old)
        return manager,old,reference,posterior,ack

    def test_delayed_state_uses_actual_ack_and_can_only_commit_once(self):
        manager,old,reference,posterior,ack=self.pending()
        previous=np.zeros(4);applied=self.r.mapping.reduce(ack.applied6)
        expected=advance(self.engine,posterior,previous,ack.t_command-100.01,self.r.dt)
        expected=advance(self.engine,expected,applied,0,self.r.dt)
        new,info=manager.take(2,old,reference,100.2,100.197)
        self.assertTrue(info['state_committed'])
        self.assertEqual(info['replayed_commands'],['middle'])
        np.testing.assert_allclose(self.r.state,expected)
        self.assertEqual(self.r.at,ack.t_command)
        self.assertEqual(self.r.last_frame,100.01)
        np.testing.assert_array_equal(new,old)
        version=self.r.version
        _,again=manager.take(2,old,reference,100.2,100.197)
        self.assertIsNone(again);self.assertEqual(self.r.version,version)

    def test_external_state_change_rejects_late_proposal(self):
        manager,old,reference,_,_=self.pending()
        self.r.version+=1;state=self.r.state.copy()
        new,info=manager.take(2,old,reference,100.2,100.197)
        self.assertEqual(info['revision_status'],'snapshot_changed')
        np.testing.assert_array_equal(self.r.state,state);np.testing.assert_array_equal(new,old)

    def test_unfinished_job_never_blocks_and_interval_step_is_not_a_failure(self):
        manager,old,reference,_,_=self.pending(done=False)
        info=manager.start(1,None,None,None,None,None,4)
        self.assertEqual(info['revision_status'],'feedback_interval');self.assertEqual(manager.skipped,0)
        state=self.r.state.copy();start=time.monotonic()
        _,info=manager.take(2,old,reference,100.2,100.197)
        self.assertLess(time.monotonic()-start,.05)
        self.assertEqual(info['revision_status'],'deadline_expired');self.assertEqual(manager.skipped,1)
        np.testing.assert_array_equal(self.r.state,state)
        manager.missed_sample(3);self.assertEqual(manager.skipped,1)
        manager.missed_sample(4);self.assertEqual(manager.skipped,2)

    def test_blind_observation_is_not_committed_or_counted_on_interval_steps(self):
        manager,old,reference,_,_=self.pending(blind=True)
        state=self.r.state.copy()
        _,info=manager.take(2,old,reference,100.2,100.197)
        self.assertEqual(info['revision_status'],'no_evidence');self.assertEqual(manager.missing,1)
        manager.start(3,None,None,None,None,None,5)
        self.assertEqual(manager.missing,1)
        np.testing.assert_array_equal(self.r.state,state)

    def test_130ms_compute_does_not_block_second_100ms_command(self):
        self.r.clock=time.monotonic;self.r.initialize([0.]*6);self.r.matrix=np.eye(3);self.r.ready=True
        plan=self._flat_plan(4)
        plan['reference']=self.engine.rollout(self.r.state,np.full((4,4),.02))
        plan['goal']=plan['reference'][-1]
        starts=[]
        def observation(snapshot,image,stamp,tail,ref):
            starts.append(time.monotonic());time.sleep(.13)
            z,u=snapshot.state_at(stamp);snapshot.state=z;snapshot.action=u;snapshot.at=stamp
            snapshot.history.append((stamp,z.copy(),u.copy()));snapshot.last_frame=stamp;snapshot.version+=1
            return tail,dict(observer=dict(count=2),edges=2,visibility=dict(coverage=1.))
        def control(engine,state,tail,previous,reference,bounds,**kwargs):
            return bounds.project(tail+.005,previous),dict(accepted=True,time_ms=0.)
        transport=MockCommandTransport()
        executor=HereditaryExecutor(self.r,transport,lambda:(np.zeros((12,12,3),np.uint8),time.monotonic()),feedback_interval_steps=2)
        with patch.object(HereditaryDeployment,'feedback',observation),patch('real_validation.execution.multirate_feedback.fast_suffix_b',control):
            try:receipts=executor.execute(plan)
            finally:
                if hasattr(self.r,'_feedback_job'):self.r._feedback_job.thread.join(1.)
        self.assertEqual(len(receipts),4);self.assertEqual(len(starts),2)
        intervals=np.diff([r.t_command for r in receipts])
        self.assertLess(intervals[0],.125)  # second command precedes 130 ms work completion
        np.testing.assert_array_equal(receipts[0].requested6,np.zeros(6))
        np.testing.assert_array_equal(receipts[1].requested6,np.zeros(6))
        self.assertGreater(max(receipts[2].requested6),0.)
        steps=[json.loads(line) for line in (executor.last_execution_dir/'steps.jsonl').read_text().splitlines()]
        self.assertEqual(steps[1]['revision_status'],'feedback_interval')
        self.assertEqual(steps[1]['consecutive_skipped'],0)
        self.assertEqual(steps[2]['feedback_application']['source_step'],0)
        self.assertTrue(steps[2]['feedback_application']['state_committed'])
        metadata=json.loads((executor.last_execution_dir/'metadata.json').read_text())
        self.assertTrue(metadata['terminal_feedback_application']['state_committed'])

    def test_planning_slew_reserve_does_not_lower_feedback_limits(self):
        self.r.ready=True;bounds=self.r.bounds
        goal=self.engine.observe(self.r.state,self.r.action);goal[1:]+=5.
        plan=self.r.plan_to_tolerance(goal,.01,.02,4,budget_s=.3,iterations=1,shooting_nfev=5,planning_rate_fraction=.8,reserve_steps=2)
        self.assertTrue(bounds.with_rate_fraction(.8).valid(plan['actions'],plan['previous']))
        self.assertIs(self.r.bounds,bounds)
        self.assertEqual(plan['planning_config']['planning_rate_fraction'],.8)
        np.testing.assert_allclose(plan['planning_config']['planning_rise_kpa_s'],40.)
        # 90% of physical slew is deliberately infeasible only for initial planning.
        action=(bounds.rise*.9)[None]
        self.assertTrue(bounds.valid(action,np.zeros(4)))
        self.assertFalse(bounds.with_rate_fraction(.8).valid(action,np.zeros(4)))

    def test_partial_auto_matching_keeps_planning_fraction(self):
        self.r.ready=True;shape=self.engine.observe(self.r.state,self.r.action)
        plan=self.r.plan_any_segment(shape[3:7],.2,.3,2,budget_s=.15,iterations=1,shooting_nfev=2,planning_rate_fraction=.6)
        self.assertEqual(plan['planning_config']['planning_rate_fraction'],.6)
        self.assertTrue(self.r.bounds.with_rate_fraction(.6).valid(plan['actions'],plan['previous']))

    def test_invalid_intervals_and_speed_fractions_are_rejected(self):
        for n in (0,1.5,True,11,float('nan')):
            with self.assertRaises(ValueError):HereditaryExecutor(self.r,None,None,feedback_interval_steps=n)
        for fraction in (0,-.1,1.1,True,float('nan')):
            with self.assertRaises(ValueError):self.r.bounds.with_rate_fraction(fraction)
