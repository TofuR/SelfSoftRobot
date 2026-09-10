import unittest

from real_validation.gui.feedback_timing import computation_text,timing_line


class FeedbackTimingDisplayTest(unittest.TestCase):
    def test_missing_measurements_are_not_zero_or_reported_as_computation(self):
        info=dict(step=0,revision_status='frame_missing',feedback_budget_ms=0.,
                  ack_ms=24.,ack_delivery_ms=110.,frame_wait_ms=0.)
        text=timing_line(info)
        self.assertIn('等待反馈 — / 可用预算 0.0 ms',text)
        self.assertIn('ACK 后唤醒 110.0 ms',text)
        self.assertIn('本步未计算',text)
        self.assertIn('未获得 ACK 后新图像',text)

    def test_late_worker_wait_is_not_its_full_compute_time(self):
        info=dict(revision_status='deadline_expired',feedback_wait_ms=12.,feedback_budget_ms=12.)
        self.assertIn('等待反馈 12.0 ms / 可用预算 12.0 ms',timing_line(info))
        self.assertIn('待后台完成',computation_text(info))
        self.assertNotIn('计算 12.0 ms',timing_line(info))

    def test_completed_compute_and_wait_are_distinct(self):
        info=dict(revision_status='committed',compute_ms=30.,feedback_wait_ms=32.,feedback_budget_ms=60.)
        self.assertEqual(computation_text(info),'计算 30.0 ms')
        self.assertIn('等待反馈 32.0 ms / 可用预算 60.0 ms',timing_line(info))
