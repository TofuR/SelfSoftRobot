import unittest
from unittest import mock

import numpy as np
import torch

from src.evaluation.ishsm_evaluation import rollout_ishsm_sequence
from src.models.model_ishsm import ISHSMModel
from scripts.evaluation.eval_ishsm import ishsm_evaluation_protocols


class ISHSMEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.model = ISHSMModel(
            action_dim=4, n_nodes=15, n_bend_modes=8,
            section_intervals=(7, 7))
        self.model.set_normalization(np.zeros(3), np.ones(3), 1.0)
        self.actions = np.zeros((6, 4), np.float32)
        self.positions = np.zeros((6, 3, 15), np.float32)
        self.positions[:, 0, :] = np.arange(15, dtype=np.float32)

    def test_periodic_two_reanchors_before_steps_1_3_5(self):
        with mock.patch.object(
                self.model, "init_rollout_state",
                wraps=self.model.init_rollout_state) as initialize, \
                mock.patch.object(
                    self.model, "assimilate_observation",
                    wraps=self.model.assimilate_observation) as assimilate:
            result = rollout_ishsm_sequence(
                self.model, self.actions, self.positions,
                protocol="periodic", reanchor_interval=2,
                window_size=4, norm_factor=1.0,
                device=torch.device("cpu"))
        self.assertEqual(initialize.call_count, 1)
        self.assertEqual(assimilate.call_count, 3)
        self.assertIsNone(
            assimilate.call_args_list[0].kwargs.get("predicted_state"))
        self.assertTrue(all(
            call.args[2] is not None for call in assimilate.call_args_list[1:]))
        self.assertTrue(np.array_equal(result["horizon"], [-1, 0, 1, 0, 1, 0]))

    def test_anchor_index_starts_single_and_periodic_protocol_at_scoring_boundary(self):
        with mock.patch.object(
                self.model, "init_rollout_state",
                wraps=self.model.init_rollout_state) as initialize:
            result = rollout_ishsm_sequence(
                self.model, self.actions, self.positions,
                protocol="single_anchor", anchor_index=2,
                window_size=4, norm_factor=1.0,
                device=torch.device("cpu"))

        initialize.assert_called_once()
        self.assertTrue(np.array_equal(
            result["horizon"], [-1, -1, -1, 0, 1, 2]))
        observed = initialize.call_args.args[1]
        expected = torch.from_numpy(self.positions[2].T).unsqueeze(0)
        self.assertTrue(torch.equal(observed.cpu(), expected))

        periodic = rollout_ishsm_sequence(
            self.model, np.zeros((8, 4), np.float32),
            np.concatenate([self.positions, self.positions[-2:]], axis=0),
            protocol="periodic", anchor_index=2, reanchor_interval=2,
            window_size=4, norm_factor=1.0, device=torch.device("cpu"))
        self.assertTrue(np.array_equal(
            periodic["horizon"], [-1, -1, -1, 0, 1, 0, 1, 0]))

    def test_h0_keeps_all_dynamic_states_zero(self):
        result = rollout_ishsm_sequence(
            self.model, self.actions, self.positions, protocol="h0",
            window_size=4, norm_factor=1.0, device=torch.device("cpu"))
        self.assertTrue(np.all(result["states"][1:] == 0.0))

    def test_periodic_40_is_not_described_as_a_history_window(self):
        protocols = ishsm_evaluation_protocols()
        by_name = {name: (protocol, interval)
                   for name, protocol, interval in protocols}
        self.assertEqual(by_name["periodic_40"], ("periodic", 40))
        self.assertNotIn("window40", by_name)


if __name__ == "__main__":
    unittest.main()
