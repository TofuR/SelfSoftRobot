import csv
import unittest

from scripts.evaluation.watch_best_checkpoint import validation_node_mean


class PeriodicEvaluatorTest(unittest.TestCase):
    def test_validation_node_mean_uses_prediction_rows_and_native_unit(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "per_frame.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("t", "is_prediction", "node_mean_mm",
                                "node_mean_est_mm"))
                writer.writeheader()
                writer.writerow({"t": 0, "is_prediction": 0, "node_mean_mm": "",
                                 "node_mean_est_mm": ""})
                writer.writerow({"t": 1, "is_prediction": 1, "node_mean_mm": 2.0,
                                 "node_mean_est_mm": 2.0})
                writer.writerow({"t": 2, "is_prediction": 1, "node_mean_mm": 4.0,
                                 "node_mean_est_mm": 4.0})

            score, unit = validation_node_mean(path)

        self.assertAlmostEqual(score, 3.0)
        self.assertEqual(unit, "mm")


if __name__ == "__main__":
    unittest.main()
