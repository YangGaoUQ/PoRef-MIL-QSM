import unittest

from poref.aggregation import sliding_window_mean


class WindowMeanTests(unittest.TestCase):
    def test_ten_slice_windows_with_stride_one(self):
        result = sliding_window_mean([0.1] * 20 + [0.8] * 10 + [0.1] * 98,
                                     window_length=10, window_threshold=0.5)
        self.assertEqual(len(result["window_scores"]), 119)
        self.assertEqual(result["best_window_slices"], list(range(21, 31)))
        self.assertEqual(result["prediction"], 1)
        self.assertAlmostEqual(result["score"], 0.8)

    def test_local_abnormal_window_overrides_healthy_global_mean(self):
        result = sliding_window_mean([0.1, 0.1, 0.9, 0.9, 0.1, 0.1],
                                     window_length=2, window_threshold=0.5)
        self.assertEqual(result["prediction"], 1)
        self.assertAlmostEqual(result["score"], 0.9)
        self.assertEqual(result["best_window_slices"], [3, 4])
        self.assertEqual(len(result["window_scores"]), 5)

    def test_threshold_boundary_and_full_length_window(self):
        for values, expected in [([0.2, 0.8], 0), ([0.3, 0.9], 1), ([0.1, 0.2], 0)]:
            result = sliding_window_mean(values, window_length=2, window_threshold=0.5)
            self.assertEqual(result["prediction"], expected)
        self.assertEqual(sliding_window_mean([0.7], window_length=1,
                                            window_threshold=0.8)["prediction"], 0)

    def test_rejects_invalid_windows_and_probabilities(self):
        for length in [0, 3, 1.5, None]:
            with self.assertRaises(ValueError):
                sliding_window_mean([0.2, 0.8], window_length=length, window_threshold=0.5)
        for values in [[], [float("nan")], [1.1]]:
            with self.assertRaises(ValueError):
                sliding_window_mean(values, window_length=1, window_threshold=0.5)


if __name__ == "__main__":
    unittest.main()
