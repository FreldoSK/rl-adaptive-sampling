import unittest

import numpy as np
import pandas as pd

from data.base import Dataset


class DummyDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame
        super().__init__()

    def _load(self):
        return self.frame


class DatasetContractTest(unittest.TestCase):
    def setUp(self):
        self.idx = pd.date_range("2019-01-01", periods=6, freq="min")
        self.frame = pd.DataFrame({"a": np.arange(6), "b": np.linspace(0, 1, 6, dtype="float32")},
                                  index=self.idx)
        self.ds = DummyDataset(self.frame)

    def test_cannot_instantiate_base_class(self):
        with self.assertRaises(TypeError):
            Dataset()

    def test_len_is_number_of_time_steps(self):
        self.assertEqual(len(self.ds), 6)

    def test_features_timestamps_and_values(self):
        self.assertEqual(self.ds.features, ["a", "b"])
        pd.testing.assert_index_equal(self.ds.timestamps, self.idx)
        self.assertEqual(self.ds.values.dtype, np.float64)
        self.assertEqual(self.ds.values.shape, (6, 2))
        np.testing.assert_array_equal(self.ds.values, self.frame.to_numpy(np.float64))

    def test_getitem_single_step(self):
        x = self.ds[2]
        self.assertEqual(x.shape, (2,))
        self.assertEqual(x.dtype, np.float64)
        np.testing.assert_array_equal(x, self.frame.iloc[2].to_numpy(np.float64))
        np.testing.assert_array_equal(self.ds[-1], self.ds.values[5])

    def test_getitem_several_steps(self):
        self.assertEqual(self.ds[1:4].shape, (3, 2))
        np.testing.assert_array_equal(self.ds[[0, 5]], self.ds.values[[0, 5]])

    def test_getitem_indexes_time_steps_only(self):
        with self.assertRaises(IndexError):
            self.ds[:, 0]

    def test_values_are_read_only(self):
        with self.assertRaises(ValueError):
            self.ds.values[0, 0] = 1.0
        with self.assertRaises(ValueError):
            self.ds[0][0] = 1.0

    def test_rejects_non_datetime_index(self):
        with self.assertRaises(TypeError):
            DummyDataset(self.frame.reset_index(drop=True))

    def test_rejects_unsorted_index(self):
        with self.assertRaises(ValueError):
            DummyDataset(self.frame.iloc[::-1])


if __name__ == "__main__":
    unittest.main()
