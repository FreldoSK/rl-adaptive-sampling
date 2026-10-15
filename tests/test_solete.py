import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data.solete import FEATURES, clean, read_columns


class CleanTest(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2019-01-01", periods=11, freq="min")
        self.df = pd.DataFrame({c: np.linspace(1, 2, 11) for c in FEATURES}, index=idx)
        for c in FEATURES:
            self.df[f"{c}_qc_worst"] = 0
        self.df["Pressure[mbar]"] = np.nan
        self.df["P_Gaia[kW]"] = 0.0

    def test_keeps_only_features(self):
        out, counts = clean(self.df)
        self.assertEqual(list(out.columns), FEATURES)
        self.assertEqual(counts, {})

    def test_flagged_and_missing_values_are_interpolated(self):
        df = self.df.copy()
        df.loc[df.index[3], "HUMIDITY[%]"] = 0.0          # long dropout kept as 0
        df.loc[df.index[3], "HUMIDITY[%]_qc_worst"] = 4
        df.loc[df.index[5], "WIND_SPEED[m1s]"] = np.nan   # out of bound -> NaN
        df.loc[df.index[5], "WIND_SPEED[m1s]_qc_worst"] = 9
        df.loc[df.index[7], "WIND_DIR[deg]_qc_worst"] = 1  # wrapped: already fixed, keep
        out, counts = clean(df)
        np.testing.assert_allclose(out.to_numpy(), self.df[FEATURES].to_numpy())
        self.assertEqual(counts, {"HUMIDITY[%]": 1, "WIND_SPEED[m1s]": 1})


    def test_per_value_qc_columns_of_1sec_file(self):
        df = self.df.rename(columns={f"{c}_qc_worst": f"{c}_qc" for c in FEATURES})
        df.loc[df.index[2], "HUMIDITY[%]"] = 0.0
        df.loc[df.index[2], "HUMIDITY[%]_qc"] = 4
        df.loc[df.index[4], "TEMPERATURE[degC]_qc"] = 5   # already fixed by the author, keep
        out, counts = clean(df)
        np.testing.assert_allclose(out.to_numpy(), self.df[FEATURES].to_numpy())
        self.assertEqual(counts, {"HUMIDITY[%]": 1})


class ReadColumnsTest(unittest.TestCase):
    def test_matches_read_hdf_for_selected_columns(self):
        idx = pd.date_range("2019-01-01", periods=1000, freq="s")
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=1000), "b_qc_worst": rng.integers(0, 9, 1000),
                           "c": rng.normal(size=1000), "d": rng.integers(0, 3, 1000).astype("int8")},
                          index=idx)
        for fmt in ("fixed", "table"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "f.h5"
                df.to_hdf(path, key="DATA", format=fmt)
                out = read_columns(path, ["c", "b_qc_worst", "missing", "a"], chunk_rows=64)
                pd.testing.assert_frame_equal(out, df[["c", "b_qc_worst", "a"]], check_freq=False)


if __name__ == "__main__":
    unittest.main()
