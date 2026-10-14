import unittest

import numpy as np
import pandas as pd

from data.solete import FEATURES, clean


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


if __name__ == "__main__":
    unittest.main()
