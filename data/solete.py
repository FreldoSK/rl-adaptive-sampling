"""Download and load the SOLETE dataset, version 4 (DTU Data / figshare).

Source: D. V. Pombo, "The SOLETE dataset", Technical University of Denmark,
https://doi.org/10.11583/DTU.17040767 (v4, MIT licence).

The release is a single 4 GB ZIP holding every resolution. Instead of
downloading all of it, the ZIP is read over HTTP range requests and only the
requested ``.h5`` member is extracted into ``data/raw/`` (not tracked by git),
then checked against the SHA256 published in the release's SHA256SUMS.txt.

Cleaning follows the author's own QC flags (``<column>_qc_worst``, codes in
``solete/qc_codes.py`` of https://github.com/DVPombo/SOLETE):

* ``Pressure[mbar]`` is dropped: code 2 (placeholder -> NaN) on 99.9% of rows.
* ``P_Gaia[kW]`` is dropped: code 7 (provenance unverified) on every row and
  almost always 0, so two active days would dominate the error tail.
* Values flagged 4 (long dropout, zero kept as recorded) or 9 (out of bound,
  set to NaN) are replaced by linear interpolation in time, so the series
  stays on a regular grid (Hankel rows assume consecutive steps).
"""

import hashlib
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ZIP_URL = "https://ndownloader.figshare.com/files/69897429"  # SOLETEdataset.zip, v4
RAW_DIR = Path(__file__).resolve().parent / "raw"

# From data/SHA256SUMS.txt inside the v4 ZIP.
SHA256 = {
    "1sec": "8dcbe44fe453ea56037eeb05c4eb6df2129a2d680bd92429b03a88ab1e69ed76",
    "1min": "5f8ecba8fef143050909f335925f8b59a5a3c3a09fcd2a4bac0df349e8320d24",
    "5min": "35f41a5f72043d476b9e37a7796d86a164686d41a58a17c557790ff58b6a9a2c",
    "60min": "2f9b6d39774be322ecd69ca526f21ee792a92dcaa78f52608e0f8869d6296fd2",
}

FEATURES = [
    "TEMPERATURE[degC]", "HUMIDITY[%]", "WIND_SPEED[m1s]", "WIND_DIR[deg]",
    "GHI[kW1m2]", "POA Irr[kW1m2]", "P_Solar[kW]", "Azimuth[deg]", "Elevation[deg]",
]

# QC codes whose value is missing or not a real measurement.
QC_INTERPOLATE = (4, 9)


class _HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, for zipfile.

    Every read re-requests the figshare URL, because it redirects to a
    signed S3 link that expires after a few seconds.
    """

    def __init__(self, url):
        self.url, self.pos = url, 0
        with self._get(0, 0) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def _get(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        return urllib.request.urlopen(req, timeout=120)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buf):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        with self._get(self.pos, end) as r:
            data = r.read()
        buf[:len(data)] = data
        self.pos += len(data)
        return len(data)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(resolution="1min"):
    """Return the local path of ``SOLETE_Pombo_<resolution>_v4.h5``.

    Downloads it on first use; later calls only verify the cached file.
    """
    if resolution not in SHA256:
        raise ValueError(f"resolution must be one of {list(SHA256)}")
    name = f"SOLETE_Pombo_{resolution}_v4.h5"
    path = RAW_DIR / name
    if path.exists() and _sha256(path) == SHA256[resolution]:
        return path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"downloading {name} from {ZIP_URL} ...")
    remote = io.BufferedReader(_HTTPRangeFile(ZIP_URL), buffer_size=8 << 20)
    with zipfile.ZipFile(remote) as zf, zf.open(f"data/hdf5/{name}") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, length=8 << 20)  # zipfile also checks CRC32

    digest = _sha256(tmp)
    if digest != SHA256[resolution]:
        tmp.unlink()
        raise RuntimeError(f"{name}: SHA256 {digest} does not match the release")
    tmp.replace(path)
    return path


def clean(df):
    """Select ``FEATURES`` and interpolate values the author flagged as unusable.

    Returns ``(cleaned_df, counts)``; ``counts`` maps each column to the
    number of interpolated values.
    """
    out = df[FEATURES].copy()
    counts = {}
    for col in FEATURES:
        flag_col = f"{col}_qc_worst"
        bad = out[col].isna()
        if flag_col in df:
            bad |= df[flag_col].isin(QC_INTERPOLATE)
        if bad.any():
            out[col] = out[col].mask(bad)
            counts[col] = int(bad.sum())
    out = out.interpolate(method="time", limit_direction="both")
    return out, counts


def load_solete(resolution="1min", path=None):
    """Load the cleaned v4 data as ``(df, counts)``, sorted by time.

    ``path`` reads an already downloaded v4 ``.h5`` file instead of
    fetching ``resolution``.
    """
    path = fetch(resolution) if path is None else Path(path)
    df = pd.read_hdf(path, key="DATA").sort_index()
    return clean(df)
