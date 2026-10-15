"""Download and load the SOLETE dataset, version 4 (DTU Data / figshare).

Source: D. V. Pombo, "The SOLETE dataset", Technical University of Denmark,
https://doi.org/10.11583/DTU.17040767 (v4, MIT licence).

The release is a single 4 GB ZIP holding every resolution. Instead of
downloading all of it, the ZIP is read over HTTP range requests and only the
requested ``.h5`` member is extracted into ``data/raw/`` (not tracked by git),
then checked against the SHA256 published in the release's SHA256SUMS.txt.

Cleaning follows the author's own QC flags, codes in ``solete/qc_codes.py``
of https://github.com/DVPombo/SOLETE. The 1-second file has one flag per
value (``<column>_qc``); resampled files keep the most severe flag of each
bucket (``<column>_qc_worst``), and ``_qc`` only where no such column exists.

Only the environmental measurements of the weather station are used
(``FEATURES``). Left out are plant outputs (``P_Solar``, ``P_Gaia``, ...),
the plane-of-array irradiance ``POA Irr`` (tied to the PV installation),
the computed sun position (``Azimuth``, ``Elevation``) and
``Pressure[mbar]``, which is a placeholder (code 2 -> NaN) on 99.9% of rows.

* Values flagged 4 (long dropout, zero kept as recorded) or 9 (out of bound,
  set to NaN) are replaced by linear interpolation in time, so the series
  stays on a regular grid (Hankel rows assume consecutive steps).
* Other flags are kept as recorded: 1 (wind direction wrapped), 3 and 5
  (short dropout / glitch already interpolated by the author).
"""

import hashlib
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from data.base import Dataset

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
    "TEMPERATURE[degC]", "HUMIDITY[%]", "WIND_SPEED[m1s]", "WIND_DIR[deg]", "GHI[kW1m2]",
]

# QC codes whose value is missing or not a real measurement.
QC_INTERPOLATE = (4, 9)
# Flag column suffixes in order of preference (resampled files, 1-second file).
QC_SUFFIXES = ("_qc_worst", "_qc")


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
        bad = out[col].isna()
        for flag_col in QC_SUFFIXES:
            if f"{col}{flag_col}" in df:
                bad |= df[f"{col}{flag_col}"].isin(QC_INTERPOLATE)
                break
        if bad.any():
            out[col] = out[col].mask(bad)
            counts[col] = int(bad.sum())
    out = out.interpolate(method="time", limit_direction="both")
    return out, counts


def read_columns(path, columns, key="DATA", chunk_rows=1 << 20):
    """Read only ``columns`` of a pandas HDF frame, chunk by chunk.

    ``pd.read_hdf`` loads every column, which for the 1-second file (~39M
    rows x 36 columns) needs ~11 GB. The release mixes both pandas formats:
    the 1-second file is a ``table`` (read with ``HDFStore.select`` in row
    chunks), the others are ``fixed``, which store one 2-D
    ``block<i>_values`` array per dtype that the wanted columns are copied
    out of. Columns missing from the file are skipped.
    """
    import tables

    with pd.HDFStore(path, "r") as store:
        storer = store.get_storer(key)
        if storer.is_table:
            present = [c for c in columns if c in storer.non_index_axes[0][1]]
            return pd.concat(store.select(key, columns=present, chunksize=chunk_rows))

    with tables.open_file(path) as f:
        node = f.get_node(f"/{key}")
        kind = node.axis1._v_attrs.kind  # datetime64[ns] in the release, [us] from pandas >= 3
        if node._v_attrs.pandas_type != "frame" or not kind.startswith("datetime64"):
            raise ValueError(f"{path}: expected a fixed-format frame with a datetime index")
        index = node.axis1.read()
        n = len(index)
        out = {}
        for b in range(node._v_attrs.nblocks):
            items = [c.decode() for c in getattr(node, f"block{b}_items").read()]
            wanted = [(c, items.index(c)) for c in columns if c in items]
            if not wanted:
                continue
            values = getattr(node, f"block{b}_values")
            cols = {c: np.empty(n, dtype=values.dtype) for c, _ in wanted}
            for start in range(0, n, chunk_rows):
                chunk = values[start:start + chunk_rows]
                for c, j in wanted:
                    cols[c][start:start + len(chunk)] = chunk[:, j]
            out.update(cols)
    return pd.DataFrame({c: out[c] for c in columns if c in out}, index=pd.DatetimeIndex(index.view(kind)))


def load_solete(resolution="1min", path=None):
    """Load the cleaned v4 data as ``(df, counts)``, sorted by time.

    ``path`` reads an already downloaded v4 ``.h5`` file instead of
    fetching ``resolution``.
    """
    path = fetch(resolution) if path is None else Path(path)
    columns = FEATURES + [f"{c}{s}" for c in FEATURES for s in QC_SUFFIXES]
    df = read_columns(path, columns).sort_index()
    return clean(df)


class SoleteDataset(Dataset):
    """SOLETE v4 at ``resolution``, cleaned by ``load_solete``, in physical units.

    ``path`` reads an already downloaded v4 ``.h5`` file instead of fetching.
    ``interpolated`` maps each column to the number of QC-interpolated values.
    """

    def __init__(self, resolution="1min", path=None):
        self.resolution, self.path = resolution, path
        super().__init__()

    def _load(self):
        df, self.interpolated = load_solete(self.resolution, self.path)
        return df
