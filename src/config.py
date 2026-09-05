"""Central configuration: paths, seeds, column groups, validation folds.

Everything that another module might want to hardcode lives here instead.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"
CONFIGS = EXPERIMENTS / "configs"
SUBMISSIONS = ROOT / "submissions"

TRAIN_CSV = DATA_RAW / "train.csv"
TEST_CSV = DATA_RAW / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_RAW / "sample_submission.csv"

EXPERIMENT_LOG = EXPERIMENTS / "log.csv"

for _d in (DATA_PROCESSED, CONFIGS, SUBMISSIONS):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------
TARGET = "PM2_5_next_hour"
ID = "id"
TIME = "observation_timestamp"
GROUP = "station"

#: Pollutant and weather measurements present in BOTH train and test.
POLLUTANTS = ["PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
NUMERIC_RAW = POLLUTANTS + WEATHER
CATEGORICAL_RAW = ["station", "wd"]

#: Calendar columns supplied in the CSVs (redundant with TIME, kept for parity).
CALENDAR_RAW = ["year", "month", "day", "hour"]

#: The full set of columns legally available at prediction time. Any feature
#: must be derivable from these alone -- see LEAKY_SOURCES below.
AVAILABLE_AT_TEST = [ID, TIME] + CALENDAR_RAW + NUMERIC_RAW + CATEGORICAL_RAW

#: Columns that exist only in train. Deriving a feature from any of these is a
#: leak, because the test set has no PM2.5 history at all (see PLAN.md Fact 1).
LEAKY_SOURCES = [TARGET]

#: Fixed category orderings, shared across train and test so that the integer
#: codes a model sees are identical in both. Order is arbitrary but must be
#: stable -- do not sort these differently later.
WD_CATEGORIES = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]
STATION_CATEGORIES = [
    "Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan", "Gucheng",
    "Huairou", "Nongzhanguan", "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong",
]

#: Compass bearing in degrees for each wind direction, used for the u/v wind
#: vector features in Phase 5 Tier B. Derived from the labels themselves.
WD_DEGREES = {name: i * 22.5 for i, name in enumerate(WD_CATEGORIES)}

# --------------------------------------------------------------------------
# Target bounds (observed in train; used for post-processing clips)
# --------------------------------------------------------------------------
TARGET_MIN = 2.0
TARGET_MAX = 999.0

# --------------------------------------------------------------------------
# Validation folds -- "seasonal analogue" design (PLAN.md Phase 3)
#
# The hidden test set is a single contiguous Sep-Feb block immediately after
# the training period, so each fold mimics that: train on everything before a
# September, validate on the following Sep-Feb.
# --------------------------------------------------------------------------
FOLDS = {
    # Same-season analogues: validate on a Sep-Feb block, exactly like test.
    "A": {"train_end": "2014-09-01", "val_start": "2014-09-01", "val_end": "2015-03-01"},
    "B": {"train_end": "2015-09-01", "val_start": "2015-09-01", "val_end": "2016-03-01"},
    # Thin third analogue: only 183 days (52k rows) to train on. MEASURED
    # baseline RMSE 49.81 vs 32.96 on fold B -- so starved of data that it
    # ranks changes differently. Do not use it to make decisions.
    "C": {"train_end": "2013-09-01", "val_start": "2013-09-01", "val_end": "2014-03-01"},
    # Most recent data, but the WRONG season (Mar-Aug, mean target 64 vs 83).
    # Measured baseline RMSE 23.11 -- summer is simply an easier problem. Use
    # only to check a change is not winter-specific; never as a primary score.
    "R": {"train_end": "2016-03-01", "val_start": "2016-03-01", "val_end": "2016-09-01"},
}

#: Fold B is the primary metric -- most training data and the closest analogue
#: to the real task. Fold A is a stability check.
PRIMARY_FOLD = "B"

#: Folds used when a caller does not specify. A+B are the trustworthy pair.
DEFAULT_FOLDS = ["A", "B"]

#: Out-of-fold predictions, kept for Phase 9 ensembling.
OOF_DIR = DATA_PROCESSED / "oof"
OOF_DIR.mkdir(parents=True, exist_ok=True)

#: Local-vs-leaderboard correlation tracker (Phase 3 Task 3.3).
TRACKER_CSV = EXPERIMENTS / "leaderboard_tracker.csv"

#: The real test period, for reference in EDA and drift checks.
TEST_PERIOD = ("2016-08-31 23:00:00", "2017-02-28 22:00:00")
