"""Data preparation for the Ames Housing price-class model (Lab 2).

- Rule steps (load_raw ... prepare_data) are deterministic and work per house, so they run once
  on all houses before any train/test split.
- Learned steps (make_preprocessor, build_pipeline) fit encoders on data, so they are returned
  unfitted and are fitted only on the training part inside a scikit-learn Pipeline.
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder
from xgboost import XGBClassifier

TARGET = "SalePrice"
ID_COLS = ["Order", "PID"]

# Feature types according to the data documentation (column names as in the CSV file)
NOMINAL = [
    "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood",
    "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl",
    "Exterior 1st", "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air",
    "Garage Type", "Misc Feature", "Sale Type", "Sale Condition",
]
ORDINAL = [
    "Lot Shape", "Utilities", "Land Slope", "Overall Qual", "Overall Cond", "Exter Qual",
    "Exter Cond", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2",
    "Heating QC", "Electrical", "Kitchen Qual", "Functional", "Fireplace Qu", "Garage Finish",
    "Garage Qual", "Garage Cond", "Paved Drive", "Pool QC", "Fence",
]
DISCRETE = [
    "Year Built", "Year Remod/Add", "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath", "Half Bath",
    "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd", "Fireplaces", "Garage Yr Blt",
    "Garage Cars", "Mo Sold", "Yr Sold",
]
CONTINUOUS = [
    "Lot Frontage", "Lot Area", "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
    "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Low Qual Fin SF", "Gr Liv Area", "Garage Area",
    "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch", "Pool Area",
    "Misc Val",
]

# Ordinal grades -> integers in their natural order (0 = feature not present).
# Overall Qual and Overall Cond are already integers, so they need no mapping.
QUALITY = {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
FINISH_TYPE = {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
ORDINAL_MAPS = {
    "Lot Shape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "Utilities": {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
    "Land Slope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "Exter Qual": QUALITY,
    "Exter Cond": QUALITY,
    "Bsmt Qual": QUALITY,
    "Bsmt Cond": QUALITY,
    "Bsmt Exposure": {"NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFin Type 1": FINISH_TYPE,
    "BsmtFin Type 2": FINISH_TYPE,
    "Heating QC": QUALITY,
    "Electrical": {"Mix": 0, "FuseP": 1, "FuseF": 2, "FuseA": 3, "SBrkr": 4},
    "Kitchen Qual": QUALITY,
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "Fireplace Qu": QUALITY,
    "Garage Finish": {"NA": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Garage Qual": QUALITY,
    "Garage Cond": QUALITY,
    "Paved Drive": {"N": 0, "P": 1, "Y": 2},
    "Pool QC": QUALITY,
    "Fence": {"NA": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
}

# Almost every house has the same value here (e.g. 2,927 of 2,930 have "AllPub" utilities)
NEAR_CONSTANT = ["Utilities", "Street", "Condition 2", "Roof Matl", "Heating", "Pool QC", "Misc Feature"]

# Dropped feature -> kept feature that measures the same thing (Spearman |r| >= 0.8, Exterior 2nd
# equals Exterior 1st for 85% of the houses); the kept one is the more detailed measurement.
# 1st Flr SF also follows exactly from Gr Liv Area - 2nd Flr SF - Low Qual Fin SF
REDUNDANT = {
    "Garage Yr Blt": "Year Built",
    "Garage Cars": "Garage Area",
    "TotRms AbvGrd": "Gr Liv Area",
    "1st Flr SF": "Total Bsmt SF",
    "Exterior 2nd": "Exterior 1st",
}

# Price classes: < 100k, 100k–130k, 130k–180k, 180k–250k, 250k–400k, ≥ 400k
PRICE_BINS = [0, 100_000, 130_000, 180_000, 250_000, 400_000, np.inf]
CLASS_NAMES = ["< 100k", "100k–130k", "130k–180k", "180k–250k", "250k–400k", "≥ 400k"]

# The five houses above 4,000 ft² living area (see Lab 1, Plot 6)
EXTREME_AREA = 4000

# How the nominal features reach XGBoost: one column per category, one integer per feature,
# price-class shares per category, or XGBoost's own category handling
STRATEGIES = ("onehot", "ordinal", "target", "native")


# ---------------------------------------------------------------------------------------------
# Rule steps: run once on all houses, before any split
# ---------------------------------------------------------------------------------------------

def load_raw(path):
    """Read the tab-separated raw file; only empty cells are missing, "NA" stays a category."""
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])


def clean(df):
    """Fill structural gaps and fix typos; all other NaNs stay, XGBoost handles them."""
    df = df.copy()

    # No basement -> empty basement areas and bathroom counts become 0
    no_basement = df["Bsmt Qual"] == "NA"
    bsmt_numeric = ["BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF", "Bsmt Full Bath", "Bsmt Half Bath"]
    df.loc[no_basement, bsmt_numeric] = df.loc[no_basement, bsmt_numeric].fillna(0)

    # Unknown masonry veneer -> no veneer, by far the most frequent case
    df["Mas Vnr Type"] = df["Mas Vnr Type"].fillna("None")
    df["Mas Vnr Area"] = df["Mas Vnr Area"].fillna(0)

    # Typo in the raw data: a garage built in 2207
    df["Garage Yr Blt"] = df["Garage Yr Blt"].replace(2207, 2007)

    # MS SubClass codes (20, 60, ...) name dwelling types, they are not quantities
    df["MS SubClass"] = df["MS SubClass"].astype(str)
    return df


def add_totals(df):
    """Add the total area, bathroom count and porch area, which a tree cannot sum in one split."""
    # A house without basement values keeps NaN in Total SF and Total Bathrooms
    df = df.copy()
    df["Total SF"] = df["Total Bsmt SF"] + df["Gr Liv Area"]
    df["Total Bathrooms"] = (
        df["Full Bath"] + df["Bsmt Full Bath"] + 0.5 * (df["Half Bath"] + df["Bsmt Half Bath"])
    )
    porches = ["Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch"]
    df["Total Porch SF"] = df[porches].sum(axis=1)
    return df


def add_ages(df):
    """Add the age of the house and of its last remodeling at the time of sale."""
    # A few partial sales happen before the house or remodeling is finished -> age 0
    df = df.copy()
    df["Age At Sale"] = (df["Yr Sold"] - df["Year Built"]).clip(lower=0)
    df["Years Since Remod"] = (df["Yr Sold"] - df["Year Remod/Add"]).clip(lower=0)
    return df


def encode_ordinal(df):
    """Map the ordinal grades to integers in their natural order (see ORDINAL_MAPS)."""
    # Central Air (Y/N) is not mapped here, it stays a nominal feature
    df = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        mapped = df[col].map(mapping)
        # A grade missing from the dict would silently become NaN
        unknown = set(df[col].dropna()) - set(mapping)
        assert mapped.isna().sum() == df[col].isna().sum(), f"{col}: values not in ORDINAL_MAPS: {unknown}"
        df[col] = mapped
    return df


def to_category(df):
    """Cast every nominal feature to a pandas category with a fixed, sorted category list."""
    # A rule step, not a learned one: the category names carry no price information, and the
    # fixed list gives the same codes in every split, which the "native" strategy relies on
    df = df.copy()
    for col in NOMINAL:
        categories = sorted(df[col].dropna().unique())
        df[col] = df[col].astype(pd.CategoricalDtype(categories))
    return df


def make_target(price):
    """Bin the sale price into the six price classes 0–5 (names in CLASS_NAMES)."""
    # right=False makes the intervals [a, b), matching "< 100000" and ">= 400000"
    return pd.cut(price, PRICE_BINS, right=False, labels=False).astype(int)


def split_xy(df):
    """Split into the features X (without target and ids) and the price classes y."""
    y = make_target(df[TARGET])
    X = df.drop(columns=[TARGET, *ID_COLS])
    return X, y


def prepare_data(path, drop_extreme=False, drop_near_constant=False, add_size_totals=False,
                 add_age_features=False, drop_redundant=False):
    """Run all rule steps on the raw file and return the features X and price classes y."""
    df = clean(load_raw(path))
    if drop_extreme:
        df = df[df["Gr Liv Area"] <= EXTREME_AREA]
    if add_size_totals:
        df = add_totals(df)
    if add_age_features:
        df = add_ages(df)
    df = to_category(encode_ordinal(df))
    if drop_near_constant:
        df = df.drop(columns=NEAR_CONSTANT)
    if drop_redundant:
        df = df.drop(columns=list(REDUNDANT))
    X, y = split_xy(df)

    # XGBoost only accepts numeric and category columns (pandas 3 reads text as dtype str)
    bad = [col for col in X if not (is_numeric_dtype(X[col]) or isinstance(X[col].dtype, pd.CategoricalDtype))]
    assert not bad, f"Columns that are neither numeric nor category: {bad}"
    return X, y


# ---------------------------------------------------------------------------------------------
# Learned steps: returned unfitted, fitted on the training part only
# ---------------------------------------------------------------------------------------------

def make_preprocessor(strategy, min_frequency=5):
    """Return the unfitted encoding step for the nominal (category) columns."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")

    # XGBoost reads the category columns itself (enable_categorical=True)
    if strategy == "native":
        return "passthrough"

    encoders = {
        # One 0/1 column per category; categories rarer than min_frequency share one
        # "infrequent" column, and unseen categories in new data land there too
        "onehot": OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency, sparse_output=False),
        # One integer per category (rare ones share a code); unseen categories become NaN
        "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, min_frequency=min_frequency),
        # Per category the smoothed share of each price class (6 columns per feature);
        # fitting uses internal cross-fitting, so a house never sees its own price.
        # The seed goes into cv because TargetEncoder(random_state=...) is deprecated since 1.9
        "target": TargetEncoder(target_type="multiclass", cv=StratifiedKFold(5, shuffle=True, random_state=42)),
    }
    # The selector picks whatever category columns are left, e.g. after drop_near_constant
    return ColumnTransformer(
        [("nominal", encoders[strategy], make_column_selector(dtype_include="category"))],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def build_pipeline(strategy, min_frequency=5, **xgb_params):
    """Return the unfitted pipeline: nominal encoding followed by an XGBoost classifier."""
    # Define default model settings
    model_params = {
        "enable_categorical": True,
        "random_state": 42,
    }
    # Allow any user-provided parameters to override defaults
    model_params.update(xgb_params)

    return Pipeline([
        ("prep", make_preprocessor(strategy, min_frequency)),
        ("model", XGBClassifier(**model_params)),
    ])
