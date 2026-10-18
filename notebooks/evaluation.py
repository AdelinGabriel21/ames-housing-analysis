"""Evaluation of the Ames price-class model (Lab 2): holdout splits and k-fold cross-validation.

- The splits are drawn once and hold index labels of houses, so different data preparation
  variants can be fitted on the same splits and scored on exactly the same test houses
  (paired comparison).
- Every result table has one row per split or fold, which makes them easy to stack and compare.
"""

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# The evaluation setups of notebook 02: test size or number of folds, with and without stratification
HOLDOUT_SETUPS = [(test_size, stratified) for test_size in (0.10, 0.20) for stratified in (False, True)]
CV_SETUPS = [(k, stratified) for k in (5, 10) for stratified in (False, True)]
HOLDOUT_SEEDS = range(10)
CV_SEED = 42


def setup_name(experiment, size, stratified):
    """Readable name of an evaluation setup, e.g. "Holdout 10%, stratified" or "CV k=5"."""
    size_text = f"{size:.0%}" if experiment == "Holdout" else f"k={size}"
    return f"{experiment} {size_text}" + (", stratified" if stratified else "")


def make_splits(y, holdout_seeds=HOLDOUT_SEEDS, cv_seed=CV_SEED):
    """Draw all holdout and cross-validation splits of the houses in y (as index labels)."""
    labels = y.index.to_numpy()
    splits = []

    for test_size, stratified in HOLDOUT_SETUPS:
        for seed in holdout_seeds:
            train, test = train_test_split(
                labels, test_size=test_size, random_state=seed, stratify=y if stratified else None
            )
            splits.append({"Setup": setup_name("Holdout", test_size, stratified), "Split": seed,
                           "train": train, "test": test})

    for k, stratified in CV_SETUPS:
        cv_class = StratifiedKFold if stratified else KFold
        folds = cv_class(n_splits=k, shuffle=True, random_state=cv_seed).split(labels, y)
        for fold, (train, test) in enumerate(folds):
            splits.append({"Setup": setup_name("CV", k, stratified), "Split": fold,
                           "train": labels[train], "test": labels[test]})
    return splits


def evaluate(pipeline, X, y, splits, extra_train=None):
    """Fit a fresh copy of the pipeline on every split and return its train and test scores."""
    rows = []
    for split in splits:
        # Houses in extra_train are only ever used for training, never scored as test houses
        train = split["train"] if extra_train is None else [*split["train"], *extra_train]
        test = split["test"]
        model = clone(pipeline).fit(X.loc[train], y.loc[train])

        train_pred, test_pred = model.predict(X.loc[train]), model.predict(X.loc[test])
        rows.append({
            "Setup": split["Setup"],
            "Split": split["Split"],
            "Train Accuracy": accuracy_score(y.loc[train], train_pred),
            "Train Macro-F1": f1_score(y.loc[train], train_pred, average="macro"),
            "Test Accuracy": accuracy_score(y.loc[test], test_pred),
            "Test Macro-F1": f1_score(y.loc[test], test_pred, average="macro"),
        })
    return pd.DataFrame(rows)


def summarize(results, metric="Test Macro-F1", by="Variant"):
    """Mean ± standard deviation of a score over the splits, per group (rows) and setup (columns)."""
    grouped = results.groupby([by, "Setup"], sort=False)[metric]
    cells = grouped.mean().map("{:.4f}".format) + " ± " + grouped.std().map("{:.4f}".format)
    return cells.unstack("Setup")


def paired_differences(results, baseline, metric="Test Macro-F1"):
    """Score of every variant minus the baseline's score on the same split (positive = better)."""
    wide = results.pivot_table(index=["Setup", "Split"], columns="Variant", values=metric, sort=False)
    wide = wide[results["Variant"].unique()]  # keep the order in which the variants were run
    return wide.drop(columns=baseline).sub(wide[baseline], axis=0)
