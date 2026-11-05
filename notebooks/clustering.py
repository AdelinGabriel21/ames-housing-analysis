"""k-means clustering of the reduced Ames Housing data (Lab 3): repeated loops and plots.

- The scaling, the columns to log and the number of clusters are chosen in the notebooks. The
  functions here take them as arguments and hold no such decision.
- A variance threshold is a number of leading components of one PCA fit, so reduced[:, :n] is the
  dataset of that threshold and its first two columns are PC1 and PC2 at every threshold.
- Result tables have one column per threshold ("98%") and one row per number of clusters.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.metrics.cluster import contingency_matrix

THRESHOLDS = [0.98, 0.95, 0.90, 0.80]
N_INIT = 10
SEED = 42

BLUE, ORANGE = "#86b6ef", "#eb6834"
MUTED, GRID, AXIS = "#898781", "#e1e0d9", "#c3c2b7"
INK = "#2b2a28"

# Okabe-Ito colors, chosen to stay apart for color-blind readers. The sixth one only appears in
# silhouette plots with six clusters, where the clusters are also separated by gaps
CLUSTER_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]

# The thresholds have a natural order, so they share one hue from dark to light
THRESHOLD_COLORS = dict(zip(THRESHOLDS, ["#104281", "#256abf", "#5598e7", BLUE]))

# Color shows the scaling and a dashed line the log transform, so that pairs read as pairs
PREPARATION_STYLES = {
    "Min-max": (CLUSTER_COLORS[0], "-"),
    "Z-score": (ORANGE, "-"),
    "Min-max + log": (CLUSTER_COLORS[0], "--"),
    "Z-score + log": (ORANGE, "--"),
}


def reduce_to_thresholds(scaled, thresholds=THRESHOLDS):
    """Fit PCA once and return it, the reduced data and the component count of each threshold."""
    # One fit is enough: the components kept at a lower threshold are the first ones of this fit
    pca = PCA(n_components=max(thresholds), svd_solver="full")
    reduced = pca.fit_transform(scaled)
    cumulative = pca.explained_variance_ratio_.cumsum()
    component_counts = {t: int(np.searchsorted(cumulative, t) + 1) for t in thresholds}
    return pca, reduced, component_counts


def inertia_table(reduced, component_counts, cluster_range, n_init=N_INIT, seed=SEED):
    """Inertia of a k-means model for each number of clusters (rows) and threshold (columns)."""
    table = {
        f"{t:.0%}": [
            KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(reduced[:, :count]).inertia_
            for k in cluster_range
        ]
        for t, count in component_counts.items()
    }
    return pd.DataFrame(table, index=pd.Index(cluster_range, name="Clusters"))


def silhouette_table(reduced, component_counts, cluster_range, n_init=N_INIT, seed=SEED):
    """Average silhouette for each number of clusters (rows) and threshold (columns)."""
    table = {
        f"{t:.0%}": [
            silhouette_score(
                reduced[:, :count],
                KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(reduced[:, :count]),
            )
            for k in cluster_range
        ]
        for t, count in component_counts.items()
    }
    return pd.DataFrame(table, index=pd.Index(cluster_range, name="Clusters"))


def relabel(model, reference):
    """Renumber the clusters of a fitted model to overlap the reference labels as much as possible."""
    overlap = contingency_matrix(reference, model.labels_)
    reference_ids, own_ids = linear_sum_assignment(overlap, maximize=True)
    new_ids = np.empty_like(own_ids)
    new_ids[own_ids] = reference_ids
    centers = np.empty_like(model.cluster_centers_)
    centers[reference_ids] = model.cluster_centers_[own_ids]
    # Labels and centroids change together, so the model still predicts the new numbers
    model.labels_ = new_ids[model.labels_]
    model.cluster_centers_ = centers
    return model


def fit_models(reduced, component_counts, n_clusters, n_init=N_INIT, seed=SEED):
    """Fit one k-means model per threshold, all numbered like the model of the highest threshold."""
    models = {
        t: KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed).fit(reduced[:, :count])
        for t, count in component_counts.items()
    }
    reference = models[max(models)].labels_
    return {t: relabel(model, reference) for t, model in models.items()}


def agreement_table(labels):
    """Adjusted Rand index between every pair of named groupings (1 = same grouping, 0 = chance)."""
    names = list(labels)
    values = [[adjusted_rand_score(labels[a], labels[b]) for b in names] for a in names]
    return pd.DataFrame(values, index=names, columns=names)


def variance_by_type(data, column_types):
    """Share of the total variance of data held by each named group of columns."""
    variance = data.var()
    return pd.Series({name: variance[columns].sum() / variance.sum() for name, columns in column_types.items()})


def style_axes(ax, grid="y"):
    """Apply the chart style of the Lab 3 notebooks: light grid, only the bottom axis line."""
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", length=0)


def plot_elbow(inertia, title):
    """Draw the inertia curve of each threshold, labeled at its right end."""
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for (label, curve), color in zip(inertia.items(), THRESHOLD_COLORS.values()):
        ax.plot(curve.index, curve, color=color, linewidth=2, marker="o", markersize=6, label=label)
        ax.annotate(
            f"{label} of the variance",
            (curve.index[-1], curve.iloc[-1]),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            color=INK,
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Inertia (sum of squared distances to the centroids)")
    ax.set_xticks(inertia.index)
    ax.set_xlim(inertia.index[0] - 0.3, inertia.index[-1] + 1.7)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.legend(title="Variance kept", frameon=False, loc="upper right")
    style_axes(ax)
    plt.tight_layout()
    plt.show()


def plot_silhouettes(reduced, component_counts, cluster_range, title, n_init=N_INIT, seed=SEED):
    """Draw one silhouette plot per threshold (rows) and number of clusters (columns)."""
    fig, axes = plt.subplots(
        len(component_counts),
        len(cluster_range),
        figsize=(3.3 * len(cluster_range), 3.1 * len(component_counts)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for column, k in enumerate(cluster_range):
        models = fit_models(reduced, component_counts, k, n_init, seed)
        for row, (threshold, count) in enumerate(component_counts.items()):
            ax = axes[row, column]
            labels = models[threshold].labels_
            values = silhouette_samples(reduced[:, :count], labels)
            bottom = 0
            for cluster in range(k):
                cluster_values = np.sort(values[labels == cluster])
                positions = np.arange(bottom, bottom + len(cluster_values))
                ax.fill_betweenx(positions, 0, cluster_values, color=CLUSTER_COLORS[cluster], linewidth=0)
                middle = bottom + len(cluster_values) / 2
                ax.text(0.02, middle, cluster + 1, transform=ax.get_yaxis_transform(), va="center", fontsize=9)
                bottom += len(cluster_values) + 60
            ax.axvline(values.mean(), color=INK, linewidth=1.2, linestyle="--")
            ax.axvline(0, color=MUTED, linewidth=0.8)
            ax.set_title(f"{threshold:.0%}, {k} clusters: average {values.mean():.3f}", fontsize=10)
            ax.set_yticks([])
            style_axes(ax, grid="x")
        axes[-1, column].set_xlabel("Silhouette value")
    for ax in axes[:, 0]:
        ax.set_ylabel("Houses, sorted inside each cluster")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_clusters(panels, title, shared_axes=True):
    """Draw PC1 against PC2 per panel, colored by cluster, with the centroids as black crosses.

    panels maps each panel title to its (fitted PCA, reduced data, fitted k-means model).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 11.5), sharex=shared_axes, sharey=shared_axes)
    for ax, (name, (pca, reduced, model)) in zip(axes.flat, panels.items()):
        ax.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=np.array(CLUSTER_COLORS)[model.labels_],
            s=9,
            alpha=0.5,
            linewidths=0,
        )
        centers = model.cluster_centers_[:, :2]
        ax.scatter(
            centers[:, 0], centers[:, 1], marker="X", s=210, c="black", edgecolors="white", linewidths=1.6, zorder=3
        )
        for number, center in enumerate(centers, start=1):
            ax.annotate(
                number,
                center,
                xytext=(9, 9),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color=INK,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                zorder=4,
            )
        first, second = pca.explained_variance_ratio_[:2]
        ax.set_title(name, fontsize=12)
        ax.set_xlabel(f"PC1 ({first:.1%} of the variance)")
        ax.set_ylabel(f"PC2 ({second:.1%} of the variance)")
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)

    handles = [
        Line2D([], [], marker="o", linestyle="", color=CLUSTER_COLORS[i], markersize=8, label=f"Cluster {i + 1}")
        for i in range(len(model.cluster_centers_))
    ]
    handles.append(
        Line2D(
            [], [], marker="X", linestyle="", markerfacecolor="black", markeredgecolor="white", markersize=12,
            label="Centroid",
        )
    )
    fig.suptitle(title, fontsize=13)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    plt.show()


def plot_variance_shares(shares, title):
    """Draw one stacked bar per row of shares, with the column shares written inside the segments."""
    colors = dict(zip(shares.columns, [BLUE, ORANGE, AXIS]))
    fig, ax = plt.subplots(figsize=(10.5, 1.1 * len(shares) + 1.8))
    left = np.zeros(len(shares))
    for name, values in shares.items():
        bars = ax.barh(
            shares.index, values, left=left, height=0.62, color=colors[name], edgecolor="white", linewidth=2, label=name
        )
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], label_type="center", color=INK, fontsize=10)
        left = left + values.to_numpy()
    ax.invert_yaxis()
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Share of the total variance")
    ax.set_ylabel("Preparation")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0%}"))
    ax.legend(title="Column type", frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(colors))
    style_axes(ax, grid="x")
    plt.tight_layout()
    plt.show()


def plot_preparation_curves(curves, title, xlabel, ylabel):
    """Draw one line per preparation, with color for the scaling and dashes for the log transform."""
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for name, curve in curves.items():
        color, linestyle = PREPARATION_STYLES[name]
        ax.plot(curve.index, curve, color=color, linestyle=linestyle, linewidth=2, label=name)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="Preparation", frameon=False)
    style_axes(ax)
    return fig, ax
