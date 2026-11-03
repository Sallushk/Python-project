"""Meaningful charts with error bars, colour bars, and polished styling."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
from data_processing import SMOKING_ORDER, add_age_bands
from statistical_analysis import proportion_table, correlation_tables

COLORS = ["#287A9B", "#CA6044"]
OUTCOME_LABELS = ["No recorded stroke", "Recorded stroke"]
ERROR_FRACTION = 0.05          # 5 % relative error bar on every bar chart


# ── helpers ──────────────────────────────────────────────────

def prepare_style():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def save_and_show(figure, folder, filename, show=True):
    os.makedirs(folder, exist_ok=True)
    figure.tight_layout()
    figure.savefig(os.path.join(folder, filename), dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def _add_colorbar(figure, axis, values, cmap_name="Blues", label="Value"):
    """Attach a colour-mapped scalar bar beside a bar chart."""
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
    sm = cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    figure.colorbar(sm, ax=axis, fraction=0.04, pad=0.02, label=label)
    return norm, sm


def _bar_colors(values, cmap_name="Blues"):
    """Return per-bar colours scaled to the value range."""
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
    cmap = plt.get_cmap(cmap_name)
    return [cmap(norm(v)) for v in values]


# ── individual chart functions ────────────────────────────────

def draw_outcome_counts(data, axis, figure=None):
    counts = data["stroke"].value_counts().reindex([0, 1], fill_value=0).to_numpy()
    errors = counts * ERROR_FRACTION
    colors = _bar_colors(counts, "Blues")
    bars = axis.bar(OUTCOME_LABELS, counts, color=colors, width=0.6,
                    yerr=errors, capsize=6, error_kw={"ecolor": "#555", "elinewidth": 1.5})
    for pos, (cnt, err) in enumerate(zip(counts, errors)):
        axis.text(pos, cnt + err + max(counts) * 0.03,
                  f"{cnt}\n({100 * cnt / len(data):.1f}%)", ha="center", fontsize=9)
    axis.set_ylim(0, max(counts) * 1.30)
    axis.set(xlabel="Recorded outcome", ylabel="Number of records",
             title=f"All-age outcome counts  (n={len(data)})")
    if figure is not None:
        _add_colorbar(figure, axis, counts, "Blues", "Record count")


def draw_age_histogram(adults, axis):
    upper = 18 + 5 * (int((adults["age"].max() - 18) / 5) + 1)
    bins = np.arange(18, upper + 1, 5)
    for stroke in [0, 1]:
        group = adults.loc[adults["stroke"] == stroke, "age"]
        axis.hist(group, bins=bins, density=True, histtype="stepfilled", alpha=0.35,
                  color=COLORS[stroke])
        axis.hist(group, bins=bins, density=True, histtype="step", linewidth=2.2,
                  color=COLORS[stroke],
                  label=f"{OUTCOME_LABELS[stroke]} (n={len(group)})")
    axis.set(title="Adult age distributions by recorded outcome",
             xlabel="Age (years)", ylabel="Density within each outcome group")
    axis.legend(fontsize=9)


def draw_boxplot(adults, column, label, axis):
    observed = adults.dropna(subset=[column]).copy()
    observed["outcome_label"] = observed["stroke"].map(
        {0: OUTCOME_LABELS[0], 1: OUTCOME_LABELS[1]})
    sns.boxplot(data=observed, x="outcome_label", y=column, order=OUTCOME_LABELS,
                hue="outcome_label", hue_order=OUTCOME_LABELS,
                palette=COLORS, legend=False, width=0.5, fliersize=2.5, ax=axis)
    # overlay mean ± 5 % error markers
    for idx, stroke in enumerate([0, 1]):
        vals = observed.loc[observed["stroke"] == stroke, column].dropna()
        if len(vals) > 0:
            mean_val = vals.mean()
            axis.errorbar(idx, mean_val, yerr=mean_val * ERROR_FRACTION,
                          fmt="D", color="white", markeredgecolor="#333",
                          ecolor="#333", elinewidth=1.5, capsize=5, zorder=5,
                          label="Mean ± 5%" if idx == 0 else "")
    counts = observed["stroke"].value_counts().reindex([0, 1], fill_value=0)
    axis.set_xticks([0, 1], [f"No recorded stroke\n(n={counts.loc[0]})",
                             f"Recorded stroke\n(n={counts.loc[1]})"])
    axis.set(xlabel="Recorded outcome", ylabel=label,
             title=f"Adult {column.replace('avg_glucose_level', 'glucose')}: "
                   f"observed values; outliers retained")
    axis.legend(fontsize=8, loc="upper right")


def draw_proportions(table, column, title, axis, figure=None, labels=None,
                     cmap_name="Blues"):
    if labels is None:
        labels = table[column].astype(str).tolist()
    values = table["stroke_percent"].to_numpy()
    errors = values * ERROR_FRACTION
    bar_cols = _bar_colors(values, cmap_name)
    bars = axis.bar(labels, values, color=bar_cols, width=0.65,
                    yerr=errors, capsize=6,
                    error_kw={"ecolor": "#555", "elinewidth": 1.5})
    for pos, bar in enumerate(bars):
        row = table.iloc[pos]
        err = errors[pos]
        axis.annotate(
            f"{row['stroke_percent']:.2f}%\n"
            f"{int(row['stroke_records'])}/{int(row['total_records'])}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height() + err),
            xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)
    axis.set_ylim(0, max(5, (max(values) + max(errors)) * 1.40 + 1))
    axis.set(title=title, xlabel=column.replace("_", " ").capitalize(),
             ylabel="Recorded stroke  (% within category)")
    if figure is not None:
        _add_colorbar(figure, axis, values, cmap_name, "Stroke %")


def draw_bmi_missingness(adults, axis, figure=None):
    bmi_rows = [
        {"outcome": OUTCOME_LABELS[s],
         "missing_percent": 100 * adults.loc[adults["stroke"] == s, "bmi"].isna().mean(),
         "missing": int(adults.loc[adults["stroke"] == s, "bmi"].isna().sum()),
         "total": int((adults["stroke"] == s).sum())}
        for s in [0, 1]
    ]
    vals = [r["missing_percent"] for r in bmi_rows]
    errors = [v * ERROR_FRACTION for v in vals]
    bar_cols = _bar_colors(vals, "Reds")
    bars = axis.bar([r["outcome"] for r in bmi_rows], vals,
                    color=bar_cols, yerr=errors, capsize=6,
                    error_kw={"ecolor": "#555", "elinewidth": 1.5})
    for bar, row, err in zip(bars, bmi_rows, errors):
        axis.annotate(
            f"{row['missing_percent']:.2f}%\n{row['missing']}/{row['total']}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height() + err),
            xytext=(0, 6), textcoords="offset points", ha="center")
    axis.set_ylim(0, max(5, (max(vals) + max(errors)) * 1.45))
    axis.set(title="Adult BMI missingness differs by recorded outcome",
             xlabel="Recorded outcome", ylabel="BMI missing  (% within outcome group)")
    if figure is not None:
        _add_colorbar(figure, axis, vals, "Reds", "Missing %")


# ── main entry point ──────────────────────────────────────────

def generate_all_charts(populations, folder, show=True):
    prepare_style()
    adults, all_ages = populations["Adults"], populations["All ages"]

    # 01 – outcome counts with colour bar
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_outcome_counts(all_ages, ax, figure=fig)
    save_and_show(fig, folder, "01_outcome_counts_all_ages.png", show)

    # 02 – age histogram (filled + step, no error bar on density)
    fig, ax = plt.subplots(figsize=(9, 5))
    draw_age_histogram(adults, ax)
    save_and_show(fig, folder, "02_age_distribution_adults.png", show)

    # 03 – glucose boxplot + mean ± 5 %
    fig, ax = plt.subplots(figsize=(9, 6))
    draw_boxplot(adults, "avg_glucose_level",
                 "Average glucose (recorded units; unverified)", ax)
    save_and_show(fig, folder, "03_glucose_boxplot_adults.png", show)

    # 04 – BMI boxplot + mean ± 5 %
    fig, ax = plt.subplots(figsize=(9, 6))
    draw_boxplot(adults, "bmi", "BMI (conventional kg/m²)", ax)
    save_and_show(fig, folder, "04_bmi_boxplot_observed_adults.png", show)

    # 05 – hypertension proportions + error bars + colour bar
    hypertension = proportion_table(adults, "hypertension").sort_values("hypertension")
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_proportions(hypertension, "hypertension",
                     "Adult stroke proportions by hypertension",
                     ax, figure=fig, labels=["Absent (0)", "Present (1)"],
                     cmap_name="YlOrRd")
    save_and_show(fig, folder, "05_hypertension_proportions_adults.png", show)

    # 06 – smoking proportions + error bars + colour bar
    smoking = proportion_table(adults, "smoking_status")
    smoking = smoking.set_index("smoking_status").reindex(SMOKING_ORDER).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    draw_proportions(smoking, "smoking_status",
                     "Adult smoking categories: stroke / category total",
                     ax, figure=fig, cmap_name="YlOrRd")
    save_and_show(fig, folder, "06_smoking_proportions_adults.png", show)

    # 07 – Spearman heatmap with colour bar (already has cbar_kws)
    _, spearman, _ = correlation_tables(adults)
    pretty_names = ["Age", "Glucose", "BMI", "Hypertension", "Heart disease", "Stroke (0/1)"]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(spearman, vmin=-1, vmax=1, center=0, cmap="vlag",
                annot=True, fmt=".2f", square=True, linewidths=0.5,
                xticklabels=pretty_names, yticklabels=pretty_names,
                cbar_kws={"label": "Spearman rank correlation", "shrink": 0.8},
                ax=ax)
    ax.set(title="Adult numerical correlations (available observed pairs)",
           xlabel="Numerical variable", ylabel="Numerical variable")
    ax.tick_params(axis="x", labelrotation=40)
    save_and_show(fig, folder, "07_spearman_heatmap_adults.png", show)

    # 08 – age-band proportions + error bars + colour bar
    age_bands = proportion_table(add_age_bands(adults), "age_band")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    draw_proportions(age_bands, "age_band",
                     "Adult stroke proportions by age band",
                     ax, figure=fig, cmap_name="YlOrRd")
    ax.set_xlabel("Age band (years)")
    save_and_show(fig, folder, "08_age_band_proportions_adults.png", show)

    # 09 – smoking counts bar chart with colour bar
    smoking_counts = adults["smoking_status"].value_counts().reindex(SMOKING_ORDER, fill_value=0)
    vals_09 = smoking_counts.to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bar_cols_09 = _bar_colors(vals_09, "Blues")
    bars_09 = ax.bar(SMOKING_ORDER, vals_09, color=bar_cols_09,
                     yerr=vals_09 * ERROR_FRACTION, capsize=5,
                     error_kw={"ecolor": "#555", "elinewidth": 1.5})
    for bar, cnt in zip(bars_09, vals_09):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + cnt * ERROR_FRACTION + 15,
                str(int(cnt)), ha="center")
    ax.set_ylim(0, max(vals_09) * 1.25)
    ax.set(title="All adult smoking categories, including Unknown",
           xlabel="Recorded smoking category", ylabel="Number of adult records")
    _add_colorbar(fig, ax, vals_09, "Blues", "Record count")
    save_and_show(fig, folder, "09_smoking_counts_adults.png", show)

    # 10 – BMI missingness with colour bar
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_bmi_missingness(adults, ax, figure=fig)
    save_and_show(fig, folder, "10_bmi_missingness_adults.png", show)

    # 11 – 2 × 2 dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Adult recorded stroke comparisons  |  observed data  |  unadjusted groups",
                 fontsize=16, fontweight="bold")
    draw_age_histogram(adults, axes[0, 0])
    draw_boxplot(adults, "avg_glucose_level",
                 "Average glucose (recorded units)", axes[0, 1])
    draw_boxplot(adults, "bmi", "BMI (conventional kg/m²)", axes[1, 0])
    draw_proportions(hypertension, "hypertension",
                     "Stroke proportions by hypertension", axes[1, 1],
                     figure=fig, labels=["Absent (0)", "Present (1)"],
                     cmap_name="YlOrRd")
    save_and_show(fig, folder, "11_adult_dashboard.png", show)

    print(f"Saved 11 charts (with 5% error bars & colour bars) to {folder}")
