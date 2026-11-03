"""Syllabus-covered summaries, Welch tests, chi-square tests and rank checks."""

import os
import numpy as np
import pandas as pd
from scipy import stats
from data_processing import CONTINUOUS_COLUMNS, FACTOR_NAMES, KNOWN_SMOKING, add_age_bands

ALPHA = 0.05
CORRELATION_COLUMNS = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease", "stroke"]


def describe_values(values):
    """Sample standard deviation (ddof=1) on observed measurements."""
    n = values.dropna().to_numpy(dtype=float)
    return {
        "n": len(n),
        "mean": np.mean(n) if len(n) > 0 else np.nan,
        "median": np.median(n) if len(n) > 0 else np.nan,
        "sample_std": np.std(n, ddof=1) if len(n) > 1 else np.nan,
        "minimum": np.min(n) if len(n) > 0 else np.nan,
        "maximum": np.max(n) if len(n) > 0 else np.nan,
    }


def descriptive_table(populations):
    OUTCOMES = {"All records": None, "No recorded stroke": 0, "Recorded stroke": 1}
    rows = [
        {"population": pop, "factor": col, "outcome": label,
         "missing_n": int((data if val is None else data.loc[data["stroke"] == val])[col].isna().sum()),
         **describe_values((data if val is None else data.loc[data["stroke"] == val])[col])}
        for pop, data in populations.items()
        for col in CONTINUOUS_COLUMNS
        for label, val in OUTCOMES.items()
    ]
    return pd.DataFrame(rows)


def proportion_table(data, column):
    """Denominator is all records IN EACH category, not the whole dataset."""
    t = data.groupby(column, observed=True)["stroke"].agg(["count", "sum", "mean"])
    t = t.rename(columns={"count": "total_records", "sum": "stroke_records"})
    t["stroke_percent"] = t["mean"] * 100
    return t.drop(columns=["mean"]).reset_index()


def conclusion(p_value, kind):
    if not np.isfinite(p_value):
        return "Test unavailable: insufficient observations or undefined statistic."
    if p_value < ALPHA:
        return f"Reject H0 at alpha=0.05: evidence of {kind}."
    return f"Do not reject H0 at alpha=0.05: insufficient evidence of {kind}."


def continuous_test(data, column, population, rank_test=False):
    stroke    = data.loc[data["stroke"] == 1, column].dropna().to_numpy(dtype=float)
    no_stroke = data.loc[data["stroke"] == 0, column].dropna().to_numpy(dtype=float)
    row = {
        "population": population, "factor": column, "alpha": ALPHA,
        "n_stroke": len(stroke), "n_no_stroke": len(no_stroke),
        "statistic": np.nan, "p_value_unadjusted": np.nan,
        "mean_stroke": np.mean(stroke) if len(stroke) > 0 else np.nan,
        "mean_no_stroke": np.mean(no_stroke) if len(no_stroke) > 0 else np.nan,
        "mean_difference_stroke_minus_no_stroke": (
            np.mean(stroke) - np.mean(no_stroke)
            if len(stroke) > 0 and len(no_stroke) > 0 else np.nan
        )
    }
    if rank_test:
        row.update(test="Mann-Whitney U (two-sided, asymptotic)", role="Supplementary",
                   H0=f"Outcome groups have the same distribution of {FACTOR_NAMES[column]}.",
                   H1=f"Outcome groups differ in the distribution of {FACTOR_NAMES[column]}.",
                   why_selected="Rank-based check of skewed values/outliers.")
        if len(stroke) > 0 and len(no_stroke) > 0:
            r = stats.mannwhitneyu(stroke, no_stroke, alternative="two-sided", method="asymptotic")
            row["statistic"], row["p_value_unadjusted"] = r.statistic, r.pvalue
        row["conclusion"] = conclusion(row["p_value_unadjusted"], "a distribution/rank difference")
    else:
        row.update(test="Welch independent-sample t-test (two-sided)",
                   role="Primary" if population == "Adults" else "Sensitivity",
                   H0=f"Outcome groups have equal means for {FACTOR_NAMES[column]}.",
                   H1=f"Outcome groups have unequal means for {FACTOR_NAMES[column]}.",
                   why_selected="Continuous measurement; unequal variances allowed.")
        if len(stroke) >= 2 and len(no_stroke) >= 2:
            r = stats.ttest_ind(stroke, no_stroke, equal_var=False, alternative="two-sided")
            row["statistic"], row["p_value_unadjusted"] = r.statistic, r.pvalue
        row["conclusion"] = conclusion(row["p_value_unadjusted"], "a mean difference")
    return row


def categorical_test(data, column, population):
    test_data = data.loc[data[column].isin(KNOWN_SMOKING)] if column == "smoking_status" else data
    observed = pd.crosstab(test_data[column], test_data["stroke"]).reindex(columns=[0, 1], fill_value=0)
    row = {
        "population": population, "factor": column,
        "test": "Pearson chi-square (no continuity correction)",
        "role": "Primary" if population == "Adults" else "Sensitivity",
        "H0": f"{FACTOR_NAMES[column]} and stroke are independent.",
        "H1": f"{FACTOR_NAMES[column]} and stroke are associated.",
        "alpha": ALPHA, "statistic": np.nan, "p_value_unadjusted": np.nan,
        "degrees_of_freedom": np.nan, "min_expected_count": np.nan,
        "n_stroke": int(test_data["stroke"].sum()),
        "n_no_stroke": int((test_data["stroke"] == 0).sum()),
        "why_selected": "Association between categorical variables; check expected cell counts."
    }
    expected = pd.DataFrame()
    if len(observed) >= 2 and (observed.sum(axis=0) > 0).all():
        stat, p, dof, exp = stats.chi2_contingency(observed, correction=False)
        row.update(statistic=stat, p_value_unadjusted=p, degrees_of_freedom=dof,
                   min_expected_count=np.min(exp))
        expected = pd.DataFrame(exp, index=observed.index, columns=observed.columns)
    row["conclusion"] = conclusion(row["p_value_unadjusted"], "an association")
    if np.isfinite(row["min_expected_count"]) and row["min_expected_count"] < 5:
        row["conclusion"] = "Expected cell < 5: chi-square approximation requires caution."
    return row, observed, expected


def primary_tests(data, population):
    rows = [continuous_test(data, col, population) for col in CONTINUOUS_COLUMNS]
    for col in ["hypertension", "smoking_status"]:
        row, _, _ = categorical_test(data, col, population)
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_tables(adults):
    nums = adults[CORRELATION_COLUMNS]
    pair_counts = pd.DataFrame(
        {c2: {c1: int(adults[[c1, c2]].notna().all(axis=1).sum()) for c1 in CORRELATION_COLUMNS}
         for c2 in CORRELATION_COLUMNS}
    )
    return nums.corr("pearson"), nums.corr("spearman"), pair_counts.astype(int)


def build_tables(populations):
    adults = populations["Adults"]
    adult_tests = primary_tests(adults, "Adults")
    sensitivity = pd.concat(
        [adult_tests,
         primary_tests(populations["All ages"], "All ages"),
         primary_tests(populations["Complete adults"], "Complete adults")],
        ignore_index=True
    )
    pearson, spearman, pair_counts = correlation_tables(adults)
    tables = {
        "descriptive_statistics": descriptive_table(populations),
        "primary_tests_adults": adult_tests,
        "sensitivity_tests_unadjusted": sensitivity,
        "mann_whitney_adults": pd.DataFrame(
            [continuous_test(adults, col, "Adults", rank_test=True) for col in CONTINUOUS_COLUMNS]
        ),
        "hypertension_proportions_adults": proportion_table(adults, "hypertension"),
        "smoking_proportions_adults": proportion_table(adults, "smoking_status"),
        "age_band_proportions_adults": proportion_table(add_age_bands(adults), "age_band"),
        "population_summary": pd.DataFrame([
            {"population": pop, "total_records": len(d), "stroke_records": int(d["stroke"].sum()),
             "stroke_percent": 100 * d["stroke"].mean(),
             "observed_bmi_n": int(d["bmi"].notna().sum()),
             "known_smoking_n": int(d["smoking_status"].isin(KNOWN_SMOKING).sum())}
            for pop, d in populations.items()
        ]),
        "categorical_proportions_all_populations": pd.concat([
            proportion_table(d, col).rename(columns={col: "category"}).assign(factor=col, population=pop)
            for pop, d in populations.items()
            for col in ["hypertension", "smoking_status"]
        ], ignore_index=True),
        "pearson_correlations_adults": pearson.reset_index(names="variable"),
        "spearman_correlations_adults": spearman.reset_index(names="variable"),
        "correlation_pair_counts_adults": pair_counts.reset_index(names="variable"),
    }
    for col in ["hypertension", "smoking_status"]:
        row, obs, exp = categorical_test(adults, col, "Adults")
        tables[col + "_observed_counts"] = obs.reset_index()
        tables[col + "_expected_counts"] = exp.reset_index()
    smoking_ages = adults.groupby("smoking_status")["age"].agg(["count", "mean", "median"])
    tables["smoking_age_composition_adults"] = smoking_ages.reset_index()
    tables["smoking_by_age_band_adults"] = proportion_table(add_age_bands(adults), ["age_band", "smoking_status"])
    return tables


def display_tests(results):
    print("\nAll p-values below are UNADJUSTED. Multiple tests increase false-positive risk.")
    for _, row in results.iterrows():
        print(f"\n{FACTOR_NAMES[row['factor']]} | {row['population']}")
        print("H0:", row["H0"])
        print("H1:", row["H1"])
        print("Test:", row["test"])
        print("Why:", row["why_selected"])
        print(f"Observed n: stroke = {row['n_stroke']}; no stroke = {row['n_no_stroke']}")
        print("Statistic: {:.6g}; p-value: {:.6g}; alpha: {:.2f}".format(
            row["statistic"], row["p_value_unadjusted"], row["alpha"]))
        if "min_expected_count" in row and pd.notna(row["min_expected_count"]):
            print("Minimum expected count: {:.2f}".format(row["min_expected_count"]))
        print(row["conclusion"])


def compare_report(tables, reference_path):
    """PDF values are labeled references; all current results come from the CSV."""
    references = pd.read_csv(reference_path)
    rows = []
    for _, ref in references.iterrows():
        src = tables[ref["table_name"]]
        sel = src.loc[src[ref["key_column"]].astype(str) == str(ref["key_value"])]
        if ref["population"] != "Any":
            sel = sel.loc[sel["population"] == ref["population"]]
        current, reported = float(sel.iloc[0][ref["result_column"]]), float(ref["report_value"])
        if ref["comparison_type"] == "Holm versus raw p":
            assessment = "Different scales: report Holm-adjusted p; project raw p."
        else:
            rounded = round(current, int(ref["decimal_places"]))
            assessment = ("Matches report to its displayed precision" if rounded == reported
                          else "Differs from report; inspect population and source")
        rows.append({"report_section": ref["report_section"], "quantity": ref["quantity"],
                     "population": ref["population"], "report_value": reported,
                     "calculated_value": current, "assessment": assessment})
    return pd.DataFrame(rows)


def export_tables(tables, folder):
    os.makedirs(folder, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(os.path.join(folder, name + ".csv"), index=False)


def write_findings(populations, tables, output_path):
    adults = populations["Adults"]
    lines = [
        "# Calculated findings", "",
        "Generated from stroke.csv; adult primary tests, alpha = 0.05.",
        "All p-values are unadjusted (no Holm correction).", "",
        "## Populations", ""
    ]
    for _, row in tables["population_summary"].iterrows():
        lines.append("- {}: {} records; {} strokes ({:.2f}%).".format(
            row["population"], row["total_records"], row["stroke_records"], row["stroke_percent"]))
    lines.extend(["", "## Five adult tests", ""])
    for _, row in tables["primary_tests_adults"].iterrows():
        text = "- {}: statistic {:.6g}; raw p = {:.6g}. {}".format(
            FACTOR_NAMES[row["factor"]], row["statistic"], row["p_value_unadjusted"], row["conclusion"])
        if row["factor"] in CONTINUOUS_COLUMNS:
            text += " Stroke mean {:.2f} vs no-stroke mean {:.2f}; diff {:.2f}.".format(
                row["mean_stroke"], row["mean_no_stroke"], row["mean_difference_stroke_minus_no_stroke"])
        lines.append(text)
    lines.extend(["", "## Sensitivity checks", "",
                  "| Factor | Adults | All ages | Complete adults |", "|---|---:|---:|---:|"])
    sens = tables["sensitivity_tests_unadjusted"]
    for factor in FACTOR_NAMES:
        vals = [float(sens.loc[(sens["factor"] == factor) & (sens["population"] == pop),
                               "p_value_unadjusted"].iloc[0])
                for pop in ["Adults", "All ages", "Complete adults"]]
        lines.append("| {} | {:.6g} | {:.6g} | {:.6g} |".format(FACTOR_NAMES[factor], *vals))
    lines.extend(["", "## Missingness and interpretation", ""])
    for stroke, label in [(1, "stroke"), (0, "no-stroke")]:
        g = adults.loc[adults["stroke"] == stroke]
        cnt = int(g["bmi"].isna().sum())
        lines.append(f"- Adult BMI missing in {cnt} / {len(g)} {label} records ({100*cnt/len(g):.2f}%).")
    lines.extend([
        "- BMI uses observed values only. Unequal missingness can bias comparisons.",
        "- Unknown smoking kept descriptively; excluded from smoking inference.",
        "- Rank tests check distributions, not necessarily medians.",
        "- Comparisons are unadjusted for age and other variables.",
        "- Record independence and random sampling cannot be verified.",
        "- See REPORT_COMPARISON.md and tables/report_comparison.csv for calculated vs reference values."
    ])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
