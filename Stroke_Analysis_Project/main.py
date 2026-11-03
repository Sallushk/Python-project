"""Run this file: python main.py. All menu choices are reusable."""

import os
import pandas as pd
from data_processing import (
    read_dataset, display_dataset_info, validate_and_clean, make_populations,
    audit_tables, FACTOR_NAMES
)
from statistical_analysis import (
    build_tables, descriptive_table, primary_tests, continuous_test,
    categorical_test, proportion_table, display_tests, export_tables,
    compare_report, write_findings
)
from visualization import generate_all_charts

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH   = os.path.join(PROJECT_FOLDER, "stroke.csv")
OUTPUT_FOLDER  = os.path.join(PROJECT_FOLDER, "outputs")
TABLE_FOLDER   = os.path.join(OUTPUT_FOLDER, "tables")
CHART_FOLDER   = os.path.join(OUTPUT_FOLDER, "charts")


def prepare_data(raw):
    cleaned, dupes = validate_and_clean(raw)
    pops = make_populations(cleaned)
    if pops["Adults"].empty or pops["Complete adults"].empty:
        raise ValueError("No adult or complete-adult records available for analysis.")
    return pops, dupes


def display_cleaning(raw, populations, duplicates_removed):
    tables = audit_tables(raw, populations, duplicates_removed)
    print("\nVALIDATION AND CLEANING AUDIT")
    print(tables["data_audit"].to_string(index=False))
    print("\nAdult unusual values (1.5 x IQR): flagged and RETAINED")
    print(tables["adult_outlier_audit"].to_string(index=False))
    print("\nNo BMI imputation. Unknown smoking kept. Adults (age >= 18) are primary.")
    export_tables(tables, TABLE_FOLDER)


def analyze_factor(populations, column):
    adults = populations["Adults"]
    if column in ["age", "avg_glucose_level", "bmi"]:
        summaries = descriptive_table({"Adults": adults})
        summaries = summaries.loc[summaries["factor"] == column].copy()
        print(f"\nAdult {FACTOR_NAMES[column]} summaries:")
        print(summaries.round(3).to_string(index=False))
        tests = pd.DataFrame([continuous_test(adults, column, "Adults"),
                              continuous_test(adults, column, "Adults", rank_test=True)])
        export_tables({column + "_adult_summary": summaries, column + "_adult_tests": tests}, TABLE_FOLDER)
    else:
        summaries = proportion_table(adults, column)
        print("\nAdult category totals and stroke / category totals:")
        print(summaries.round(3).to_string(index=False))
        row, observed, expected = categorical_test(adults, column, "Adults")
        tests = pd.DataFrame([row])
        print("\nObserved counts (cols: 0=no stroke, 1=stroke):")
        print(observed.to_string())
        print("\nExpected counts:")
        print(expected.round(2).to_string())
        export_tables({column + "_adult_summary": summaries, column + "_adult_tests": tests}, TABLE_FOLDER)
    display_tests(tests)
    if column == "bmi":
        print("BMI uses observed values only; missingness differs by outcome.")
    elif column == "smoking_status":
        print("Unknown shown descriptively; excluded from chi-square test.")


def run_complete_analysis(raw, populations, duplicates_removed, show_charts=True):
    """Reproducible non-interactive full run."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    display_dataset_info(raw)
    display_cleaning(raw, populations, duplicates_removed)
    tables = audit_tables(raw, populations, duplicates_removed)
    tables.update(build_tables(populations))
    reference_path = os.path.join(PROJECT_FOLDER, "report_reference_values.csv")
    if os.path.isfile(reference_path):
        tables["report_comparison"] = compare_report(tables, reference_path)
    else:
        print("Report reference file missing; report comparison unavailable.")
    export_tables(tables, TABLE_FOLDER)
    write_findings(populations, tables, os.path.join(OUTPUT_FOLDER, "FINDINGS.md"))
    display_tests(tables["primary_tests_adults"])
    print("\nSUPPLEMENTARY MANN-WHITNEY TESTS")
    display_tests(tables["mann_whitney_adults"])
    print("\nSensitivity raw p-values:")
    print(tables["sensitivity_tests_unadjusted"][
        ["population", "factor", "p_value_unadjusted", "conclusion"]].to_string(index=False))
    generate_all_charts(populations, CHART_FOLDER, show=show_charts)
    print(f"\nComplete analysis saved in {OUTPUT_FOLDER}")
    return tables


MENU_OPTIONS = {
    1: "Display Dataset Information",
    2: "Check and Clean Dataset",
    3: "Analyze Age",
    4: "Analyze Glucose Level",
    5: "Analyze BMI",
    6: "Analyze Hypertension",
    7: "Analyze Smoking Status",
    8: "Display Statistical Test Results",
    9: "Show All Visualizations",
    10: "Generate Complete Analysis",
    0: "Exit"
}
MENU_ACTIONS = {
    3: "age", 4: "avg_glucose_level", 5: "bmi", 6: "hypertension", 7: "smoking_status"
}


def display_menu():
    print("\n===== STROKE RISK FACTOR ANALYSIS =====")
    for k, v in MENU_OPTIONS.items():
        print(f"{k}. {v}")


def main():
    print("Stroke Risk Factor Analysis Using Patient Health Records")
    print("Salahuddin Shaikh | Roll 16014025056 | Python Programming Laboratory")
    try:
        raw = read_dataset(DATASET_PATH)
        populations, dupes = prepare_data(raw)
    except (FileNotFoundError, ValueError, OSError, pd.errors.ParserError) as e:
        print("Unable to prepare dataset:", e)
        return

    while True:
        display_menu()
        try:
            choice = int(input("Enter your choice: "))
        except ValueError:
            print("Please enter a whole number from 0 to 10.")
            continue
        except (EOFError, KeyboardInterrupt):
            print("\nInput ended. Exiting.")
            break

        try:
            if choice == 0:
                print("Project closed.")
                break
            elif choice == 1:
                display_dataset_info(raw)
            elif choice == 2:
                display_cleaning(raw, populations, dupes)
            elif choice in MENU_ACTIONS:
                analyze_factor(populations, MENU_ACTIONS[choice])
            elif choice == 8:
                results = primary_tests(populations["Adults"], "Adults")
                display_tests(results)
                export_tables({"primary_tests_adults": results}, TABLE_FOLDER)
            elif choice == 9:
                print("Close each plot window to display the next.")
                generate_all_charts(populations, CHART_FOLDER)
            elif choice == 10:
                print("Close each plot window to continue.")
                run_complete_analysis(raw, populations, dupes)
            else:
                print("Invalid choice. Select 0 to 10.")
        except (ValueError, OSError) as e:
            print("This operation could not finish:", e)
        except KeyboardInterrupt:
            print("\nOperation interrupted; returning to menu.")


if __name__ == "__main__":
    main()
