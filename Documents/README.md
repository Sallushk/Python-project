# Stroke Risk Factor Analysis Using Patient Health Records

**Student:** Salahuddin Shaikh  
**Roll number:** 16014025056  
**College:** K. J. Somaiya School of Engineering  
**Branch:** Second-year Electronics and Computer Engineering (EXCP)  
**Subject:** Python Programming Laboratory

This beginner-friendly capstone investigates associations between recorded stroke status and **age, average glucose, BMI, hypertension and smoking status**. It uses the supplied CSV, the Modules 1-5 syllabus image and the two supplied PDFs. It does not fit a predictive model or establish causes. All statistical results are recomputed from `stroke.csv`.

## Run in VS Code on Windows

1. Extract the ZIP to a normal folder. Open **Stroke_Analysis_Project** using **File > Open Folder** in VS Code.
2. Install Python **3.11 or newer** and the VS Code Python extension if needed. Select the Python interpreter using **Ctrl+Shift+P > Python: Select Interpreter**.
3. Open **Terminal > New Terminal**. Make sure the terminal is inside `Stroke_Analysis_Project`, where `main.py`, `stroke.csv` and `requirements.txt` are visible.
4. Install the five necessary packages, then run the program:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Choose **10** to generate the complete analysis. Close each displayed chart window to proceed to the next chart. After all 11 windows close, the menu returns. Choose **0** to exit. Files are saved automatically in `outputs`; they are also included in this package from the successful verification run.

If `python` is not recognized, use the Windows Python launcher `py` in both commands. If a virtual environment is wanted, create it with `python -m venv .venv`, then select `.venv\Scripts\python.exe` in VS Code. Using its full path in the terminal also avoids PowerShell activation-policy problems:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

The program uses paths relative to its own file, so moving the extracted project or starting it from another folder works. No Kaggle account, notebook, network access or dataset download is needed after installation.

## Files and module responsibilities

```text
Stroke_Analysis_Project/
  main.py
  data_processing.py
  statistical_analysis.py
  visualization.py
  stroke.csv
  requirements.txt
  README.md
  REPORT_COMPARISON.md
  SYLLABUS_COVERAGE.md
  VIVA_QUESTIONS.md
  TESTING.md
  report_reference_values.csv
  outputs/
    FINDINGS.md
    charts/                 10 individual PNGs and a 2 x 2 dashboard
    tables/                 audit, summaries, tests and comparison CSVs
```

| Module | How it works |
|---|---|
| `main.py` | Loads and validates data once; controls a `while` menu using `if/elif`; calls the other modules; exports complete results. |
| `data_processing.py` | Reads the CSV, checks columns/types/categories/duplicates, preserves missing BMI and Unknown smoking, creates all-age/adult/complete-adult copies, and audits unusual values. |
| `statistical_analysis.py` | Calculates observed-value mean/median/sample SD; performs Welch, chi-square and supplementary Mann-Whitney tests; computes proportions, correlations and sensitivity results; writes CSVs and findings. |
| `visualization.py` | Uses Matplotlib and Seaborn to create, label, save and display 11 charts. Reuses plotting functions for the dashboard. |

## Menu

| Choice | Operation |
|---|---|
| 1 | Raw shape, names, first five records, data types, missing values, duplicates and outcome counts. |
| 2 | Validation/cleaning audit, analysis populations and retained IQR outliers; saves audit tables. |
| 3 | Observed adult age summaries, Welch t-test and supplementary rank test. |
| 4 | Observed adult glucose summaries and tests. |
| 5 | Observed adult BMI summaries and tests; notes missingness. |
| 6 | Hypertension totals, stroke proportions and chi-square test with contingency tables. |
| 7 | All smoking-category totals/proportions; known-smoking chi-square test. |
| 8 | The five adult primary tests with hypotheses, statistics, raw p-values and conclusions. |
| 9 | Displays and saves all 11 charts. |
| 10 | Runs the complete audit, summaries, primary/supplementary/sensitivity tests, correlations, report comparisons and charts. |
| 0 | Exits the program. |

Invalid numbers and text return to the menu. Missing or invalid input data produce a readable error. CSV output is overwritten on rerun; the source CSV is never written to. Close any output CSV open in Excel if Windows reports a file-permission error.

## Preparation and populations

The input has 5,110 rows, 12 columns, 249 recorded strokes and 4,861 no-stroke records. These facts were verified on the supplied file; the program calculates them each time. The raw CSV is copied byte-for-byte into this package.

- Validate all 12 expected columns, numeric values, binary outcome/conditions, positive observed BMI/glucose, nonnegative age, positive integer IDs and listed categories. Missing values outside BMI or invalid values stop analysis for inspection.
- Remove exact duplicate records from an analysis copy, if present. The supplied file has none. Conflicting repeated IDs stop the analysis. No valid observations are removed as outliers: the adult 1.5 x IQR audit flags them and retains them.
- **All ages:** all valid records after duplicate checks; used for descriptions and sensitivity comparisons.
- **Adults:** age >= 18; the primary population, matching the report. Age/glucose/hypertension comparisons retain adults with missing BMI or Unknown smoking.
- **BMI analysis:** use observed BMI within the selected population. There are 201 all-age missing BMI values and 181 in adults. No imputation is performed anywhere.
- **Smoking inference:** exclude `Unknown` from that test; do not recode it as `never smoked`. All descriptive smoking charts include `Unknown`.
- **Complete adults:** adult records with observed BMI AND known smoking. This is a separate sensitivity population (3,256 records), never a replacement for the primary adult population.

## Statistical methods and interpretation

| Question | Method | Why |
|---|---|---|
| Do adult mean age, glucose and BMI differ by outcome? | Two-sided Welch independent-sample t-tests, `equal_var=False` | Continuous measurements with unequal sample sizes and potentially unequal variances. Only observed values enter each test. |
| Are hypertension and stroke labels associated? | Pearson chi-square, `correction=False` | Both variables are categorical. Matches the report's no-continuity-correction choice. |
| Does outcome vary across known smoking categories? | Pearson chi-square on former/never/current categories | Three recorded categories versus a binary outcome; Unknown excluded only here. |
| Are continuous distributions different by outcome? | Two-sided asymptotic Mann-Whitney U, adult supplementary tests | Rank-based check useful for glucose skewness/outliers; does not automatically test medians or replace Welch. |
| How do numerical variables vary together? | Pearson and Spearman correlation matrices | Pearson describes linear association; Spearman correlates ranks and matches the report. Available observed pairs are used, with pair counts exported. |
| Do findings depend on inclusion rules? | Repeat the five tests on all ages and complete adults | Shows sensitivity to age composition and incomplete records. These are exploratory checks. |

Every inferential test has H0, H1, test name, reason, group sample sizes, statistic, **unadjusted** p-value, alpha=0.05 and a plain-language conclusion in the output. Chi-square degrees of freedom, expected cell counts and their minimum are exported; a minimum below 5 triggers a caution instead of a firm decision. Hypotheses for rank tests concern distributions; tests of means concern population means.

Records are assumed independent, but the source cannot verify independence or random sampling. Welch allows unequal variances but still relies on valid independent-group mean comparisons; rank checks supplement the skewed glucose comparison. There are multiple tests, so unadjusted 0.05 decisions do not control the family's false-positive rate. Failing to reject H0 does not establish equality or prove no association.

IDs and arbitrary numeric encodings of smoking/gender/work type are excluded from correlations. Binary conditions and the stroke label are explicitly 0/1. Heart disease appears only as a supplementary correlation variable; it does not add a sixth primary factor. No correlation hypothesis p-values are generated.

ANOVA is unnecessary for the specified two-outcome mean comparisons. A one-sample KS normality test against a fitted normal distribution would need care about estimated parameters; it is not needed for this plan. Time-series forecasting is unsuitable because this dataset has no dates. These methods and unused standard-library modules are not added solely to demonstrate syllabus coverage.

## Charts and tables

| PNG | Population and meaning |
|---|---|
| `01_outcome_counts_all_ages.png` | All-age outcome counts; percentage denominator is all records. |
| `02_age_distribution_adults.png` | Adult age histogram by outcome; each group's area is one to show shapes despite imbalance. |
| `03_glucose_boxplot_adults.png` | Adult observed glucose with unverified recorded units; outliers visible. |
| `04_bmi_boxplot_observed_adults.png` | Adult observed BMI; missing measurements omitted for this graph only. |
| `05_hypertension_proportions_adults.png` | Stroke / total within each adult hypertension group. |
| `06_smoking_proportions_adults.png` | Stroke / total within each adult smoking category, including Unknown. |
| `07_spearman_heatmap_adults.png` | Spearman correlations of numerical variables, available pairs, scale -1 to +1. |
| `08_age_band_proportions_adults.png` | Stroke / total within each adult age band; bands are left-inclusive, right-exclusive. |
| `09_smoking_counts_adults.png` | All adult smoking categories and their record counts. |
| `10_bmi_missingness_adults.png` | Missing BMI / total within each adult outcome group. |
| `11_adult_dashboard.png` | Related adult age/glucose/BMI/hypertension plots in a 2 x 2 layout. |

Proportion bars start at zero and label the numerator/denominator alongside percentages. They describe dataset proportions, not population incidence or future risk. Unlike the original report, bars have no Wilson confidence intervals; counts show group size but do not quantify uncertainty.

Key CSVs include `data_audit`, `raw_missing_values`, `population_summary`, `descriptive_statistics`, `missingness_by_outcome`, `primary_tests_adults`, `mann_whitney_adults`, `sensitivity_tests_unadjusted`, observed/expected contingency counts, category/age-band proportions, smoking age composition, Pearson/Spearman correlations and pair counts, and `report_comparison`.

Read **outputs/FINDINGS.md** for calculated results and **REPORT_COMPARISON.md** for exactly which PDF sections need changing. The reference values file contains transcribed, explicitly labeled report values only; it never supplies inputs to statistical tests or graphs. Removing that file disables report comparison but does not change analysis results.

## Syllabus and practical preparation

Read **SYLLABUS_COVERAGE.md** for the mapping to Modules 1-5 and **VIVA_QUESTIONS.md** for 22 questions and answers about the actual code. No classes, decorators, lambdas, generators, asynchronous code, machine learning, multivariable models or unfamiliar packages are used. `os` is the only additional standard-library module in project code and supports file paths/folder creation.

## Provenance and references

Input: user-supplied `stroke.csv`, `Problem_Definition_and_Dataset.pdf`, `Stroke_Capstone_Report.pdf` and syllabus image. The CSV name in the pasted request (`stroke(1).csv`) differs from the actual supplied file name; this project includes and loads **stroke.csv**. The PDF names likewise use the actual supplied names without `(2)`.

The supplied report identifies the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) and a public GitHub acquisition mirror. This project analyzes the user's exact local file; it does not substitute a downloaded dataset. Hospital, population, measurement timing, sampling and glucose units are not independently verified. Former smokers' age composition and BMI missingness constrain interpretation; all comparisons are unadjusted for other factors.

API details checked against official documentation:

- [SciPy independent-sample t-test and Welch option](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
- [SciPy contingency-table chi-square test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)
- [SciPy Mann-Whitney U test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)

Pinned package versions in `requirements.txt` match the environment actually used for execution; see **TESTING.md** for what was verified. No original PDF was edited by this project.
