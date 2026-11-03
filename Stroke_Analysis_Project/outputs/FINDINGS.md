# Calculated findings

Generated from stroke.csv; adult primary tests, alpha = 0.05.
All p-values are unadjusted (no Holm correction).

## Populations

- All ages: 5110 records; 249 strokes (4.87%).
- Adults: 4254 records; 247 strokes (5.81%).
- Complete adults: 3256 records; 180 strokes (5.53%).

## Five adult tests

- Age: statistic 24.3453; raw p = 7.25356e-75. Reject H0 at alpha=0.05: evidence of a mean difference. Stroke mean 68.21 vs no-stroke mean 49.09; diff 19.12.
- Average glucose: statistic 6.519; raw p = 3.59113e-10. Reject H0 at alpha=0.05: evidence of a mean difference. Stroke mean 133.10 vs no-stroke mean 107.00; diff 26.10.
- BMI: statistic 0.0844419; raw p = 0.932776. Do not reject H0 at alpha=0.05: insufficient evidence of a mean difference. Stroke mean 30.47 vs no-stroke mean 30.43; diff 0.04.
- Hypertension: statistic 57.4679; raw p = 3.43542e-14. Reject H0 at alpha=0.05: evidence of an association.
- Smoking status: statistic 9.87438; raw p = 0.00717472. Reject H0 at alpha=0.05: evidence of an association.

## Sensitivity checks

| Factor | Adults | All ages | Complete adults |
|---|---:|---:|---:|
| Age | 7.25356e-75 | 2.11568e-95 | 1.90916e-51 |
| Average glucose | 3.59113e-10 | 2.40144e-11 | 3.79567e-09 |
| BMI | 0.932776 | 0.000333971 | 0.797692 |
| Hypertension | 3.43542e-14 | 6.06812e-20 | 1.48156e-15 |
| Smoking status | 0.00717472 | 0.00328546 | 0.0944168 |

## Missingness and interpretation

- Adult BMI missing in 39 / 247 stroke records (15.79%).
- Adult BMI missing in 142 / 4007 no-stroke records (3.54%).
- BMI uses observed values only. Unequal missingness can bias comparisons.
- Unknown smoking kept descriptively; excluded from smoking inference.
- Rank tests check distributions, not necessarily medians.
- Comparisons are unadjusted for age and other variables.
- Record independence and random sampling cannot be verified.
- See REPORT_COMPARISON.md and tables/report_comparison.csv for calculated vs reference values.
