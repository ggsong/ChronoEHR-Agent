# SHARE Longitudinal Survey Extension

Version: v0.1
Status: data-connection proof; no model training required
Recommended data path: `${SHARE_ROOT}`

## Positioning

SHARE is a longitudinal ageing survey, not an EHR database. In ChronoEHR-Agent it is useful as a cross-database extension because it tests whether the agent can handle wave-based prediction times, follow-up outcomes, survey identifiers, and future-wave leakage controls.

Recommended first task:

> Use SHARE wave 1 baseline variables to define a baseline cohort without diabetes, then identify incident diabetes in wave 2 or wave 4.

This task demonstrates:

- database readiness detection;
- harmonized variable mapping;
- baseline-vs-follow-up role separation;
- cohort skeleton construction without expensive model training;
- explicit leakage status for future-wave outcome variables.

## Cohort Definition

Baseline:

- SHARE wave 1.

Include:

- participated in wave 1;
- valid `mergeid`;
- baseline age available;
- baseline diabetes status can be determined;
- at least one follow-up diabetes status in wave 2 or wave 4.

Exclude:

- diabetes at baseline;
- missing person identifier;
- no usable follow-up diabetes information.

## Outcome

Primary outcome:

`incident_diabetes_followup`: baseline no diabetes, then diabetes diagnosis or diabetes medication in wave 2 or wave 4.

The first implementation is a skeleton suitable for data auditing. It does not train a model and does not produce clinical recommendations.

## Baseline Feature Groups

Allowed wave 1 feature groups:

- demographics: age, sex, education, country;
- lifestyle: smoking, alcohol;
- health status: BMI, hypertension, heart disease, stroke, lung disease;
- function: ADL, IADL, mobility;
- mental health: depression;
- optional survey design: person-level analysis weight.

## Leakage Audit Rules

Future-wave variables must not be used as baseline features:

- wave 2 or wave 4 diabetes diagnosis;
- wave 2 or wave 4 diabetes medication;
- future interview status, attrition, death, or follow-up participation as ordinary predictors;
- any post-baseline health status that encodes or proxies the outcome.

## Commands

```bash
export SHARE_ROOT=/path/to/SHARE
python3 src/chrono_ehr/run_study.py --share-readiness
python3 src/chrono_ehr/run_study.py --share-wave-map
python3 src/chrono_ehr/run_study.py --validate-share-wave-map
python3 src/chrono_ehr/run_study.py --share-incident-diabetes-cohort
python3 src/chrono_ehr/run_study.py --validate-share-incident-diabetes-cohort
```

## Expected Outputs

- `outputs/reports/share_data_readiness_report.md`
- `outputs/tables/share_wave_variable_map.csv`
- `outputs/reports/share_wave_variable_map.md`
- `data/processed/share_incident_diabetes_cohort_skeleton.csv`
- `outputs/reports/share_incident_diabetes_cohort_skeleton.md`
