# Synthetic Demo

The synthetic demo is a small generated temporal-prediction example. It is designed for GitHub CI and reviewer smoke tests, not for scientific inference.

Run:

```bash
python3 src/chrono_ehr/run_study.py --synthetic-demo
python3 src/chrono_ehr/run_study.py --validate-synthetic-demo
```

Outputs:

- `outputs/demo/synthetic_cohort.csv`
- `outputs/demo/synthetic_demo_metrics.csv`
- `outputs/demo/synthetic_demo_report.md`

The generated cohort includes artificial demographics, utilization variables, a prediction-time label, and deterministic train/test splits. It contains no real patients and no protected health information.
