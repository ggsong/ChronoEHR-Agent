# ChronoEHR-Agent

ChronoEHR-Agent is a research workflow agent for temporal EHR prediction studies. It organizes cohort definitions, prediction-time windows, leakage checks, model summaries, reporting assets, and handoff audits through a single command-line entry point.

The current private workspace contains completed MIMIC-IV chronic-disease readmission demos and external-benchmark preparation artifacts. This public repository is arranged so that code, configuration templates, documentation, and a synthetic demo can be shared without exposing controlled clinical datasets or local generated outputs.

## What It Does

- Registers multiple temporal clinical prediction studies in `configs/study_registry.json`.
- Runs reproducible study pipelines through `src/chrono_ehr/run_study.py`.
- Separates safe checks from expensive/model-running phases.
- Produces leakage audits, readiness checks, manuscript/report assets, and handoff manifests.
- Provides a synthetic demo path for CI and external reviewers who do not have clinical data access.

## Positioning

ChronoEHR-Agent is not a simulated clinical diagnosis agent or a patient-interaction benchmark. It is a research workflow agent for temporal structured-EHR prediction studies. Its design is positioned relative to recent medical-agent evaluation and audit work that emphasizes strong conventional baselines, workflow auditability, leakage control, reproducible task execution, and clear safety boundaries.

| Evaluation idea | Current support |
|---|---|
| Conventional baselines | Logistic, Random Forest, gradient-boosting, calibration, threshold, and decision-curve summaries |
| Structured EHR prediction | MIMIC-IV chronic-disease readmission workflows and planned eICU/CHARLS tasks |
| Leakage and prediction-time governance | Feature-window specs, leakage gates, and prediction-time audits |
| Agent workflow audit | Self-checks, doctor checks, status cards, runbooks, task queues, and handoff checklists |
| Reproducibility without controlled data | Config registry, synthetic demo, release audit, and GitHub Actions CI |

See [docs/RELATED_WORK.md](docs/RELATED_WORK.md) for how this project relates to MedAgentBoard, MedAgentAudit-style workflow auditing, MedAgentBench, AgentClinic, MedAgentGym, and related medical-agent frameworks.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 src/chrono_ehr/run_study.py --list
python3 src/chrono_ehr/run_study.py --synthetic-demo
python3 src/chrono_ehr/run_study.py --validate-synthetic-demo
```

The synthetic demo writes small generated artifacts under `outputs/demo/`. These files are intentionally ignored by git.

## Agent Demo Commands

These commands demonstrate the agent-control layer without requiring controlled clinical data:

```bash
python3 src/chrono_ehr/run_study.py --agent-demo-trace
python3 src/chrono_ehr/run_study.py --agent-capability-card
```

`--agent-demo-trace` records a safe planner/executor/auditor/release-guard trace over the synthetic EHR workflow. `--agent-capability-card` summarizes the public demo capabilities, including the unified entrypoint, action catalog, data contract audit, behavior trace, release guard, and CI smoke test.

## Real Data Setup

Controlled datasets are not included. To run real-data workflows, set local dataset roots in your shell or `.env`:

```bash
export MIMIC_IV_ROOT=/path/to/mimic-iv-3.1
export EICU_ROOT=/path/to/eicu-2.0
export CHARLS_ROOT=/path/to/CHARLS
```

Then run data-specific checks such as:

```bash
python3 src/chrono_ehr/run_study.py --agent-doctor
python3 src/chrono_ehr/run_study.py --study mimic_iv_3_1_diabetes_readmission --no-expensive
```

## Repository Layout

- `src/chrono_ehr/`: agent entrypoints, study runners, validators, report builders.
- `configs/`: registry and study configuration templates.
- `docs/`: public installation, release, and demo notes.
- `scripts/`: release-audit helpers.
- `.github/workflows/`: minimal CI that runs without controlled data.

Ignored local-only directories include `data/`, `outputs/`, `references/`, virtual environments, and caches.

## Publication Boundary

This project is a research workflow tool. It is not a clinical decision system and should not be used to guide patient care. Real MIMIC-IV, eICU, CHARLS, or other controlled datasets must be obtained from their official providers and kept out of this repository.

See [docs/INSTALL.md](docs/INSTALL.md), [docs/SYNTHETIC_DEMO.md](docs/SYNTHETIC_DEMO.md), [docs/RELATED_WORK.md](docs/RELATED_WORK.md), and [docs/GITHUB_RELEASE_AUDIT.md](docs/GITHUB_RELEASE_AUDIT.md).
