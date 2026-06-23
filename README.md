# ChronoEHR-Agent

ChronoEHR-Agent is a research workflow agent for temporal EHR prediction studies. It organizes cohort definitions, prediction-time windows, leakage checks, model summaries, reporting assets, and handoff audits through a single command-line entry point.

The current private workspace contains completed MIMIC-IV chronic-disease readmission demos and external-benchmark preparation artifacts. This public repository is arranged so that code, configuration templates, documentation, and a synthetic demo can be shared without exposing controlled clinical datasets or local generated outputs.

## What It Does

- Registers multiple temporal clinical prediction studies in `configs/study_registry.json`.
- Runs reproducible study pipelines through `src/chrono_ehr/run_study.py`.
- Separates safe checks from expensive/model-running phases.
- Produces leakage audits, readiness checks, manuscript/report assets, and handoff manifests.
- Provides a synthetic demo path for CI and external reviewers who do not have clinical data access.

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

See [docs/INSTALL.md](docs/INSTALL.md), [docs/SYNTHETIC_DEMO.md](docs/SYNTHETIC_DEMO.md), and [docs/GITHUB_RELEASE_AUDIT.md](docs/GITHUB_RELEASE_AUDIT.md).
