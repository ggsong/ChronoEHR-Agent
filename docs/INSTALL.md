# Installation

ChronoEHR-Agent uses plain Python scripts and CSV/JSON/YAML configuration files.

```bash
git clone <repo-url>
cd ChronoEHR-Agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

For a no-data smoke test:

```bash
python3 src/chrono_ehr/run_study.py --list
python3 src/chrono_ehr/run_study.py --synthetic-demo
python3 src/chrono_ehr/run_study.py --validate-synthetic-demo
python3 scripts/release_audit.py --project-root .
```

For real-data workflows, set local dataset roots:

```bash
export MIMIC_IV_ROOT=/path/to/mimic-iv-3.1
export EICU_ROOT=/path/to/eicu-2.0
export CHARLS_ROOT=/path/to/CHARLS
```

The repository does not include controlled datasets or generated analysis outputs.
