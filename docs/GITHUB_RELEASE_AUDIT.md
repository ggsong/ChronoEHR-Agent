# GitHub Release Audit

Before publishing, run:

```bash
python3 scripts/release_audit.py --project-root .
```

The audit checks the public-facing files for local absolute paths and confirms that controlled data/output directories are ignored.

Publishable by default:

- `README.md`
- `.env.example`
- `.gitignore`
- `requirements.txt`
- `configs/`
- `src/`
- `scripts/`
- `.github/workflows/`
- public docs listed in `.gitignore`

Do not publish:

- `data/`
- `outputs/`
- `references/`
- `.env`
- generated Word, PowerPoint, PDF, zip, or rendered manuscript files
- any controlled clinical dataset tables

The private workspace can keep local handoff packages and generated reports. The public repository should point reviewers to the synthetic demo and describe how authorized users can rerun real-data analyses locally.
