# Release Structure

Recommended public repository structure:

```text
ChronoEHR-Agent/
  README.md
  .env.example
  .gitignore
  requirements.txt
  configs/
  src/chrono_ehr/
  scripts/
  docs/
    INSTALL.md
    SYNTHETIC_DEMO.md
    GITHUB_RELEASE_AUDIT.md
    RELEASE_STRUCTURE.md
  .github/workflows/ci.yml
```

Private-only workspace content:

```text
data/
outputs/
references/
docs/resume_state.md
docs/data_inventory.md
docs/low_quota_handoff.md
```

If a mentor needs the completed formal package, send the curated handoff zip from the private workspace rather than committing generated artifacts to GitHub.
