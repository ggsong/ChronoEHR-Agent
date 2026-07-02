# Agent Development Alignment

ChronoEHR-Agent is a compact implementation exercise for medical-agent research workflow development. It is designed to show that the project can translate medical-agent concepts into a runnable, auditable, and data-governed workflow rather than only describing those concepts.

## Learning Plan Mapping

| Learning-plan area | How ChronoEHR-Agent connects it to implementation |
|---|---|
| MedAgentBoard | Uses the same broad concern with comparing agentic workflows, single-run baselines, and conventional methods. ChronoEHR-Agent focuses on the structured EHR / longitudinal prediction side: cohorts, prediction times, leakage controls, and reproducible outputs. |
| MedAgentAudit-style auditing | Implements local audit gates: release audit, boundary audit, command linting, artifact freshness checks, handoff checks, safe/expensive phase separation, and validation reports. |
| Conventional baselines | Organizes logistic regression, Random Forest, gradient boosting, calibration, threshold, decision-curve, and subgroup summaries where real-data outputs exist, while keeping the public demo no-data safe. |
| AgentClinic | Used as a reference for simulated clinical-agent evaluation. ChronoEHR-Agent deliberately does not simulate patient interaction or diagnosis. |
| MedAgentBench / MedAgentBench-style EHR benchmarks | Used as a reference for EHR task framing and interactive medical-record environments. ChronoEHR-Agent does not implement a FHIR sandbox, but it provides a task registry and command layer that could wrap such tasks later. |
| PhysicianBench / MedAgentGym | Used as reference points for benchmark-style task instances, tool-use trajectories, and coding-agent evaluation. ChronoEHR-Agent currently focuses on reproducible workflow execution rather than benchmark training. |
| ColaCare, MDAgents, MedAgent-Pro | Used as design references for planner/executor/auditor roles, multi-step task decomposition, and tool-routed medical workflows. ChronoEHR-Agent implements these ideas as local command routes and validation stages, not as a clinical multi-agent product. |
| MCP-style tool access | Mapped to explicit local tool boundaries: `run_study.py` exposes stable commands, `configs/agent_action_catalog.json` describes safe/model/report actions, and validators check command consistency. |
| Skills | Mapped to reusable workflow modules: readiness checks, cohort builders, leakage gates, report generators, release audits, and validation scripts. |
| Progressive disclosure | Reflected in the documentation and command structure: README quick start, dedicated docs, registry entries, detailed reports, then ignored local outputs for deeper inspection. |
| Tool calling | Implemented through a single agent command surface that routes to focused scripts and records validation outputs. |
| Codex-assisted development | The project demonstrates a Codex-assisted agent development workflow: planning, implementation, validation, GitHub release hygiene, package preparation, and iterative data-source extension. |

## Implemented Agent Concepts

| Concept | Concrete project artifact |
|---|---|
| Unified entrypoint | `src/chrono_ehr/run_study.py` |
| Action catalog | `configs/agent_action_catalog.json` |
| Stable entrypoints | `configs/agent_entrypoints.json` |
| Task registry | `configs/study_registry.json` |
| Safe vs expensive phases | runbook and task-queue logic in `src/chrono_ehr/agent_*` modules |
| No-data reproducibility | `--synthetic-demo`, `--validate-synthetic-demo`, GitHub Actions CI |
| Behavior trace | `--agent-demo-trace` |
| Capability summary | `--agent-capability-card` |
| Release hygiene | `scripts/release_audit.py` and `docs/GITHUB_RELEASE_AUDIT.md` |
| Longitudinal database extension | SHARE readiness, wave map, cohort skeleton, and validation commands |

## What This Shows

The project is meant to demonstrate the ability to:

- turn medical-agent reading into an executable architecture;
- define boundaries for what the agent should and should not do;
- build safe command surfaces around data workflows;
- separate public reproducibility from controlled data access;
- extend the workflow to a new database without committing raw data;
- validate outputs before packaging or publishing.

## Boundary

This alignment document is not claiming that ChronoEHR-Agent replaces MedAgentBoard, MedAgentAudit, MedAgentBench, AgentClinic, PhysicianBench, MedAgentGym, ColaCare, MDAgents, or MedAgent-Pro. It documents how those ideas informed a narrower research workflow agent for temporal EHR and longitudinal survey prediction studies.
