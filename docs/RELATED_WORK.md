# Related Work and Positioning

ChronoEHR-Agent is positioned as a research workflow agent for temporal structured-EHR prediction studies. It is not a clinical diagnosis chatbot, a simulated patient-interaction benchmark, or a FHIR sandbox benchmark. The closest alignment is with medical-agent evaluation and audit work that asks whether agentic systems are reliable, auditable, reproducible, and competitive with strong conventional baselines.

## Alignment Summary

| Project or area | Primary focus | Relationship to ChronoEHR-Agent |
|---|---|---|
| MedAgentBoard | Benchmarking multi-agent, single-LLM, and conventional methods across medical tasks, including structured EHR prediction and workflow automation | ChronoEHR-Agent aligns with the structured EHR prediction and workflow-audit parts: it keeps conventional baselines, prediction-time definitions, leakage checks, and delivery artifacts in one reproducible workflow |
| MedAgentAudit-style workflow auditing | Auditing agent safety, task boundaries, reproducibility, and operational readiness | ChronoEHR-Agent implements local self-checks, status cards, release audits, handoff checklists, safe/expensive phase separation, and artifact freshness checks |
| MedAgentBench | Interactive medical-record agent benchmark in a FHIR-like EHR environment | Future evaluation reference. ChronoEHR-Agent currently does not implement a FHIR environment, but its task registry and runbook structure could later wrap interactive EHR tasks |
| AgentClinic | Simulated clinical agent benchmark for patient interaction, multimodal information gathering, tool use, and diagnosis | Background motivation only. ChronoEHR-Agent does not simulate doctor-patient dialogue or make diagnoses |
| MedAgentGym | Training/evaluation environment for coding-based biomedical reasoning agents | Possible future direction for task-instance formatting and coding-agent trajectories |
| ColaCare, MDAgents, MedAgent-Pro, and related frameworks | Medical agent workflows, multi-agent collaboration, and tool-use patterns | Design references for planner/executor/evaluator roles, tool routing, and audit loops, not current dependencies |

## What This Project Claims

ChronoEHR-Agent claims to provide a reproducible workflow layer for temporal EHR prediction research:

- study registration through config files;
- prediction-time and feature-window governance;
- leakage and readiness audits;
- conventional baseline organization;
- result, report, and handoff artifact tracking;
- a no-data synthetic demo for public CI.

## What This Project Does Not Claim

ChronoEHR-Agent does not claim to be:

- a clinical decision support system;
- a diagnostic medical agent;
- a replacement for MedAgentBench, AgentClinic, or MedAgentBoard;
- a validated agent benchmark;
- a system for direct patient-care recommendations.

## Why This Position Matters

Medical-agent projects can look impressive while hiding weak baselines, unclear prediction times, data leakage, or irreproducible execution. ChronoEHR-Agent focuses on the less flashy but important layer around EHR prediction experiments: keeping the research workflow explicit, auditable, and safe to hand off.

This makes the project complementary to medical-agent benchmarks. Benchmarks evaluate what agents can do; ChronoEHR-Agent organizes how one specific class of medical AI research workflow is specified, checked, run, and delivered.
