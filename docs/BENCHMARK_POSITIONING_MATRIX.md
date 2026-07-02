# Benchmark Positioning Matrix

This matrix clarifies how ChronoEHR-Agent relates to common medical-agent benchmarks and frameworks. It helps keep the project positioned as a research workflow agent rather than a clinical chatbot or benchmark replacement.

## Matrix

| Reference area | Typical emphasis | ChronoEHR-Agent relationship | Current status |
|---|---|---|---|
| MedAgentBoard | Agent and baseline comparison across medical tasks, including structured data workflows | Aligns with the need for strong baselines, reproducible execution, and explicit task framing | Implemented as local study registry, baseline organization, and validation gates |
| MedAgentAudit | Safety, reproducibility, task boundaries, and operational auditability | Directly aligned with the project's audit-first design | Implemented through release audit, command lint, boundary audit, status checks, and handoff checks |
| AgentClinic | Simulated clinical interactions, information gathering, tool use, diagnosis | Background benchmark reference only | Not implemented; project avoids diagnosis and patient simulation |
| MedAgentBench-style EHR environments | Interactive medical-record tasks and EHR tool use | Useful future evaluation target | Not implemented; current project provides non-interactive command workflows |
| PhysicianBench | Physician-task benchmark framing | Reference for task design and evaluation boundaries | Not implemented as a benchmark; informs positioning |
| MedAgentGym | Training/evaluation environments for biomedical coding or agent trajectories | Possible future task-format reference | Not implemented; current work focuses on reproducible local workflow execution |
| ColaCare | Medical agent workflow framework with coordinated roles | Design inspiration for multi-step medical workflows | Partially reflected through planner/executor/auditor/release-guard trace |
| MDAgents | Multi-agent collaboration for medical reasoning | Design inspiration for role separation and consensus/checking | Reflected as local role-like stages, not as independent LLM agents |
| MedAgent-Pro | Medical tool-use and agentic execution patterns | Design reference for tool routing and task decomposition | Reflected through command routing and action catalog |

## Why ChronoEHR-Agent Is Narrower

ChronoEHR-Agent intentionally focuses on a narrower but concrete slice:

- temporal cohort definition;
- prediction-time governance;
- leakage control;
- conventional baseline organization;
- public no-data demo;
- controlled-data-safe packaging;
- SHARE/CHARLS-style longitudinal survey extension.

This narrower scope makes the project easier to audit and safer to share. It also avoids overclaiming clinical interaction abilities that the project does not implement.

## Recommended Evaluation Lens

Review ChronoEHR-Agent as a demonstration of agent development capability:

1. Can it expose a stable agent command surface?
2. Can it route tasks into safe, model, expensive, and report phases?
3. Can it validate its own outputs?
4. Can it handle controlled-data boundaries?
5. Can it extend to a new database without changing the public-data policy?
6. Can it explain its relationship to medical-agent benchmarks without claiming to replace them?

Under this lens, the current deliverables are intentionally aligned with workflow-agent engineering rather than large-scale model training.
