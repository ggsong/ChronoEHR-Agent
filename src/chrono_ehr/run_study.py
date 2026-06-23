#!/usr/bin/env python3
"""Run a registered ChronoEHR-Agent study pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
REGISTRY = PROJECT / "configs" / "study_registry.json"


def load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def studies_by_id(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {study["id"]: study for study in registry.get("studies", [])}


def list_studies(registry: dict[str, Any]) -> None:
    active = registry.get("active_study")
    print("Registered studies:")
    for study in registry.get("studies", []):
        marker = "*" if study["id"] == active else " "
        print(
            f"{marker} {study['id']} | status={study.get('status', 'unknown')} | "
            f"cohort={study.get('cohort', 'NA')} | outcome={study.get('outcome', 'NA')}"
        )
    planned = registry.get("planned_studies", [])
    if planned:
        print("\nPlanned studies:")
        for study in planned:
            print(f"  {study['id']} | status={study.get('status', 'planned')} | priority={study.get('priority', 'NA')}")


def build_pipeline_command(study: dict[str, Any], args: argparse.Namespace, project_root: Path) -> list[str]:
    pipeline = project_root / study["pipeline"]
    cmd = [sys.executable, str(pipeline), "--project-root", str(project_root)]
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.no_expensive:
        cmd.append("--no-expensive")
    if args.only:
        cmd.append("--only")
        cmd.extend(args.only)
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--study", help="Study id from configs/study_registry.json. Defaults to active_study.")
    parser.add_argument("--list", action="store_true", help="List registered and planned studies without running.")
    parser.add_argument(
        "--benchmark-summary",
        action="store_true",
        help="Generate the registry-level chronic disease benchmark summary without running a cohort pipeline.",
    )
    parser.add_argument(
        "--progress-report",
        action="store_true",
        help="Generate a module-level ChronoEHR-Agent progress report without running a cohort pipeline.",
    )
    parser.add_argument(
        "--validate-prediction-specs",
        action="store_true",
        help="Validate registry-level prediction-time model specs without running models.",
    )
    parser.add_argument(
        "--leakage-gate",
        action="store_true",
        help="Run the prediction-time leakage gate before modeling.",
    )
    parser.add_argument(
        "--validate-feature-windows",
        action="store_true",
        help="Validate feature extraction time-window specs against generated files.",
    )
    parser.add_argument(
        "--audit-extractor-windows",
        action="store_true",
        help="Audit which extractors consume shared feature-window specs.",
    )
    parser.add_argument(
        "--check-model-deps",
        action="store_true",
        help="Check optional model dependencies such as scikit-learn, XGBoost, and LightGBM.",
    )
    parser.add_argument(
        "--cdsl-readiness",
        action="store_true",
        help="Audit whether local CDSL data is ready for external benchmark or validation work.",
    )
    parser.add_argument(
        "--cdsl-temporal-benchmark",
        action="store_true",
        help="Run a lightweight CDSL mortality prediction-time benchmark.",
    )
    parser.add_argument(
        "--cdsl-leakage-audit",
        action="store_true",
        help="Audit CDSL temporal benchmark leakage risks.",
    )
    parser.add_argument(
        "--cdsl-traditional-baselines",
        action="store_true",
        help="Run CDSL logistic, Random Forest, and HistGradientBoosting baselines.",
    )
    parser.add_argument(
        "--cdsl-summary-figures",
        action="store_true",
        help="Plot CDSL temporal benchmark summary figures.",
    )
    parser.add_argument(
        "--cdsl-calibration-decision",
        action="store_true",
        help="Summarize CDSL calibration and decision-curve outputs from traditional baseline predictions.",
    )
    parser.add_argument(
        "--validate-cdsl-calibration-decision",
        action="store_true",
        help="Validate CDSL calibration and decision-curve outputs.",
    )
    parser.add_argument(
        "--eicu-readiness",
        action="store_true",
        help="Audit local eICU raw-data readiness for an external ICU benchmark.",
    )
    parser.add_argument(
        "--eicu-cohort",
        action="store_true",
        help="Build the eICU first-24h hospital mortality cohort skeleton.",
    )
    parser.add_argument(
        "--validate-eicu-cohort",
        action="store_true",
        help="Validate the eICU first-24h hospital mortality cohort skeleton.",
    )
    parser.add_argument(
        "--eicu-temporal-features",
        action="store_true",
        help="Extract eICU first-24h lab/vital temporal feature skeletons.",
    )
    parser.add_argument(
        "--validate-eicu-temporal-features",
        action="store_true",
        help="Validate eICU first-24h temporal feature skeletons.",
    )
    parser.add_argument(
        "--eicu-leakage-gate",
        action="store_true",
        help="Run the eICU first-24h leakage gate.",
    )
    parser.add_argument(
        "--eicu-logistic-baseline",
        action="store_true",
        help="Run the eICU first-24h lightweight logistic regression baseline.",
    )
    parser.add_argument(
        "--validate-eicu-logistic-baseline",
        action="store_true",
        help="Validate eICU first-24h logistic baseline outputs.",
    )
    parser.add_argument(
        "--eicu-model-comparison",
        action="store_true",
        help="Run eICU first-24h traditional model comparisons.",
    )
    parser.add_argument(
        "--validate-eicu-model-comparison",
        action="store_true",
        help="Validate eICU first-24h model-comparison outputs.",
    )
    parser.add_argument(
        "--eicu-baseline-figures",
        action="store_true",
        help="Generate eICU first-24h baseline ROC/PR/calibration figures.",
    )
    parser.add_argument(
        "--validate-eicu-baseline-figures",
        action="store_true",
        help="Validate eICU first-24h baseline figure and calibration outputs.",
    )
    parser.add_argument(
        "--eicu-probability-recalibration",
        action="store_true",
        help="Recalibrate eICU first-24h baseline probabilities on the validation split.",
    )
    parser.add_argument(
        "--validate-eicu-probability-recalibration",
        action="store_true",
        help="Validate eICU first-24h probability recalibration outputs.",
    )
    parser.add_argument(
        "--charls-readiness",
        action="store_true",
        help="Audit local CHARLS wave-data readiness for longitudinal chronic-disease tasks.",
    )
    parser.add_argument(
        "--charls-wave-map",
        action="store_true",
        help="Build the concrete CHARLS harmonized wave-variable map for incident diabetes.",
    )
    parser.add_argument(
        "--validate-charls-wave-map",
        action="store_true",
        help="Validate the concrete CHARLS wave-variable map and leakage roles.",
    )
    parser.add_argument(
        "--charls-incident-diabetes-cohort",
        action="store_true",
        help="Build the CHARLS 2011 baseline to 2013/2015 incident diabetes cohort skeleton.",
    )
    parser.add_argument(
        "--validate-charls-incident-diabetes-cohort",
        action="store_true",
        help="Validate the CHARLS incident diabetes cohort skeleton.",
    )
    parser.add_argument(
        "--charls-baseline-features",
        action="store_true",
        help="Build the CHARLS 2011 baseline feature matrix for incident diabetes.",
    )
    parser.add_argument(
        "--validate-charls-baseline-features",
        action="store_true",
        help="Validate the CHARLS 2011 baseline feature matrix.",
    )
    parser.add_argument(
        "--charls-leakage-gate",
        action="store_true",
        help="Run leakage gates for the CHARLS incident diabetes baseline slice.",
    )
    parser.add_argument(
        "--charls-logistic-baseline",
        action="store_true",
        help="Run a lightweight CHARLS incident diabetes balanced logistic regression baseline.",
    )
    parser.add_argument(
        "--validate-charls-logistic-baseline",
        action="store_true",
        help="Validate CHARLS incident diabetes logistic baseline outputs.",
    )
    parser.add_argument(
        "--charls-sensitivity",
        action="store_true",
        help="Run CHARLS incident diabetes baseline sensitivity analyses.",
    )
    parser.add_argument(
        "--validate-charls-sensitivity",
        action="store_true",
        help="Validate CHARLS incident diabetes sensitivity outputs.",
    )
    parser.add_argument(
        "--charls-model-comparison",
        action="store_true",
        help="Run CHARLS incident diabetes traditional model comparisons.",
    )
    parser.add_argument(
        "--validate-charls-model-comparison",
        action="store_true",
        help="Validate CHARLS incident diabetes model-comparison outputs.",
    )
    parser.add_argument(
        "--charls-calibration-decision",
        action="store_true",
        help="Summarize CHARLS incident diabetes calibration and decision-curve outputs.",
    )
    parser.add_argument(
        "--validate-charls-calibration-decision",
        action="store_true",
        help="Validate CHARLS incident diabetes calibration and decision-curve outputs.",
    )
    parser.add_argument(
        "--charls-probability-recalibration",
        action="store_true",
        help="Recalibrate CHARLS incident diabetes baseline probabilities on the validation split.",
    )
    parser.add_argument(
        "--validate-charls-probability-recalibration",
        action="store_true",
        help="Validate CHARLS incident diabetes probability recalibration outputs.",
    )
    parser.add_argument(
        "--external-readiness-summary",
        action="store_true",
        help="Summarize CDSL, eICU, and CHARLS external benchmark readiness.",
    )
    parser.add_argument(
        "--external-benchmark-summary",
        action="store_true",
        help="Build a concise CDSL/eICU external benchmark summary table.",
    )
    parser.add_argument(
        "--validate-external-benchmark-summary",
        action="store_true",
        help="Validate the concise CDSL/eICU external benchmark summary table.",
    )
    parser.add_argument(
        "--external-technical-summary",
        action="store_true",
        help="Build an external technical summary from hard metrics, subgroup CI, calibration, and decision curves.",
    )
    parser.add_argument(
        "--validate-external-technical-summary",
        action="store_true",
        help="Validate the external technical summary artifact.",
    )
    parser.add_argument(
        "--external-calibration-decision-summary",
        action="store_true",
        help="Build a unified external calibration and decision-curve comparison table.",
    )
    parser.add_argument(
        "--validate-external-calibration-decision-summary",
        action="store_true",
        help="Validate the unified external calibration and decision-curve comparison table.",
    )
    parser.add_argument(
        "--external-model-selection-rationale",
        action="store_true",
        help="Build a deterministic model-selection rationale table for external benchmark summary rows.",
    )
    parser.add_argument(
        "--validate-external-model-selection-rationale",
        action="store_true",
        help="Validate the external model-selection rationale table.",
    )
    parser.add_argument(
        "--external-metric-consistency-audit",
        action="store_true",
        help="Audit metric consistency across external summary, CI, calibration, and rationale tables.",
    )
    parser.add_argument(
        "--validate-external-metric-consistency-audit",
        action="store_true",
        help="Validate the external metric consistency audit.",
    )
    parser.add_argument(
        "--external-summary-asset-manifest",
        action="store_true",
        help="Build the formal external-summary asset manifest for mentor handoff.",
    )
    parser.add_argument(
        "--validate-external-summary-asset-manifest",
        action="store_true",
        help="Validate the formal external-summary asset manifest.",
    )
    parser.add_argument(
        "--external-handoff-package",
        action="store_true",
        help="Build the concrete external handoff package directory and zip archive.",
    )
    parser.add_argument(
        "--validate-external-handoff-package",
        action="store_true",
        help="Validate the concrete external handoff package directory and zip archive.",
    )
    parser.add_argument(
        "--external-subgroup-robustness-summary",
        action="store_true",
        help="Build a subgroup robustness summary for selected external benchmark rows.",
    )
    parser.add_argument(
        "--validate-external-subgroup-robustness-summary",
        action="store_true",
        help="Validate the selected-row external subgroup robustness summary.",
    )
    parser.add_argument(
        "--external-threshold-band-sensitivity",
        action="store_true",
        help="Build threshold-band sensitivity summaries for selected external decision curves.",
    )
    parser.add_argument(
        "--validate-external-threshold-band-sensitivity",
        action="store_true",
        help="Validate threshold-band sensitivity summaries for selected external decision curves.",
    )
    parser.add_argument(
        "--external-calibration-method-rationale",
        action="store_true",
        help="Build calibration-method rationale rows for selected external benchmark rows.",
    )
    parser.add_argument(
        "--validate-external-calibration-method-rationale",
        action="store_true",
        help="Validate calibration-method rationale rows for selected external benchmark rows.",
    )
    parser.add_argument(
        "--external-bootstrap-ci",
        action="store_true",
        help="Bootstrap test-set confidence intervals for CDSL, eICU, and CHARLS external benchmark predictions.",
    )
    parser.add_argument(
        "--validate-external-bootstrap-ci",
        action="store_true",
        help="Validate external benchmark bootstrap confidence interval outputs.",
    )
    parser.add_argument(
        "--external-subgroup-performance",
        action="store_true",
        help="Summarize CDSL, eICU, and CHARLS external subgroup performance.",
    )
    parser.add_argument(
        "--validate-external-subgroup-performance",
        action="store_true",
        help="Validate external subgroup performance outputs.",
    )
    parser.add_argument(
        "--external-subgroup-bootstrap-ci",
        action="store_true",
        help="Bootstrap confidence intervals for external subgroup performance.",
    )
    parser.add_argument(
        "--validate-external-subgroup-bootstrap-ci",
        action="store_true",
        help="Validate external subgroup bootstrap confidence interval outputs.",
    )
    parser.add_argument(
        "--external-model-comparison-recalibration",
        action="store_true",
        help="Recalibrate eICU/CHARLS RF and HGB model-comparison probabilities.",
    )
    parser.add_argument(
        "--validate-external-model-comparison-recalibration",
        action="store_true",
        help="Validate eICU/CHARLS RF and HGB model-comparison recalibration outputs.",
    )
    parser.add_argument(
        "--external-field-role-catalog",
        action="store_true",
        help="Build a unified eICU/CHARLS field-role catalog for planned external studies.",
    )
    parser.add_argument(
        "--validate-external-field-role-catalog",
        action="store_true",
        help="Validate the unified external field-role catalog.",
    )
    parser.add_argument(
        "--next-study-plan",
        action="store_true",
        help="Generate a local action plan for the next ChronoEHR-Agent study steps.",
    )
    parser.add_argument(
        "--validate-next-study-plan",
        action="store_true",
        help="Validate the local next-study action plan against current readiness state.",
    )
    parser.add_argument(
        "--study-capabilities",
        action="store_true",
        help="Audit study-level capabilities across completed, replication, and planned studies.",
    )
    parser.add_argument(
        "--single-study-drafts",
        action="store_true",
        help="Generate missing single-study Methods/Results drafts for replication chronic cohorts.",
    )
    parser.add_argument(
        "--pipeline-steps",
        action="store_true",
        help="Inspect registered study pipeline steps, outputs, and expensive steps without running them.",
    )
    parser.add_argument(
        "--report-presets",
        action="store_true",
        help="Discover manuscript/report export presets and generated Word outputs.",
    )
    parser.add_argument(
        "--asset-manifest",
        action="store_true",
        help="Build a manifest of reports, tables, figures, and Word assets for writing.",
    )
    parser.add_argument(
        "--agent-control",
        action="store_true",
        help="Build the local Agent control panel from registry, readiness, and audit state.",
    )
    parser.add_argument(
        "--diabetes-agent-demo",
        action="store_true",
        help="Run the configured one-click diabetes Agent demo workflow.",
    )
    parser.add_argument(
        "--diabetes-agent-demo-plan-only",
        action="store_true",
        help="With --diabetes-agent-demo, write the demo plan without executing commands.",
    )
    parser.add_argument(
        "--validate-agent-demo-workflow",
        action="store_true",
        help="Validate configured Agent demo workflows and the latest diabetes demo run.",
    )
    parser.add_argument(
        "--agent-entrypoints",
        action="store_true",
        help="Generate a concise command index for stable ChronoEHR-Agent entrypoints.",
    )
    parser.add_argument(
        "--agent-status-card",
        action="store_true",
        help="Generate a short human-readable status card from current Agent outputs.",
    )
    parser.add_argument(
        "--agent-progress-score",
        action="store_true",
        help="Compute a concise local MVP progress score for ChronoEHR-Agent.",
    )
    parser.add_argument(
        "--validate-agent-progress-score",
        action="store_true",
        help="Validate the generated Agent progress score.",
    )
    parser.add_argument(
        "--validate-agent-status-card",
        action="store_true",
        help="Validate the generated Agent status card.",
    )
    parser.add_argument(
        "--validate-agent-entrypoints",
        action="store_true",
        help="Validate stable Agent entrypoint config, commands, and generated outputs.",
    )
    parser.add_argument(
        "--validate-mainline-mvp",
        action="store_true",
        help="Validate the ChronoEHR-Agent v0.1 mainline MVP gate.",
    )
    parser.add_argument(
        "--config-coverage-audit",
        action="store_true",
        help="Audit which study definitions are config-driven and which runner parts remain hard-coded.",
    )
    parser.add_argument(
        "--config-migration-backlog",
        action="store_true",
        help="Generate prioritized backlog items for migrating hard-coded runner logic to config.",
    )
    parser.add_argument(
        "--validate-config-code-rules",
        action="store_true",
        help="Validate config-driven ICD code-rule loading for chronic disease cohorts.",
    )
    parser.add_argument(
        "--validate-study-config-schema",
        action="store_true",
        help="Validate registry study configs as workflow schemas, including planned external studies.",
    )
    parser.add_argument(
        "--agent-goal",
        default="status",
        help="Free-text goal passed to --agent-control, such as status, leakage, external, demo.",
    )
    parser.add_argument(
        "--agent-execute-safe-checks",
        action="store_true",
        help="With --agent-control, refresh lightweight self-checks before building the panel.",
    )
    parser.add_argument(
        "--validate-agent-control",
        action="store_true",
        help="Validate Agent control-panel goal routing.",
    )
    parser.add_argument(
        "--agent-task",
        help="Route a natural-language research task to local Agent actions.",
    )
    parser.add_argument(
        "--agent-task-risk-mode",
        choices=["safe", "expensive", "model", "report", "auto"],
        default="auto",
        help="Maximum action risk allowed for --agent-task.",
    )
    parser.add_argument(
        "--agent-task-execute-safe",
        action="store_true",
        help="With --agent-task, execute selected safe actions only.",
    )
    parser.add_argument(
        "--agent-task-post-run-refresh",
        action="store_true",
        help="With --agent-task --agent-task-execute-safe, refresh state, next tasks, handoff, and status outputs after execution.",
    )
    parser.add_argument(
        "--agent-task-scenarios",
        action="store_true",
        help="Export the natural-language Agent task scenario library.",
    )
    parser.add_argument(
        "--validate-agent-task-scenarios",
        action="store_true",
        help="Validate the natural-language Agent task scenario library.",
    )
    parser.add_argument(
        "--agent-task-queue",
        action="store_true",
        help="Build a safety-aware queue from Agent next-task recommendations.",
    )
    parser.add_argument(
        "--validate-agent-task-queue",
        action="store_true",
        help="Validate the safety-aware Agent task queue.",
    )
    parser.add_argument(
        "--agent-task-queue-run",
        action="store_true",
        help="Plan or execute READY_SAFE_AUTO items from the Agent task queue.",
    )
    parser.add_argument(
        "--agent-task-queue-execute-safe",
        action="store_true",
        help="With --agent-task-queue-run, execute READY_SAFE_AUTO queue items only.",
    )
    parser.add_argument(
        "--agent-task-queue-id",
        action="append",
        default=[],
        help="With --agent-task-queue-run, limit execution to a queue id such as Q003. Can be repeated.",
    )
    parser.add_argument(
        "--agent-task-queue-scenario",
        action="append",
        default=[],
        help="With --agent-task-queue-run, limit execution to a scenario id such as agent_control_focus. Can be repeated.",
    )
    parser.add_argument(
        "--validate-agent-task-queue-execution",
        action="store_true",
        help="Validate Agent task queue execution boundaries.",
    )
    parser.add_argument(
        "--validate-agent-cooldown-fingerprint",
        action="store_true",
        help="Validate Agent safe-auto cooldown fingerprint configuration.",
    )
    parser.add_argument(
        "--agent-runbook",
        help="Build a phased runbook for a longer natural-language Agent task.",
    )
    parser.add_argument(
        "--agent-runbook-execute-safe-phase",
        action="store_true",
        help="With --agent-runbook, execute only the safe phase.",
    )
    parser.add_argument(
        "--agent-runbook-execute-expensive-phase",
        action="store_true",
        help="With --agent-runbook, execute expensive non-model actions only when --confirm-expensive is set.",
    )
    parser.add_argument(
        "--confirm-expensive",
        action="store_true",
        help="Required confirmation for expensive non-model Agent runbook phase execution.",
    )
    parser.add_argument(
        "--agent-runbook-post-phase-refresh",
        action="store_true",
        help="With --agent-runbook phase execution, refresh recovery plan, next tasks, and state afterwards.",
    )
    parser.add_argument(
        "--validate-agent-runbook",
        action="store_true",
        help="Validate Agent runbook phase policies and execution boundaries.",
    )
    parser.add_argument(
        "--validate-agent-runbook-confirmation",
        action="store_true",
        help="Validate that expensive runbook execution requires explicit confirmation.",
    )
    parser.add_argument(
        "--agent-runbook-state",
        action="store_true",
        help="Build the Agent runbook phase state machine from current runbook outputs.",
    )
    parser.add_argument(
        "--validate-agent-runbook-state",
        action="store_true",
        help="Validate Agent runbook state-machine phase gates.",
    )
    parser.add_argument(
        "--agent-runbook-retry-plan",
        action="store_true",
        help="Plan retry/resume commands from the current runbook state machine.",
    )
    parser.add_argument(
        "--validate-agent-runbook-retry-plan",
        action="store_true",
        help="Validate Agent runbook retry/resume recommendations.",
    )
    parser.add_argument(
        "--agent-next-tasks",
        action="store_true",
        help="Recommend the next natural-language Agent tasks from current local state.",
    )
    parser.add_argument(
        "--validate-agent-next-tasks",
        action="store_true",
        help="Validate Agent next-task recommendations and safety boundaries.",
    )
    parser.add_argument(
        "--agent-state",
        action="store_true",
        help="Build a persistent local Agent state snapshot.",
    )
    parser.add_argument(
        "--validate-agent-state",
        action="store_true",
        help="Validate the persistent local Agent state snapshot.",
    )
    parser.add_argument(
        "--agent-self-check",
        action="store_true",
        help="Run lightweight health checks for the Agent itself.",
    )
    parser.add_argument(
        "--agent-doctor",
        action="store_true",
        help="Run the final local Agent health bundle: self-check, delivery readiness, and artifact freshness.",
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Run a no-data synthetic ChronoEHR demo suitable for GitHub CI and external review.",
    )
    parser.add_argument(
        "--validate-synthetic-demo",
        action="store_true",
        help="Validate no-data synthetic demo artifacts.",
    )
    parser.add_argument(
        "--validate-agent-doctor",
        action="store_true",
        help="Validate the Agent doctor health-check bundle and its last local output.",
    )
    parser.add_argument(
        "--agent-control-consistency",
        action="store_true",
        help="Audit consistency across Agent entrypoints, action catalog, self-checks, and readiness gates.",
    )
    parser.add_argument(
        "--agent-dependency-audit",
        action="store_true",
        help="Audit Agent control-layer dependencies to prevent circular gates.",
    )
    parser.add_argument(
        "--agent-doc-command-audit",
        action="store_true",
        help="Audit documented run_study commands in README and quickstart docs.",
    )
    parser.add_argument(
        "--agent-handoff-checklist",
        action="store_true",
        help="Validate the files and commands needed to resume ChronoEHR-Agent work.",
    )
    parser.add_argument(
        "--agent-recovery-plan",
        action="store_true",
        help="Plan minimal recovery actions from Agent self-check and readiness failures.",
    )
    parser.add_argument(
        "--agent-recovery-execute-safe",
        action="store_true",
        help="With --agent-recovery-plan, execute planned safe recovery commands.",
    )
    parser.add_argument(
        "--validate-agent-task-router",
        action="store_true",
        help="Validate natural-language Agent task routing.",
    )
    parser.add_argument(
        "--validate-agent-task-execution",
        action="store_true",
        help="Validate the latest natural-language Agent task execution and post-run refresh outputs.",
    )
    parser.add_argument(
        "--validate-agent-action-catalog",
        action="store_true",
        help="Validate the Agent action catalog used for risk-aware task routing.",
    )
    parser.add_argument(
        "--agent-command-lint",
        action="store_true",
        help="Lint Agent command strings across configs and state outputs.",
    )
    parser.add_argument(
        "--agent-boundary-audit",
        action="store_true",
        help="Audit project wording against medical-QA and clinical-advice drift.",
    )
    parser.add_argument(
        "--agent-artifact-freshness",
        action="store_true",
        help="Audit freshness of core Agent control artifacts against their producers.",
    )
    parser.add_argument(
        "--english-brief-draft",
        action="store_true",
        help="Generate an English Markdown Methods/Results brief from completed benchmark outputs.",
    )
    parser.add_argument(
        "--english-brief-audit",
        action="store_true",
        help="Audit the generated English brief for required sections and research-tool boundaries.",
    )
    parser.add_argument(
        "--english-brief-docx",
        action="store_true",
        help="Export the reviewed English Markdown brief as a DOCX draft.",
    )
    parser.add_argument(
        "--english-brief-docx-audit",
        action="store_true",
        help="Audit the English brief DOCX draft and rendered review pages.",
    )
    parser.add_argument(
        "--vital-features",
        action="store_true",
        help="Extract shared ICU vital-sign features for all chronic disease cohorts.",
    )
    parser.add_argument(
        "--procedure-features",
        action="store_true",
        help="Extract shared ICU procedure-event features for all chronic disease cohorts.",
    )
    parser.add_argument(
        "--general-med-features",
        action="store_true",
        help="Extract shared general medication features for all chronic disease cohorts.",
    )
    parser.add_argument(
        "--random-forest-baseline",
        action="store_true",
        help="Run or status-check optional Random Forest baselines.",
    )
    parser.add_argument(
        "--gradient-boosting-baseline",
        action="store_true",
        help="Run optional gradient boosting baselines, with sklearn HistGradientBoosting fallback.",
    )
    parser.add_argument(
        "--model-baseline-summary",
        action="store_true",
        help="Summarize logistic regression and Random Forest baselines across cohorts.",
    )
    parser.add_argument(
        "--model-calibration-summary",
        action="store_true",
        help="Generate calibration decile summaries for logistic and Random Forest baselines.",
    )
    parser.add_argument(
        "--feature-group-ablation",
        action="store_true",
        help="Summarize grouped feature contributions from the prediction-time benchmark.",
    )
    parser.add_argument(
        "--feature-selection-summary",
        action="store_true",
        help="Summarize fine-grained feature-selection signals from logistic coefficients.",
    )
    parser.add_argument(
        "--selected-feature-sets",
        action="store_true",
        help="Train selected-concept logistic models and compare them with full feature sets.",
    )
    parser.add_argument(
        "--selected-feature-calibration",
        action="store_true",
        help="Create calibration summaries and supplementary table for selected feature set models.",
    )
    parser.add_argument(
        "--ed-los-sensitivity",
        action="store_true",
        help="Run 24-hour model sensitivity analysis after removing ed_los_hours.",
    )
    parser.add_argument(
        "--threshold-analysis",
        action="store_true",
        help="Summarize fixed alert-burden threshold metrics for final models.",
    )
    parser.add_argument(
        "--decision-curve",
        action="store_true",
        help="Summarize decision-curve net benefit for final models.",
    )
    parser.add_argument(
        "--subgroup-analysis",
        action="store_true",
        help="Summarize subgroup performance for final models.",
    )
    parser.add_argument(
        "--summary-figures",
        action="store_true",
        help="Generate manuscript-oriented figures from completed summary tables.",
    )
    parser.add_argument(
        "--delivery-readiness",
        action="store_true",
        help="Audit whether the local ChronoEHR-Agent demo deliverables are ready to hand off.",
    )
    parser.add_argument(
        "--cross-cohort-draft",
        action="store_true",
        help="Generate a cross-cohort Methods/Results draft and manuscript-ready tables.",
    )
    parser.add_argument(
        "--supplementary-appendix",
        action="store_true",
        help="Generate a manuscript-ready supplementary appendix from completed outputs.",
    )
    parser.add_argument(
        "--manuscript-docx",
        action="store_true",
        help="Export completed manuscript drafts and supplementary appendix as DOCX files.",
    )
    parser.add_argument(
        "--manuscript-export-config",
        type=Path,
        help="Optional JSON config used with --manuscript-docx.",
    )
    parser.add_argument(
        "--calibrated-random-forest",
        action="store_true",
        help="Run post-hoc calibrated Random Forest baselines using validation-set calibration.",
    )
    parser.add_argument(
        "--calibrated-gradient-boosting",
        action="store_true",
        help="Run post-hoc calibrated HistGradientBoosting baselines using validation-set calibration.",
    )
    parser.add_argument(
        "--validate-diagnosis-builder",
        action="store_true",
        help="Validate the generic diagnosis-code cohort builder against existing cohorts.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Forwarded to the selected study pipeline.")
    parser.add_argument("--no-expensive", action="store_true", help="Forwarded to the selected study pipeline.")
    parser.add_argument("--only", nargs="*", help="Forwarded step names for the selected study pipeline.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_registry(args.registry)
    if args.synthetic_demo:
        script = args.project_root / "src" / "chrono_ehr" / "synthetic_demo.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_synthetic_demo:
        script = args.project_root / "src" / "chrono_ehr" / "validate_synthetic_demo.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.list:
        list_studies(registry)
        return
    if args.benchmark_summary:
        script = args.project_root / "src" / "chrono_ehr" / "summarize_chronic_benchmark.py"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--project-root",
                str(args.project_root),
                "--registry",
                str(args.registry),
            ],
            cwd=args.project_root,
            check=True,
        )
        return
    if args.progress_report:
        script = args.project_root / "src" / "chrono_ehr" / "generate_agent_progress_report.py"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--project-root",
                str(args.project_root),
                "--registry",
                str(args.registry),
            ],
            cwd=args.project_root,
            check=True,
        )
        return
    if args.validate_prediction_specs:
        script = args.project_root / "src" / "chrono_ehr" / "validate_prediction_time_specs.py"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--project-root",
                str(args.project_root),
            ],
            cwd=args.project_root,
            check=True,
        )
        return
    if args.leakage_gate:
        script = args.project_root / "src" / "chrono_ehr" / "leakage_gate.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_feature_windows:
        script = args.project_root / "src" / "chrono_ehr" / "validate_feature_window_specs.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.audit_extractor_windows:
        script = args.project_root / "src" / "chrono_ehr" / "audit_extractor_window_config_usage.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.check_model_deps:
        script = args.project_root / "src" / "chrono_ehr" / "check_model_dependencies.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.cdsl_readiness:
        script = args.project_root / "src" / "chrono_ehr" / "cdsl_external_validation_readiness.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.cdsl_temporal_benchmark:
        script = args.project_root / "src" / "chrono_ehr" / "cdsl_temporal_benchmark.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.cdsl_leakage_audit:
        script = args.project_root / "src" / "chrono_ehr" / "cdsl_leakage_audit.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.cdsl_traditional_baselines:
        script = args.project_root / "src" / "chrono_ehr" / "cdsl_traditional_baselines.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.cdsl_summary_figures:
        script = args.project_root / "src" / "chrono_ehr" / "plot_cdsl_benchmark_figures.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.cdsl_calibration_decision:
        script = args.project_root / "src" / "chrono_ehr" / "cdsl_calibration_decision_curve.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_cdsl_calibration_decision:
        script = args.project_root / "src" / "chrono_ehr" / "validate_cdsl_calibration_decision_curve.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.eicu_readiness:
        script = args.project_root / "src" / "chrono_ehr" / "eicu_data_readiness.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.eicu_cohort:
        script = args.project_root / "src" / "chrono_ehr" / "eicu_temporal_mortality_cohort.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_eicu_cohort:
        script = args.project_root / "src" / "chrono_ehr" / "validate_eicu_temporal_mortality_cohort.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.eicu_temporal_features:
        script = args.project_root / "src" / "chrono_ehr" / "eicu_temporal_features_24h.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_eicu_temporal_features:
        script = args.project_root / "src" / "chrono_ehr" / "validate_eicu_temporal_features_24h.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.eicu_leakage_gate:
        script = args.project_root / "src" / "chrono_ehr" / "eicu_leakage_gate.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.eicu_logistic_baseline:
        script = args.project_root / "src" / "chrono_ehr" / "eicu_first24h_logistic_baseline.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_eicu_logistic_baseline:
        script = args.project_root / "src" / "chrono_ehr" / "validate_eicu_first24h_baseline.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.eicu_model_comparison:
        script = args.project_root / "src" / "chrono_ehr" / "eicu_first24h_model_comparison.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_eicu_model_comparison:
        script = args.project_root / "src" / "chrono_ehr" / "validate_eicu_first24h_model_comparison.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.eicu_baseline_figures:
        script = args.project_root / "src" / "chrono_ehr" / "eicu_baseline_figures.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_eicu_baseline_figures:
        script = args.project_root / "src" / "chrono_ehr" / "validate_eicu_baseline_figures.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.eicu_probability_recalibration:
        script = args.project_root / "src" / "chrono_ehr" / "eicu_probability_recalibration.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_eicu_probability_recalibration:
        script = args.project_root / "src" / "chrono_ehr" / "validate_eicu_probability_recalibration.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.charls_readiness:
        script = args.project_root / "src" / "chrono_ehr" / "charls_data_readiness.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.charls_wave_map:
        script = args.project_root / "src" / "chrono_ehr" / "charls_wave_variable_map.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_charls_wave_map:
        script = args.project_root / "src" / "chrono_ehr" / "validate_charls_wave_variable_map.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.charls_incident_diabetes_cohort:
        script = args.project_root / "src" / "chrono_ehr" / "charls_incident_diabetes_cohort.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_charls_incident_diabetes_cohort:
        script = args.project_root / "src" / "chrono_ehr" / "validate_charls_incident_diabetes_cohort.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.charls_baseline_features:
        script = args.project_root / "src" / "chrono_ehr" / "charls_baseline_features.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_charls_baseline_features:
        script = args.project_root / "src" / "chrono_ehr" / "validate_charls_baseline_features.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.charls_leakage_gate:
        script = args.project_root / "src" / "chrono_ehr" / "charls_leakage_gate.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.charls_logistic_baseline:
        script = args.project_root / "src" / "chrono_ehr" / "charls_incident_diabetes_baseline.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_charls_logistic_baseline:
        script = args.project_root / "src" / "chrono_ehr" / "validate_charls_incident_diabetes_baseline.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.charls_sensitivity:
        script = args.project_root / "src" / "chrono_ehr" / "charls_incident_diabetes_sensitivity.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_charls_sensitivity:
        script = args.project_root / "src" / "chrono_ehr" / "validate_charls_incident_diabetes_sensitivity.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.charls_model_comparison:
        script = args.project_root / "src" / "chrono_ehr" / "charls_incident_diabetes_model_comparison.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_charls_model_comparison:
        script = args.project_root / "src" / "chrono_ehr" / "validate_charls_incident_diabetes_model_comparison.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.charls_calibration_decision:
        script = args.project_root / "src" / "chrono_ehr" / "charls_calibration_decision_curve.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_charls_calibration_decision:
        script = args.project_root / "src" / "chrono_ehr" / "validate_charls_calibration_decision_curve.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.charls_probability_recalibration:
        script = args.project_root / "src" / "chrono_ehr" / "charls_probability_recalibration.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_charls_probability_recalibration:
        script = args.project_root / "src" / "chrono_ehr" / "validate_charls_probability_recalibration.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.external_readiness_summary:
        script = args.project_root / "src" / "chrono_ehr" / "summarize_external_readiness.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.external_benchmark_summary:
        script = args.project_root / "src" / "chrono_ehr" / "external_benchmark_summary_table.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_external_benchmark_summary:
        script = args.project_root / "src" / "chrono_ehr" / "validate_external_benchmark_summary_table.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.external_technical_summary:
        script = args.project_root / "src" / "chrono_ehr" / "external_technical_summary.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_external_technical_summary:
        script = args.project_root / "src" / "chrono_ehr" / "validate_external_technical_summary.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.external_calibration_decision_summary:
        script = args.project_root / "src" / "chrono_ehr" / "external_calibration_decision_summary.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_external_calibration_decision_summary:
        script = args.project_root / "src" / "chrono_ehr" / "validate_external_calibration_decision_summary.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.external_model_selection_rationale:
        script = args.project_root / "src" / "chrono_ehr" / "external_model_selection_rationale.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_external_model_selection_rationale:
        script = args.project_root / "src" / "chrono_ehr" / "validate_external_model_selection_rationale.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.external_metric_consistency_audit:
        script = args.project_root / "src" / "chrono_ehr" / "external_metric_consistency_audit.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_external_metric_consistency_audit:
        script = args.project_root / "src" / "chrono_ehr" / "validate_external_metric_consistency_audit.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.external_summary_asset_manifest:
        script = args.project_root / "src" / "chrono_ehr" / "external_summary_asset_manifest.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_external_summary_asset_manifest:
        script = args.project_root / "src" / "chrono_ehr" / "validate_external_summary_asset_manifest.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.external_handoff_package:
        script = args.project_root / "src" / "chrono_ehr" / "external_handoff_package.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_external_handoff_package:
        script = args.project_root / "src" / "chrono_ehr" / "validate_external_handoff_package.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.external_subgroup_robustness_summary:
        script = args.project_root / "src" / "chrono_ehr" / "external_subgroup_robustness_summary.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_external_subgroup_robustness_summary:
        script = args.project_root / "src" / "chrono_ehr" / "validate_external_subgroup_robustness_summary.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.external_threshold_band_sensitivity:
        script = args.project_root / "src" / "chrono_ehr" / "external_threshold_band_sensitivity.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_external_threshold_band_sensitivity:
        script = args.project_root / "src" / "chrono_ehr" / "validate_external_threshold_band_sensitivity.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.external_calibration_method_rationale:
        script = args.project_root / "src" / "chrono_ehr" / "external_calibration_method_rationale.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_external_calibration_method_rationale:
        script = args.project_root / "src" / "chrono_ehr" / "validate_external_calibration_method_rationale.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.external_bootstrap_ci:
        script = args.project_root / "src" / "chrono_ehr" / "external_model_bootstrap_ci.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_external_bootstrap_ci:
        script = args.project_root / "src" / "chrono_ehr" / "validate_external_model_bootstrap_ci.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.external_subgroup_performance:
        script = args.project_root / "src" / "chrono_ehr" / "external_subgroup_performance.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_external_subgroup_performance:
        script = args.project_root / "src" / "chrono_ehr" / "validate_external_subgroup_performance.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.external_subgroup_bootstrap_ci:
        script = args.project_root / "src" / "chrono_ehr" / "external_subgroup_bootstrap_ci.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_external_subgroup_bootstrap_ci:
        script = args.project_root / "src" / "chrono_ehr" / "validate_external_subgroup_bootstrap_ci.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.external_model_comparison_recalibration:
        script = args.project_root / "src" / "chrono_ehr" / "external_model_comparison_recalibration.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_external_model_comparison_recalibration:
        script = args.project_root / "src" / "chrono_ehr" / "validate_external_model_comparison_recalibration.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.external_field_role_catalog:
        script = args.project_root / "src" / "chrono_ehr" / "external_field_role_catalog.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_external_field_role_catalog:
        script = args.project_root / "src" / "chrono_ehr" / "validate_external_field_role_catalog.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.next_study_plan:
        script = args.project_root / "src" / "chrono_ehr" / "next_study_planner.py"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--project-root",
                str(args.project_root),
                "--registry",
                str(args.registry),
            ],
            cwd=args.project_root,
            check=True,
        )
        return
    if args.validate_next_study_plan:
        script = args.project_root / "src" / "chrono_ehr" / "validate_next_study_plan.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.study_capabilities:
        script = args.project_root / "src" / "chrono_ehr" / "study_capability_audit.py"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--project-root",
                str(args.project_root),
                "--registry",
                str(args.registry),
            ],
            cwd=args.project_root,
            check=True,
        )
        return
    if args.single_study_drafts:
        script = args.project_root / "src" / "chrono_ehr" / "generate_single_study_methods_results.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.pipeline_steps:
        script = args.project_root / "src" / "chrono_ehr" / "pipeline_step_introspection.py"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--project-root",
                str(args.project_root),
                "--registry",
                str(args.registry),
            ],
            cwd=args.project_root,
            check=True,
        )
        return
    if args.report_presets:
        script = args.project_root / "src" / "chrono_ehr" / "report_preset_discovery.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.asset_manifest:
        script = args.project_root / "src" / "chrono_ehr" / "build_manuscript_asset_manifest.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.diabetes_agent_demo:
        script = args.project_root / "src" / "chrono_ehr" / "run_agent_demo_workflow.py"
        cmd = [sys.executable, str(script), "--project-root", str(args.project_root), "--workflow", "diabetes_v0"]
        if args.diabetes_agent_demo_plan_only:
            cmd.append("--plan-only")
        subprocess.run(cmd, cwd=args.project_root, check=True)
        return
    if args.validate_agent_demo_workflow:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_demo_workflow.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_entrypoints:
        script = args.project_root / "src" / "chrono_ehr" / "generate_agent_entrypoints.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_status_card:
        script = args.project_root / "src" / "chrono_ehr" / "agent_status_card.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_progress_score:
        script = args.project_root / "src" / "chrono_ehr" / "agent_progress_score.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_agent_progress_score:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_progress_score.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_agent_status_card:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_status_card.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_agent_entrypoints:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_entrypoints.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_mainline_mvp:
        script = args.project_root / "src" / "chrono_ehr" / "validate_mainline_mvp.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.config_coverage_audit:
        script = args.project_root / "src" / "chrono_ehr" / "audit_config_coverage.py"
        subprocess.run(
            [sys.executable, str(script), "--project-root", str(args.project_root), "--registry", str(args.registry)],
            cwd=args.project_root,
            check=True,
        )
        return
    if args.config_migration_backlog:
        script = args.project_root / "src" / "chrono_ehr" / "generate_config_migration_backlog.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_config_code_rules:
        script = args.project_root / "src" / "chrono_ehr" / "validate_config_code_rules.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_study_config_schema:
        script = args.project_root / "src" / "chrono_ehr" / "validate_study_config_schema.py"
        subprocess.run(
            [sys.executable, str(script), "--project-root", str(args.project_root), "--registry", str(args.registry)],
            cwd=args.project_root,
            check=True,
        )
        return
    if args.agent_control:
        script = args.project_root / "src" / "chrono_ehr" / "agent_control_panel.py"
        cmd = [sys.executable, str(script), "--project-root", str(args.project_root), "--registry", str(args.registry), "--goal", args.agent_goal]
        if args.agent_execute_safe_checks:
            cmd.append("--execute-safe-checks")
        subprocess.run(cmd, cwd=args.project_root, check=True)
        return
    if args.validate_agent_control:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_control_panel.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_task:
        script = args.project_root / "src" / "chrono_ehr" / "agent_task_router.py"
        cmd = [
            sys.executable,
            str(script),
            "--project-root",
            str(args.project_root),
            "--task",
            args.agent_task,
            "--risk-mode",
            args.agent_task_risk_mode,
        ]
        if args.agent_task_execute_safe:
            cmd.append("--execute-safe")
        if args.agent_task_post_run_refresh:
            cmd.append("--post-run-refresh")
        subprocess.run(cmd, cwd=args.project_root, check=True)
        return
    if args.agent_task_scenarios:
        script = args.project_root / "src" / "chrono_ehr" / "agent_task_scenario_library.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_agent_task_scenarios:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_task_scenarios.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_runbook:
        script = args.project_root / "src" / "chrono_ehr" / "agent_runbook.py"
        cmd = [
            sys.executable,
            str(script),
            "--project-root",
            str(args.project_root),
            "--task",
            args.agent_runbook,
        ]
        if args.agent_runbook_execute_safe_phase:
            cmd.append("--execute-safe-phase")
        if args.agent_runbook_execute_expensive_phase:
            cmd.append("--execute-expensive-phase")
        if args.confirm_expensive:
            cmd.append("--confirm-expensive")
        if args.agent_runbook_post_phase_refresh:
            cmd.append("--post-phase-refresh")
        subprocess.run(cmd, cwd=args.project_root, check=True)
        return
    if args.validate_agent_runbook:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_runbook.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_agent_runbook_confirmation:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_runbook_confirmation.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_runbook_state:
        script = args.project_root / "src" / "chrono_ehr" / "agent_runbook_state_machine.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_agent_runbook_state:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_runbook_state_machine.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_runbook_retry_plan:
        script = args.project_root / "src" / "chrono_ehr" / "agent_runbook_retry_planner.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_agent_runbook_retry_plan:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_runbook_retry_plan.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_next_tasks:
        script = args.project_root / "src" / "chrono_ehr" / "agent_next_task_planner.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_agent_next_tasks:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_next_tasks.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_task_queue:
        script = args.project_root / "src" / "chrono_ehr" / "agent_task_queue.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_agent_task_queue:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_task_queue.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_task_queue_run:
        script = args.project_root / "src" / "chrono_ehr" / "agent_task_queue_runner.py"
        cmd = [sys.executable, str(script), "--project-root", str(args.project_root)]
        if args.agent_task_queue_execute_safe:
            cmd.append("--execute-safe")
        for queue_id in args.agent_task_queue_id:
            cmd.extend(["--queue-id", queue_id])
        for scenario_id in args.agent_task_queue_scenario:
            cmd.extend(["--scenario-id", scenario_id])
        subprocess.run(cmd, cwd=args.project_root, check=True)
        return
    if args.validate_agent_task_queue_execution:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_task_queue_execution.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_agent_cooldown_fingerprint:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_cooldown_fingerprint.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_state:
        script = args.project_root / "src" / "chrono_ehr" / "build_agent_state.py"
        subprocess.run(
            [sys.executable, str(script), "--project-root", str(args.project_root), "--registry", str(args.registry)],
            cwd=args.project_root,
            check=True,
        )
        return
    if args.validate_agent_state:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_state.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_self_check:
        script = args.project_root / "src" / "chrono_ehr" / "agent_self_check.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_doctor:
        script = args.project_root / "src" / "chrono_ehr" / "agent_doctor.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_agent_doctor:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_doctor.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_control_consistency:
        script = args.project_root / "src" / "chrono_ehr" / "agent_control_consistency_audit.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_dependency_audit:
        script = args.project_root / "src" / "chrono_ehr" / "agent_dependency_audit.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_doc_command_audit:
        script = args.project_root / "src" / "chrono_ehr" / "agent_doc_command_audit.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_handoff_checklist:
        script = args.project_root / "src" / "chrono_ehr" / "agent_handoff_checklist.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_recovery_plan:
        script = args.project_root / "src" / "chrono_ehr" / "agent_recovery_planner.py"
        cmd = [sys.executable, str(script), "--project-root", str(args.project_root)]
        if args.agent_recovery_execute_safe:
            cmd.append("--execute-safe-recovery")
        subprocess.run(cmd, cwd=args.project_root, check=True)
        return
    if args.validate_agent_task_router:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_task_router.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_agent_task_execution:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_task_execution.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_agent_action_catalog:
        script = args.project_root / "src" / "chrono_ehr" / "validate_agent_action_catalog.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_command_lint:
        script = args.project_root / "src" / "chrono_ehr" / "agent_command_linter.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_boundary_audit:
        script = args.project_root / "src" / "chrono_ehr" / "agent_boundary_audit.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.agent_artifact_freshness:
        script = args.project_root / "src" / "chrono_ehr" / "agent_artifact_freshness.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.english_brief_draft:
        script = args.project_root / "src" / "chrono_ehr" / "generate_english_brief_methods_results.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.english_brief_audit:
        script = args.project_root / "src" / "chrono_ehr" / "audit_english_brief.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.english_brief_docx:
        script = args.project_root / "src" / "chrono_ehr" / "export_english_brief_docx.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.english_brief_docx_audit:
        script = args.project_root / "src" / "chrono_ehr" / "audit_english_brief_docx.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.vital_features:
        script = args.project_root / "src" / "chrono_ehr" / "mimic_chronic_vital_features.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.procedure_features:
        script = args.project_root / "src" / "chrono_ehr" / "mimic_chronic_procedure_features.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.general_med_features:
        script = args.project_root / "src" / "chrono_ehr" / "mimic_chronic_med_features.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.random_forest_baseline:
        script = args.project_root / "src" / "chrono_ehr" / "random_forest_baseline.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.gradient_boosting_baseline:
        script = args.project_root / "src" / "chrono_ehr" / "gradient_boosting_baseline.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.model_baseline_summary:
        script = args.project_root / "src" / "chrono_ehr" / "summarize_model_baselines.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.model_calibration_summary:
        script = args.project_root / "src" / "chrono_ehr" / "summarize_model_calibration.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.feature_group_ablation:
        script = args.project_root / "src" / "chrono_ehr" / "summarize_feature_group_ablation.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.feature_selection_summary:
        script = args.project_root / "src" / "chrono_ehr" / "summarize_feature_selection.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.selected_feature_sets:
        script = args.project_root / "src" / "chrono_ehr" / "run_selected_feature_sets.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.selected_feature_calibration:
        script = args.project_root / "src" / "chrono_ehr" / "summarize_selected_feature_calibration.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.ed_los_sensitivity:
        script = args.project_root / "src" / "chrono_ehr" / "run_ed_los_sensitivity.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.threshold_analysis:
        script = args.project_root / "src" / "chrono_ehr" / "summarize_threshold_analysis.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.decision_curve:
        script = args.project_root / "src" / "chrono_ehr" / "summarize_decision_curve.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.subgroup_analysis:
        script = args.project_root / "src" / "chrono_ehr" / "summarize_subgroup_analysis.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.summary_figures:
        script = args.project_root / "src" / "chrono_ehr" / "plot_summary_figures.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.delivery_readiness:
        script = args.project_root / "src" / "chrono_ehr" / "delivery_readiness_audit.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.cross_cohort_draft:
        script = args.project_root / "src" / "chrono_ehr" / "generate_cross_cohort_methods_results.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.supplementary_appendix:
        script = args.project_root / "src" / "chrono_ehr" / "generate_supplementary_appendix.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.manuscript_docx:
        script = args.project_root / "src" / "chrono_ehr" / "export_manuscript_docx.py"
        cmd = [sys.executable, str(script), "--project-root", str(args.project_root)]
        if args.manuscript_export_config is not None:
            cmd.extend(["--export-config", str(args.manuscript_export_config)])
        subprocess.run(cmd, cwd=args.project_root, check=True)
        return
    if args.calibrated_random_forest:
        script = args.project_root / "src" / "chrono_ehr" / "calibrated_random_forest_baseline.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.calibrated_gradient_boosting:
        script = args.project_root / "src" / "chrono_ehr" / "calibrated_gradient_boosting_baseline.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return
    if args.validate_diagnosis_builder:
        script = args.project_root / "src" / "chrono_ehr" / "validate_diagnosis_cohort_builder.py"
        subprocess.run([sys.executable, str(script), "--project-root", str(args.project_root)], cwd=args.project_root, check=True)
        return

    study_id = args.study or registry.get("active_study")
    studies = studies_by_id(registry)
    if study_id not in studies:
        available = ", ".join(studies) or "none"
        raise SystemExit(f"Unknown study id `{study_id}`. Available studies: {available}")

    study = studies[study_id]
    cmd = build_pipeline_command(study, args, args.project_root)
    print(f"Running study `{study_id}`", flush=True)
    print(f"Config: {study.get('config', 'NA')}", flush=True)
    print(f"Pipeline: {study['pipeline']}", flush=True)
    subprocess.run(cmd, cwd=args.project_root, check=True)


if __name__ == "__main__":
    main()
