"""Build V1-compatible and V2 industrial assembly diagnostic reports."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PHASES = ("transport_approach", "insertion_adjustment")
CATEGORIES = ("finger_object", "object_fixture", "hand_fixture")
RESULTS = ("first_pass", "provisional_first_pass", "recovered_success", "successful_not_first_attempt", "successful_unclassified", "failed")


def _first(events: list[dict], *names: str) -> dict | None:
    return next((event for event in events if event.get("event") in names), None)


def _safe_sub(end: float | None, start: float | None) -> float | None:
    return end - start if end is not None and start is not None and end >= start else None


def _correct_legacy_trace(trace_path: Path, first_contact_s: float, seat_s: float | None, deadband_m: float) -> tuple[float, float, int] | None:
    if not trace_path.is_file():
        return None
    with np.load(trace_path) as data:
        columns, trace = data["columns"].tolist(), data["trace"]
    if "time_s" not in columns or "depth_m" not in columns:
        return None
    time, depth = trace[:, columns.index("time_s")], trace[:, columns.index("depth_m")]
    mask = time >= first_contact_s
    if seat_s is not None:
        mask &= time <= seat_s
    depth = depth[mask]
    if depth.size < 2:
        return None
    raw, current, runs = 0.0, 0.0, []
    for delta in np.diff(depth):
        if delta < 0.0:
            current += float(-delta)
            raw += float(-delta)
        elif current:
            runs.append(current)
            current = 0.0
    if current:
        runs.append(current)
    meaningful = [distance for distance in runs if distance > deadband_m]
    return raw, sum(distance - deadband_m for distance in meaningful), len(meaningful)


def _threshold(metadata: dict, old: str, new: str, default: float) -> float:
    values = metadata.get("thresholds", {})
    return float(values.get(new, values.get(old, default)))


def _normalize_episode(episode: dict, events: list[dict], metadata: dict, problem_dir: Path) -> dict:
    row, quality = dict(episode), []
    schema = int(metadata.get("schema_version", 1))
    problem = str(row.get("problem", metadata.get("problem", "unknown")))
    stable = _first(events, "grasp_established", "stable_grasp")
    lifted = _first(events, "object_lifted")
    contact = _first(events, "first_fixture_contact")
    seated = _first(events, "assembly_completed", "assembly_seated")
    grasp_s = float(stable["time_s"]) if stable else None
    lift_s = float(lifted["time_s"]) if lifted else None
    contact_s = float(contact["time_s"]) if contact else None
    seat_s = float(seated["time_s"]) if seated else None

    attempt_value = row.get("insertion_attempt_count", row.get("attempt_count", 0))
    attempts = None if attempt_value is None else int(attempt_value)
    retry_value = row.get("insertion_retry_count", None if attempts is None else max(0, attempts - 1))
    retries = None if retry_value is None else int(retry_value)
    drops, regrasp = int(row.get("drop_count", 0)), int(row.get("regrasp_count", 0))
    restarts = int(row.get("task_restart_count", 0))
    recoveries = int(row.get("recovery_event_count", row.get("task_recovery_count", 0)))
    reported_v1_first_pass = bool(
        row.get("task_success") and attempts == 1 and not drops and not regrasp and not recoveries
    )
    reported_v1_recovery = recoveries > 0
    row.update(attempt_count=attempts, insertion_attempt_count=attempts, insertion_retry_count=retries,
               drop_count=drops, regrasp_count=regrasp, task_restart_count=restarts,
               task_restart_observable=bool(row.get("task_restart_observable", False)),
               local_adjustment_count=int(row.get("local_adjustment_count", 0)),
               recovery_event_count=recoveries, task_recovery_count=recoveries)
    row.update(first_pass_reported_v1=reported_v1_first_pass,
               recovery_reported_v1=reported_v1_recovery)

    if schema < 3:
        quality += [
            "legacy_schema_missing_v2_state_and_peak_timestamps",
            "legacy_contact_phase_boundaries_not_recomputable",
            "legacy_contact_force_aggregation_differs_from_v2_total_normal_load",
        ]
        row["regrasp_count_v1"] = regrasp
        row["task_recovery_count_v1"] = recoveries
        legacy_backtracks = recoveries
        pending_drop, confirmed_regrasp = False, 0
        for event in events:
            if event.get("event") == "drop":
                pending_drop = True
            elif event.get("event") == "regrasp" and pending_drop:
                confirmed_regrasp += 1
                pending_drop = False
        regrasp = confirmed_regrasp
        recoveries += drops
        row.update(regrasp_count=regrasp, recovery_event_count=recoveries,
                   task_recovery_count=recoveries)
        quality.append("legacy_regrasp_recomputed_as_drop_then_grasp")
        row["recovery_reasons"] = [reason for present, reason in (
            (legacy_backtracks > 0, "legacy_insertion_backtrack"), (drops > 0, "object_drop"),
            (restarts > 0, "task_restart")) if present]
        if contact_s is not None:
            trace_path = problem_dir / "traces" / f"env{int(row['env_id']):04d}_ep{int(row['episode']):04d}.npz"
            corrected = _correct_legacy_trace(trace_path, contact_s, seat_s, _threshold(metadata, "micro_backout_m", "micro_backout_m", 0.0005))
            if corrected:
                row["reverse_distance_raw_m"], row["reverse_distance_effective_m"], row["backoff_count"] = corrected
                quality.append("legacy_reverse_recomputed_from_control_rate_trace")
    else:
        row.setdefault("recovery_reasons", [])

    task_ok, assembly_ok = bool(row.get("task_success")), bool(row.get("assembly_success"))
    recovery_free = (
        None if problem == "screwing" and not (recoveries or drops or regrasp or restarts)
        else task_ok and not (recoveries or drops or regrasp or restarts)
    )
    insertion_observed = attempts is not None and attempts > 0 and problem != "screwing"
    classification = (
        "unverified_screwing_engagement" if problem == "screwing"
        else ("unknown_no_attempt" if not insertion_observed
              else ("verified" if schema >= 3 else "provisional_legacy_detector"))
    )
    first_attempt = assembly_ok and attempts == 1 and retries == 0 if insertion_observed else None
    first_pass = task_ok and bool(first_attempt) and bool(recovery_free) if insertion_observed else None
    row.update(first_attempt_insertion_success=first_attempt, recovery_free_success=recovery_free,
               first_pass_success=first_pass, first_pass_classification=classification,
               first_pass_provisional=first_pass if classification == "provisional_legacy_detector" else None,
               success_without_drop_regrasp_recovery=recovery_free,
               first_pass_v1=bool(recovery_free and attempts == 1), first_pass=first_pass)
    if not task_ok:
        result = "failed"
    elif first_pass is True:
        result = "first_pass" if classification == "verified" else "provisional_first_pass"
    elif first_pass is None:
        result = "successful_unclassified"
    elif recoveries or drops or regrasp or restarts:
        result = "recovered_success"
    else:
        result = "successful_not_first_attempt"
    row["result_code"] = result

    duration = float(row.get("duration_s", 0.0))
    row.setdefault("task_completion_time_s", duration if task_ok else None)
    row.setdefault("failure_observed_duration_s", None if task_ok else duration)
    row.setdefault("grasp_acquisition_time_s", grasp_s)
    row.setdefault("object_lift_time_s", lift_s)
    row.setdefault("first_contact_time_s", contact_s)
    row.setdefault("transport_duration_s", row.get("transport_approach_s"))
    row.setdefault("contact_to_success_duration_s", _safe_sub(seat_s, contact_s))
    row.setdefault("insertion_duration_s", _safe_sub(seat_s, contact_s))
    row.setdefault("recovery_duration_s", None if schema < 3 else 0.0)
    row.setdefault("insertion_backtrack_distance_m", row.get("reverse_distance_effective_m"))

    if task_ok and attempts == 0:
        quality.append("success_with_zero_valid_insertion_attempts")
    if problem == "screwing":
        quality.append("screwing_first_pass_unverified_without_thread_engagement_signal")
    if not row["task_restart_observable"]:
        quality.append("task_restart_not_observable_inside_first_episode_eval")
    if schema < 3 and row.get("trace_saved"):
        quality.append("legacy_trace_missing_relative_pose_joint_action_and_screw_signals")
    if contact_s is not None and grasp_s is not None and contact_s < grasp_s:
        quality.append("event_time_contradiction:first_contact_before_grasp")
    if seat_s is not None and contact_s is not None and seat_s < contact_s:
        quality.append("event_time_contradiction:assembly_before_first_contact")

    begins, ends, zero_points = Counter(), Counter(), set()
    for event in events:
        key = (event.get("phase"), event.get("category"))
        if event.get("event") == "contact_begin":
            begins[key] += 1
            point = event.get("point_w_m", ())
            if point and max(abs(float(value)) for value in point) < 1e-9:
                zero_points.add(str(event.get("category")))
        elif event.get("event") == "contact_end":
            ends[key] += 1
    if zero_points:
        quality.append("invalid_contact_points:" + ",".join(sorted(zero_points)))
    open_contacts = sum(max(0, begins[key] - ends[key]) for key in begins)
    if open_contacts:
        quality.append(f"{open_contacts}_contact_episode_open_at_rollout_end")

    contacts = row.get("contact_metrics", {})
    peak, high_s = 0.0, 0.0
    anomaly_n = _threshold(metadata, "anomalous_force_N", "anomalous_force_threshold_n", 500.0)
    anomaly_ns = _threshold(metadata, "anomalous_impulse_Ns", "anomalous_impulse_threshold_ns", 100.0)
    for phase in PHASES:
        for category in CATEGORIES:
            metric = contacts.get(phase, {}).get(category, {})
            force = float(metric.get("peak_normal_N", 0.0))
            impulse = float(metric.get("max_contact_episode_impulse_Ns", 0.0))
            peak, high_s = max(peak, force), high_s + float(metric.get("high_force_time_s", 0.0))
            if force >= anomaly_n:
                when = metric.get("peak_time_s")
                quality.append(f"extreme_contact_force:{phase}:{category}:{force:.3f}N:" + (f"t={float(when):.6f}s" if when is not None else "time_unavailable"))
            if impulse >= anomaly_ns:
                quality.append(f"extreme_contact_event_impulse:{phase}:{category}:{impulse:.3f}Ns")
    row.update(peak_normal_N=peak, high_force_time_s=high_s,
               contact_safety_code="threshold_exceeded" if high_s else "within_threshold")
    messages = {
        "first_pass": "任务成功；V2确认首次插入且无明确恢复事件",
        "provisional_first_pass": "任务成功；旧检测器暂定首次插入且无明确恢复事件",
        "recovered_success": "任务成功；检测到明确恢复事件",
        "successful_not_first_attempt": "任务成功；未发生明确恢复，但不是首次插入通过",
        "successful_unclassified": "任务成功；一次通过分类不可用，需人工/新日志验证",
        "failed": "任务未成功完成",
    }
    row["conclusion_zh"] = f"{messages[result]}；局部调整{row['local_adjustment_count']}次；峰值接触载荷{peak:.2f} N"
    row["data_quality"] = sorted(set(quality))
    row.update(_schema_version=schema, _problem_dir=str(problem_dir), _events=events)
    return row


def _load(roots: list[Path]) -> list[dict]:
    files = sorted({path for root in roots for path in root.rglob("episodes.jsonl")})
    if not files:
        raise FileNotFoundError(f"no episodes.jsonl found under {roots}")
    rows = []
    for episode_file in files:
        problem_dir = episode_file.parent
        metadata_path, events_path = problem_dir / "metadata.json", problem_dir / "events.jsonl"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        by_episode = defaultdict(list)
        if events_path.is_file():
            for line in events_path.read_text().splitlines():
                event = json.loads(line)
                by_episode[(int(event["env_id"]), int(event["episode"]))].append(event)
        for line in episode_file.read_text().splitlines():
            episode = json.loads(line)
            key = (int(episode["env_id"]), int(episode["episode"]))
            events = sorted(by_episode[key], key=lambda item: float(item["time_s"]))
            rows.append(_normalize_episode(episode, events, metadata, problem_dir))
    return rows


def _contact_rows(rows: list[dict]) -> list[dict]:
    return [{"problem": row["problem"], "env_id": row["env_id"], "episode": row["episode"],
             "phase": phase, "contact_pair": category,
             **row.get("contact_metrics", {}).get(phase, {}).get(category, {})}
            for row in rows for phase in PHASES for category in CATEGORIES]


def _stat(values) -> dict:
    array = np.asarray([float(v) for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not array.size:
        return {key: None for key in ("n", "mean", "p50", "p90", "p95", "std")}
    return {"n": int(array.size), "mean": float(array.mean()), "p50": float(np.percentile(array, 50)),
            "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95)), "std": float(array.std())}


def _rate(rows: list[dict], numerator, eligible=lambda _: True) -> dict:
    selected = [row for row in rows if eligible(row)]
    count = sum(bool(numerator(row)) for row in selected)
    return {"numerator": count, "denominator": len(selected), "value": count / len(selected) if selected else None}


def _problem_summary(rows: list[dict]) -> dict:
    rates = {
        "reported_v1_first_pass_rate": {**_rate(rows, lambda r: r["first_pass_reported_v1"]), "definition": "original report definition / all rollouts"},
        "reported_v1_recovery_rate": {**_rate(rows, lambda r: r["recovery_reported_v1"]), "definition": "original task_recovery_count > 0 / all rollouts"},
        "recovery_event_rate": {**_rate(rows, lambda r: int(r["recovery_event_count"]) > 0), "definition": "explicit recovery initiation detected / all rollouts"},
        "task_success_rate": {**_rate(rows, lambda r: r["task_success"]), "definition": "task_success / all rollouts"},
        "first_attempt_insertion_rate": {**_rate(rows, lambda r: r["first_attempt_insertion_success"] is True, lambda r: r["first_pass_classification"] == "verified"), "definition": "assembly succeeded on one valid attempt / rollouts with verified insertion classification"},
        "recovery_free_success_rate": {**_rate(rows, lambda r: r["recovery_free_success"] is True, lambda r: r["recovery_free_success"] is not None), "definition": "task success with no detected recovery/drop/regrasp / rollouts with recovery-free classification available; task restart is not observable"},
        "first_pass_success_rate": {**_rate(rows, lambda r: r["first_pass_success"] is True, lambda r: r["first_pass_classification"] == "verified"), "definition": "task success on first valid attempt and recovery-free / rollouts with verified V2 first-pass classification"},
        "legacy_provisional_first_pass_rate": {**_rate(rows, lambda r: r.get("first_pass_provisional") is True, lambda r: r["first_pass_classification"] == "provisional_legacy_detector"), "definition": "provisional only: recomputed recovery semantics but legacy insertion detector / legacy-classifiable rollouts"},
    }
    efficiency = {}
    fields = ("task_completion_time_s", "grasp_acquisition_time_s", "transport_duration_s", "first_contact_time_s",
              "contact_to_success_duration_s", "insertion_duration_s", "recovery_duration_s", "insertion_backtrack_distance_m")
    for field in fields:
        efficiency[field] = {"successful": _stat(row.get(field) for row in rows if row["task_success"]),
                             "failed": _stat(row.get(field) for row in rows if not row["task_success"])}
    efficiency["failure_observed_duration_s"] = {"successful": _stat([]), "failed": _stat(row.get("failure_observed_duration_s") for row in rows if not row["task_success"])}
    return {"rollouts": len(rows), "rates": rates, "result_counts": dict(Counter(row["result_code"] for row in rows)),
            "data_quality_issue_rollouts": sum(bool(row["data_quality"]) for row in rows), "efficiency": efficiency}


def _contact_summary(contact_rows: list[dict]) -> dict:
    contacts = {}
    for phase in PHASES:
        contacts[phase] = {}
        for category in CATEGORIES:
            selected = [row for row in contact_rows if row["phase"] == phase and row["contact_pair"] == category]
            peaks = [float(row.get("peak_normal_N", 0.0)) for row in selected]
            over = sum(float(row.get("high_force_time_s", 0.0)) > 0 for row in selected)
            contacts[phase][category] = {
                "max_peak_normal_N": max(peaks, default=0.0), "median_peak_normal_N": float(np.median(peaks)) if peaks else 0.0,
                "p95_peak_normal_N": float(np.percentile(peaks, 95)) if peaks else 0.0, "mean_peak_normal_N": float(np.mean(peaks)) if peaks else 0.0,
                "rollouts_over_high_force_threshold": over, "over_high_force_threshold_rate": over / len(selected) if selected else None,
                "total_high_force_time_s": sum(float(row.get("high_force_time_s", 0.0)) for row in selected),
                "max_contact_event_impulse_Ns": max((float(row.get("max_contact_episode_impulse_Ns", 0.0)) for row in selected), default=0.0),
            }
    return contacts


def _summary(rows: list[dict], contact_rows: list[dict]) -> dict:
    problems = sorted({row["problem"] for row in rows})
    by_problem = {problem: _problem_summary([row for row in rows if row["problem"] == problem]) for problem in problems}
    contact_by_problem = {
        problem: _contact_summary([row for row in contact_rows if row["problem"] == problem])
        for problem in problems
    }
    contacts = _contact_summary(contact_rows)
    overall = _problem_summary(rows)
    verified = [row for row in rows if row["first_pass_classification"] == "verified"]
    return {"schema_version": 3, "rollouts": len(rows), "completed": sum(bool(row.get("completed")) for row in rows),
            "assembly_success_rate": sum(bool(row.get("assembly_success")) for row in rows) / len(rows),
            "task_success_rate": overall["rates"]["task_success_rate"]["value"],
            "first_pass_rate": sum(row["first_pass_success"] is True for row in verified) / len(verified) if verified else None,
            "recovery_episode_rate": sum(int(row["recovery_event_count"]) > 0 for row in rows) / len(rows),
            "drop_episode_rate": sum(int(row["drop_count"]) > 0 for row in rows) / len(rows),
            "regrasp_episode_rate": sum(int(row["regrasp_count"]) > 0 for row in rows) / len(rows),
            "rates_v2": overall["rates"], "result_counts": overall["result_counts"],
            "data_quality_issue_rollouts": overall["data_quality_issue_rollouts"], "by_problem": by_problem,
            "efficiency": overall["efficiency"], "contact_metrics": contacts,
            "contact_metrics_by_problem": contact_by_problem,
            "limitations": ["task_restart_count is unavailable inside first-episode evaluation unless an explicit restart signal is added",
                            "screwing first-pass remains unverified without a thread engagement signal",
                            "legacy runs lack peak timestamps and mechanism-analysis traces except explicitly stored fields"]}


def _csv_value(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (list, dict)) else value


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: _csv_value(row.get(key)) for key in fields} for row in rows)


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _bar_chart(items: list[tuple[str, float | None]], percent=True) -> str:
    valid = [v for _, v in items if v is not None]
    scale, rows = (1.0 if percent else max(valid, default=1.0)), []
    for idx, (label, value) in enumerate(items):
        y, bar = idx * 30, 0 if value is None else 480 * value / max(scale, 1e-9)
        text = "N/A" if value is None else (_pct(value) if percent else f"{value:.0f}")
        rows.append(f"<text x='0' y='{y+19}'>{html.escape(label)}</text><rect x='180' y='{y+5}' width='{bar:.1f}' height='18' rx='3' fill='#3b82f6'/><text x='670' y='{y+19}' text-anchor='end'>{text}</text>")
    return f"<svg class='chart' viewBox='0 0 700 {max(30,len(items)*30)}'>{''.join(rows)}</svg>"


def _hist_chart(values: list[float], title: str) -> str:
    if not values:
        return f"<div class='plot'><b>{html.escape(title)}</b><p>N/A</p></div>"
    counts, edges = np.histogram(values, bins=min(12, max(3, int(math.sqrt(len(values))))))
    bars = [f"<rect x='{35+i*420/len(counts):.1f}' y='{145-120*n/max(counts.max(),1):.1f}' width='{390/len(counts):.1f}' height='{120*n/max(counts.max(),1):.1f}' fill='#10b981'/>" for i, n in enumerate(counts)]
    return f"<div class='plot'><b>{html.escape(title)}</b><svg viewBox='0 0 480 180'>{''.join(bars)}<text x='35' y='165'>{edges[0]:.2f}</text><text x='455' y='165' text-anchor='end'>{edges[-1]:.2f}</text></svg></div>"


def _trace_payload(rows: list[dict]) -> list[dict]:
    payload = []
    wanted = {"goal_translation_error_m": "goal_translation_error_m", "goal_rotation_error_rad": "goal_rotation_error_rad",
              "palm_object_translation_change_m": "palm_object_translation_change_m", "palm_object_rotation_change_rad": "palm_object_rotation_change_rad",
              "object_fixture_normal_N": "object_fixture_normal_N", "arm_action_l2": "arm_action_l2", "hand_action_l2": "hand_action_l2",
              "insertion_progress_m": "depth_m", "lateral_error_m": "lateral_error_m"}
    event_names = {"grasp_established", "stable_grasp", "object_lifted", "first_fixture_contact", "insertion_started", "attempt_start",
                   "insertion_backtracked", "insertion_retried", "object_dropped", "object_regrasped", "goal_reached", "assembly_completed", "assembly_seated"}
    for row in rows:
        path = Path(row["_problem_dir"]) / "traces" / f"env{int(row['env_id']):04d}_ep{int(row['episode']):04d}.npz"
        if not path.is_file():
            continue
        with np.load(path) as data:
            columns, trace = [str(v) for v in data["columns"].tolist()], data["trace"]
        if "time_s" not in columns:
            continue
        trace = trace[::max(1, math.ceil(len(trace) / 1000))]
        signals = {label: trace[:, columns.index(column)].astype(float).tolist() for label, column in wanted.items() if column in columns}
        payload.append({"id": f"{row['problem']} / env {row['env_id']} / ep {row['episode']}",
                        "time": trace[:, columns.index("time_s")].astype(float).tolist(), "signals": signals,
                        "events": [{"t": float(event["time_s"]), "name": event["event"]} for event in row["_events"] if event.get("event") in event_names],
                        "missing_mechanism_signals": not {"palm_object_translation_change_m", "arm_action_l2"}.issubset(signals)})
    return payload


def _write_html(path: Path, rows: list[dict], summary: dict) -> None:
    rates = summary["rates_v2"]
    cards = [("Rollouts", str(summary["rollouts"])), ("任务成功率", _pct(rates["task_success_rate"]["value"])),
             ("首次插入成功率", _pct(rates["first_attempt_insertion_rate"]["value"])),
             ("无恢复成功率", _pct(rates["recovery_free_success_rate"]["value"])),
             ("一次通过成功率", _pct(rates["first_pass_success_rate"]["value"]))]
    comparison, bars = [], []
    for problem, values in summary["by_problem"].items():
        r = values["rates"]
        max_force = max(float(row["peak_normal_N"]) for row in rows if row["problem"] == problem)
        comparison.append("<tr>" + f"<td>{html.escape(problem)}</td><td>{values['rollouts']}</td>" + "".join(f"<td>{_pct(r[k]['value'])}<small>{r[k]['numerator']}/{r[k]['denominator']}</small></td>" for k in ("task_success_rate", "reported_v1_first_pass_rate", "first_attempt_insertion_rate", "recovery_free_success_rate", "first_pass_success_rate", "legacy_provisional_first_pass_rate", "reported_v1_recovery_rate", "recovery_event_rate")) + f"<td>{max_force:.2f}</td></tr>")
        bars += [(f"{problem} · task", r["task_success_rate"]["value"]), (f"{problem} · provisional first-pass", r["legacy_provisional_first_pass_rate"]["value"]), (f"{problem} · V2 first-pass", r["first_pass_success_rate"]["value"])]
    counts = Counter(row["result_code"] for row in rows)
    distributions = _bar_chart([(name, float(counts[name])) for name in RESULTS], False) + "<div class='plots'>" + "".join((
        _hist_chart([float(r["insertion_attempt_count"]) for r in rows if r["insertion_attempt_count"] is not None], "插入尝试次数"),
        _hist_chart([float(r["local_adjustment_count"]) for r in rows], "局部调整次数"),
        _hist_chart([float(r["task_completion_time_s"]) for r in rows if r.get("task_completion_time_s") is not None], "成功完成时间 / s"),
        _hist_chart([float(r["peak_normal_N"]) for r in rows], "峰值接触载荷 / N"))) + "</div>"
    contact_table = "".join(f"<tr><td>{problem}</td><td>{phase}</td><td>{category}</td><td>{m['max_peak_normal_N']:.2f}</td><td>{m['median_peak_normal_N']:.2f}</td><td>{m['p95_peak_normal_N']:.2f}</td><td>{_pct(m['over_high_force_threshold_rate'])}</td><td>{m['max_contact_event_impulse_Ns']:.3f}</td></tr>" for problem, phases in summary["contact_metrics_by_problem"].items() for phase, cats in phases.items() for category, m in cats.items())
    efficiency = "".join(f"<tr><td>{metric}</td><td>{group}</td><td>{s['n'] or 0}</td>" + "".join(f"<td>{_num(s[k])}</td>" for k in ("mean", "p50", "p90", "p95", "std")) + "</tr>" for metric, groups in summary["efficiency"].items() for group, s in groups.items())
    rollout_table = "".join("<tr>" + f"<td>{html.escape(str(r['problem']))}</td><td>{r['env_id']}</td><td>{r['result_code']}</td><td>{'N/A' if r['insertion_attempt_count'] is None else r['insertion_attempt_count']}</td><td>{'N/A' if r['insertion_retry_count'] is None else r['insertion_retry_count']}</td><td>{r['local_adjustment_count']}</td><td>{r['recovery_event_count']}</td><td>{_num(r.get('task_completion_time_s'))}</td><td>{r['peak_normal_N']:.2f}</td><td>{html.escape('; '.join(r['data_quality']) or 'ok')}</td></tr>" for r in rows)
    traces = _trace_payload(rows)
    trace_json = json.dumps(traces, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    options = "".join(f"<option value='{i}'>{html.escape(t['id'])}</option>" for i, t in enumerate(traces))
    quality_counts = Counter(item.split(":", 1)[0] for row in rows for item in row["data_quality"])
    quality = "".join(f"<li><b>{n}</b> × {html.escape(name)}</li>" for name, n in quality_counts.most_common()) or "<li>未发现警告</li>"
    trace_section = (f"<select id='traceSelect'>{options}</select><p id='traceWarning'></p>" + "".join(f"<h3>{title}</h3><canvas id='plot{i}' width='1200' height='220'></canvas>" for i, title in enumerate(("目标位姿误差", "手—物相对位姿变化", "物体—工装接触载荷", "机械臂与手指动作强度", "插入进度")))) if traces else "<p>没有保存可用时间序列；需要重新运行并选择 diagnostics trace env。</p>"
    script = ""
    if traces:
        script = "<script>const traces=" + trace_json + ";const groups=[[\"goal_translation_error_m\",\"goal_rotation_error_rad\"],[\"palm_object_translation_change_m\",\"palm_object_rotation_change_rad\"],[\"object_fixture_normal_N\"],[\"arm_action_l2\",\"hand_action_l2\"],[\"insertion_progress_m\",\"lateral_error_m\"]];function draw(canvas,t,signals,events){const c=canvas.getContext('2d'),W=canvas.width,H=canvas.height;c.clearRect(0,0,W,H);c.strokeStyle='#ccd5df';c.strokeRect(45,10,W-55,H-35);const vals=Object.values(signals).flat().filter(Number.isFinite);if(!t.length||!vals.length){c.fillText('signal unavailable',55,35);return}const lo=Math.min(...vals),hi=Math.max(...vals),span=hi-lo||1,t0=t[0],ts=(t[t.length-1]-t0)||1;events.forEach(e=>{const x=45+(W-55)*(e.t-t0)/ts;c.strokeStyle='#d1d5db';c.beginPath();c.moveTo(x,10);c.lineTo(x,H-25);c.stroke()});const colors=['#2563eb','#dc2626','#059669'];Object.entries(signals).forEach(([name,v],j)=>{c.strokeStyle=colors[j%3];c.beginPath();v.forEach((y,i)=>{const x=45+(W-55)*(t[i]-t0)/ts,py=10+(H-35)*(hi-y)/span;i?c.lineTo(x,py):c.moveTo(x,py)});c.stroke();c.fillStyle=colors[j%3];c.fillText(name,55+j*260,25)});c.fillStyle='#111827';c.fillText(lo.toPrecision(3),2,H-27);c.fillText(hi.toPrecision(3),2,18);c.fillText(t0.toFixed(2)+'s',45,H-7);c.fillText(t[t.length-1].toFixed(2)+'s',W-55,H-7)}function render(){const tr=traces[+document.getElementById('traceSelect').value];document.getElementById('traceWarning').textContent=tr.missing_mechanism_signals?'该旧轨迹缺少手—物相对位姿/动作信号，需重跑 V2。':'';groups.forEach((g,i)=>{const s={};g.forEach(k=>{if(tr.signals[k])s[k]=tr.signals[k]});draw(document.getElementById('plot'+i),tr.time,s,tr.events)})}document.getElementById('traceSelect').onchange=render;render();</script>"
    path.write_text("<!doctype html><meta charset='utf-8'><title>装配诊断报告 V2</title><style>body{font:14px system-ui;margin:30px;color:#17202a;max-width:1500px}.cards{display:flex;gap:10px;flex-wrap:wrap}.card{padding:14px 20px;background:#eef3f8;border-radius:9px}.value{font-size:25px;font-weight:700}table{border-collapse:collapse;width:100%;margin:14px 0}th,td{border:1px solid #d7dee7;padding:7px;vertical-align:top}th{background:#edf2f7}small{display:block;color:#667}.chart{max-width:760px;background:#fafcff}.plots{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:12px}.plot{border:1px solid #d7dee7;padding:10px}.plot svg{width:100%}.warning{background:#fff5db;border-left:5px solid #f59e0b;padding:10px}canvas{width:100%;height:190px;border:1px solid #d7dee7}h1,h2{margin-top:28px}</style><h1>Play2Perfect 工业装配诊断 V2</h1><div class='cards'>" + "".join(f"<div class='card'><div>{label}</div><div class='value'>{value}</div></div>" for label, value in cards) + "</div><div class='warning'><b>解释边界：</b> task restart 在当前首次 episode 流程内不可观测；Screwing 缺少螺纹啮合真值，一次通过不强制分类；旧日志不能恢复未采集的峰值时间与机制时序。</div><h2>四场景比较</h2>" + _bar_chart(bars) + "<table><thead><tr><th>场景</th><th>N</th><th>任务成功</th><th>V1一次通过</th><th>首次插入</th><th>无恢复成功</th><th>一次通过 V2</th><th>旧检测器暂定一次通过</th><th>V1恢复率</th><th>明确恢复事件率</th><th>最大峰值/N</th></tr></thead><tbody>" + "".join(comparison) + "</tbody></table><h2>Rollout 分布</h2>" + distributions + "<h2>接触载荷</h2><table><thead><tr><th>场景</th><th>阶段</th><th>接触对象</th><th>最大/N</th><th>中位/N</th><th>P95/N</th><th>超限比例</th><th>最大连续接触事件冲量/N·s</th></tr></thead><tbody>" + contact_table + "</tbody></table><h2>执行效率统计</h2><table><thead><tr><th>指标</th><th>组别</th><th>N</th><th>Mean</th><th>P50</th><th>P90</th><th>P95</th><th>Std</th></tr></thead><tbody>" + efficiency + "</tbody></table><h2>典型轨迹（共享时间轴）</h2>" + trace_section + "<h2>数据质量</h2><ul>" + quality + "</ul><h2>Rollout 明细</h2><table><thead><tr><th>任务</th><th>Env</th><th>分类</th><th>尝试</th><th>重试</th><th>调整</th><th>恢复</th><th>完成/s</th><th>峰值/N</th><th>数据质量</th></tr></thead><tbody>" + rollout_table + "</tbody></table>" + script, encoding="utf-8")


def generate_report(root: str | Path | list[str | Path], output_dir: str | Path | None = None) -> dict[str, str]:
    roots = [Path(value) for value in root] if isinstance(root, list) else [Path(root)]
    output = Path(output_dir) if output_dir else roots[0]
    output.mkdir(parents=True, exist_ok=True)
    rows, paths = _load(roots), {}
    contact_rows = _contact_rows(rows)
    summary = _summary(rows, contact_rows)
    rollout_fields = ["problem", "env_id", "episode", "completed", "task_success", "assembly_success", "result_code", "first_pass", "first_pass_v1", "first_pass_reported_v1", "recovery_reported_v1", "first_pass_classification", "insertion_attempt_count", "insertion_retry_count", "drop_count", "regrasp_count", "regrasp_count_v1", "task_restart_count", "local_adjustment_count", "recovery_event_count", "task_recovery_count_v1", "recovery_reasons", "recovery_free_success", "task_completion_time_s", "failure_observed_duration_s", "grasp_acquisition_time_s", "transport_duration_s", "first_contact_time_s", "contact_to_success_duration_s", "insertion_duration_s", "recovery_duration_s", "insertion_backtrack_distance_m", "peak_normal_N", "high_force_time_s", "contact_safety_code", "conclusion_zh", "data_quality"]
    paths["rollouts_csv"], paths["contact_metrics_csv"] = output / "rollouts.csv", output / "contact_metrics.csv"
    paths["summary_json"], paths["report_html"] = output / "summary.json", output / "report.html"
    _write_csv(paths["rollouts_csv"], rows, rollout_fields)
    _write_csv(paths["contact_metrics_csv"], contact_rows, sorted({key for row in contact_rows for key in row}))
    paths["summary_json"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_html(paths["report_html"], rows, summary)
    return {key: str(value) for key, value in paths.items()}


def _self_test() -> None:
    def episode(problem="tight_insertion", success=True, attempts=1, recovery=0):
        return {"problem": problem, "env_id": 0, "episode": 0, "completed": True, "duration_s": 2.0,
                "task_success": success, "assembly_success": success, "attempt_count": attempts, "drop_count": 0,
                "regrasp_count": 0, "task_recovery_count": recovery, "local_adjustment_count": 3,
                "contact_metrics": {phase: {category: {} for category in CATEGORIES} for phase in PHASES}}
    with tempfile.TemporaryDirectory() as directory:
        root, metadata = Path(directory), {"schema_version": 3}
        normal = _normalize_episode(episode(), [], metadata, root)
        assert normal["first_pass_success"] is True and normal["local_adjustment_count"] == 3
        retry = _normalize_episode(episode(attempts=2, recovery=1), [], metadata, root)
        assert retry["first_pass_success"] is False and retry["result_code"] == "recovered_success"
        dropped = episode(recovery=1); dropped.update(drop_count=1, regrasp_count=1)
        assert _normalize_episode(dropped, [], metadata, root)["recovery_free_success"] is False
        screw_episode = episode(problem="screwing", attempts=8)
        screw_episode.update(insertion_attempt_count=None, insertion_retry_count=None)
        screw = _normalize_episode(screw_episode, [], metadata, root)
        assert screw["first_pass_success"] is None and screw["recovery_free_success"] is None
        assert "unverified" in screw["first_pass_classification"]
    print("assembly diagnostics reporter V2: ok")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="*")
    parser.add_argument("--output-dir")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test(); return 0
    if not args.run_dir:
        parser.error("run_dir is required unless --self-test is used")
    if len(args.run_dir) > 1 and not args.output_dir:
        parser.error("--output-dir is required when multiple run directories are supplied")
    source = args.run_dir if len(args.run_dir) > 1 else args.run_dir[0]
    print(json.dumps(generate_report(source, args.output_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
