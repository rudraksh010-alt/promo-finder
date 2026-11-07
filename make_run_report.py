#!/usr/bin/env python3
"""Build outputs/RUN_REPORT.md from the artifacts a run already wrote.

This is a REPORTING script. It reads only; it never calls a model and never
imports the pipeline stages, so it cannot change what a run produced.

    python make_run_report.py --out outputs --cache .cache/groq --provider groq

Inputs it reads
    <out>/funnel.json      funnel counters + ranking diagnostics
    <out>/promos.json      List A / List B / clusters
    <out>/rejections.csv   one row per rejected or repaired candidate
    <out>/beats.jsonl      every beat, with its real text and type
    <cache>/*.json         cached model replies; the tag replies carry the
                           hook ids, which the run itself does not save
    <out>/run_state.json   the PREVIOUS run's snapshot, for section 7

Outputs it writes
    <out>/RUN_REPORT.md    the report
    <out>/run_state.json   this run's snapshot, so the next run can diff
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------- loading

def load_json(path, default=None):
    """Read a JSON file, or return `default` if it is missing."""
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_beats(path):
    """beats.jsonl -> {beat_id: beat dict}. Gives us real text and type."""
    beats = {}
    if not os.path.exists(path):
        return beats
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                beat = json.loads(line)
                beats[beat["beat_id"]] = beat
    return beats


def load_rejections(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_tagged_hooks(outdir):
    """Hook and cliffhanger ids THIS run tagged, from <out>/tags.json.

    This must come from the run's own output. An earlier version recovered
    them from the response cache, which is keyed by prompt hash and persists
    across runs: a 5-episode report then sampled hooks from all 50 episodes
    of a previous run, and four of five sampled ids were not in that run's
    beats.jsonl at all. The run now writes tags.json and this reads only that.
    """
    doc = load_json(os.path.join(outdir, "tags.json"))
    if not doc:
        return [], [], []
    hooks, cliffs = [], []
    for _ep, entry in sorted((doc.get("tags") or {}).items()):
        hooks.extend(entry.get("hooks") or [])
        cliffs.extend(entry.get("cliffhangers") or [])
    return hooks, cliffs, doc.get("episodes") or []


# ----------------------------------------------------------------- helpers

def fmt_time(seconds):
    """Seconds -> MM:SS.ss, the same shape the screenplay uses."""
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return "?"
    return "{:02d}:{:05.2f}".format(int(seconds // 60), seconds % 60)


def base_beat(beat_ref):
    """'e001b012#3' -> 'e001b012'. Clause ids share their beat's text."""
    return (beat_ref or "").split("#")[0]


def beat_line(beat_ref, beats, limit=190):
    """One readable line for a beat: its type, speaker and actual text."""
    beat = beats.get(base_beat(beat_ref))
    if not beat:
        return "`" + str(beat_ref) + "` - **beat not found in beats.jsonl**"
    text = " ".join((beat.get("text") or "").split())
    if len(text) > limit:
        text = text[:limit].rstrip() + "..."
    label = str(beat.get("type", "?"))
    who = beat.get("speaker") or ""
    if who:
        label += " / " + who
    return ("`{}` - **{}** - {}-{}\n  > {}".format(
        beat_ref, label, fmt_time(beat.get("t_start")),
        fmt_time(beat.get("t_end")), text or "(no text)"))


def estimate_tokens(beats, funnel, camera_words):
    """Rough input-token estimate, by characters / 4.

    This is an ESTIMATE, not billed usage: the response cache stores replies
    but not the token counts the provider reported. Labelled as an estimate in
    the report so nobody mistakes it for a measured figure.
    """
    tag_chars = 0
    for beat in beats.values():
        text = beat.get("text") or ""
        if beat.get("type") == "camera":
            text = " ".join(text.split()[:camera_words])
        tag_chars += len(text) + 24      # id/type/speaker scaffolding per line
    tag_tokens = tag_chars // 4
    judged = int(funnel.get("candidates_scored") or 0)
    episodes = int(funnel.get("episodes") or 0)
    judge_tokens = judged * 80 + episodes * 400
    return tag_tokens, judge_tokens, tag_tokens + judge_tokens


def flatten_config(cfg, prefix=""):
    """Nested config -> {'gates.max_duration': 60.0, ...} for easy diffing."""
    flat = {}
    for key, value in (cfg or {}).items():
        path = prefix + str(key)
        if isinstance(value, dict):
            flat.update(flatten_config(value, path + "."))
        else:
            flat[path] = value
    return flat


# ----------------------------------------------------------------- sections

def section_settings(provider, cfg, funnel, changed, tokens, label):
    llm = cfg.get("llm", {}) or {}
    tag_t, judge_t, total_t = tokens
    lines = ["## 1. Settings", ""]
    if label:
        lines += ["_" + label + "_", ""]
    lines += [
        "| | |",
        "|---|---|",
        "| provider | `{}` |".format(provider),
        "| tag model | `{}` |".format(llm.get("tag_model")),
        "| judge model | `{}` |".format(llm.get("judge_model")),
        "| cluster model | `{}` |".format(llm.get("cluster_model")),
        "| episodes | {} |".format(funnel.get("episodes", "?")),
        "| beats parsed | {} |".format(funnel.get("beats", "?")),
        "| shortlist per episode | {} |".format(cfg.get("shortlist_per_episode")),
        "| standalone gate (config) | {} |".format(
            (cfg.get("ranking") or {}).get("standalone_gate")),
        "| runtime | {} s |".format(funnel.get("runtime_seconds", "?")),
        "| est. input tokens | ~{:,} (tag ~{:,} + judge ~{:,}) |".format(
            total_t, tag_t, judge_t),
        "",
        "Token figures are **estimated** at characters / 4; the response "
        "cache keeps replies, not billed usage counts.",
        "",
        "**Runtime is rate-limit pacing, not model latency.** The free tier "
        "meters tokens per minute per model, so calls are deliberately spaced "
        "{}s apart; measured model inference is 1.7s per call. A 101-call "
        "pass therefore takes about an hour on this tier and would take "
        "minutes on a paid one. Do not read the runtime as model "
        "slowness.".format(
            (cfg.get("llm") or {}).get("min_call_interval_seconds", "?")),
        "",
    ]
    if changed is None:
        lines.append("**Config vs previous run:** no previous run recorded, "
                     "so this run is the baseline.")
    elif not changed:
        lines.append("**Config vs previous run:** no config values changed.")
    else:
        lines += ["**Config values changed since the last run:**", "",
                  "| setting | previous | now |", "|---|---|---|"]
        for key in sorted(changed):
            old, new = changed[key]
            lines.append("| `{}` | `{}` | `{}` |".format(key, old, new))
    lines.append("")
    return "\n".join(lines)


def section_funnel(funnel, diagnostics):
    order = ["episodes", "episodes_failed", "episodes_with_no_candidates",
             "beats", "hooks_tagged", "cliffhangers_tagged",
             "invalid_beat_references", "candidates_passing_gates",
             "candidates_shortlisted", "candidates_scored",
             "judge_dimension_missing", "list_a", "list_b",
             "rejections_logged"]
    lines = ["## 2. Funnel and gate admission", "", "| stage | count |",
             "|---|---:|"]
    for key in order:
        if key in funnel:
            value = funnel[key]
            shown = "{:,}".format(value) if isinstance(value, int) else str(value)
            lines.append("| {} | {} |".format(key.replace("_", " "), shown))

    invalid = funnel.get("invalid_beat_references")
    if invalid == 0:
        lines += ["", "**0 invalid beat references** across {} beats / {} "
                      "episodes - the no-timestamp design held.".format(
                          funnel.get("beats", "?"), funnel.get("episodes", "?"))]
    elif invalid:
        lines += ["", "**{} invalid beat references** - the model named beats "
                      "that do not exist. These were rejected, never "
                      "repaired.".format(invalid)]

    table = diagnostics.get("gate_admission_table") or {}
    used = diagnostics.get("standalone_gate_used")
    if table:
        lines += ["", "### Standalone-clarity gate admission", "",
                  "| gate threshold | candidates admitted |", "|---|---:|"]
        for level in sorted(table, key=lambda x: -int(x)):
            mark = "  <- used" if str(used) == str(level) else ""
            lines.append("| >= {} | {}{} |".format(level, table[level], mark))
    if used is not None:
        lines += ["", "Gate settled at **>= {}**, relaxed stepwise until List "
                      "A held {} candidates.".format(
                          used, diagnostics.get("list_a_size", "?"))]
    if diagnostics.get("rank_divergence") is not None:
        lines += ["", "Round-robin vs pure ranking: {} clusters in the top 10, "
                      "{} in the pure-ranked top 5; divergence {}.".format(
                          diagnostics.get("clusters_in_top10", "?"),
                          diagnostics.get("pure_rank_top5_clusters", "?"),
                          diagnostics.get("rank_divergence"))]
    lines.append("")
    return "\n".join(lines)


def section_scores(promos):
    """Judge score distribution. The point is whether the judge discriminates
    or saturates: a dimension where nearly every candidate scores the same
    carries no information and wastes a ranking tie-break."""
    rows = ((promos or {}).get("list_a") or []) + \
           ((promos or {}).get("list_b") or [])
    lines = ["## 3. Judge score distribution", ""]
    if not rows:
        return "\n".join(lines + ["No scored candidates.", ""])
    dims = ("hook", "cliffhanger", "standalone", "escalation")
    lines += ["Across the {} ranked candidates (List A + List B). These are "
              "the candidates that reached a list, not every candidate "
              "scored.".format(len(rows)), "",
              "| dimension | 5 | 4 | 3 | 2 | 1 | mean | share 4-5 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dim in dims:
        vals = []
        for r in rows:
            v = (r["candidate"].get("scores") or {}).get(dim)
            if isinstance(v, int):
                vals.append(v)
        if not vals:
            continue
        counts = collections.Counter(vals)
        mean = sum(vals) / len(vals)
        top = sum(1 for v in vals if v >= 4)
        lines.append("| {} | {} | {} | {} | {} | {} | {:.2f} | {:.0f}% |".format(
            dim, counts.get(5, 0), counts.get(4, 0), counts.get(3, 0),
            counts.get(2, 0), counts.get(1, 0), mean,
            100.0 * top / len(vals)))
    tuples = collections.Counter()
    for r in rows:
        sc = r["candidate"].get("scores") or {}
        tuples[tuple(sc.get(d) for d in dims)] += 1
    top10 = {tuple((r["candidate"].get("scores") or {}).get(d) for d in dims)
             for r in ((promos or {}).get("list_a") or [])[:10]}
    lines += ["", "Distinct score tuples: **{}** across {} candidates, but "
              "only **{}** among the top 10 of List A. Where the top of the "
              "list shares a tuple, its ordering comes from the duration-fit "
              "and hook-offset tie-breaks rather than from the "
              "judge.".format(len(tuples), len(rows), len(top10)), ""]
    return "\n".join(lines)


def section_rejections(rejections):
    lines = ["## 4. Rejections by gate", ""]
    if not rejections:
        return "\n".join(lines + ["No rejections logged.", ""])
    counts = collections.Counter(r["gate_fired"] for r in rejections)
    actions = collections.Counter(r["action"] for r in rejections)
    total = len(rejections)
    lines += ["{:,} rows logged - ".format(total)
              + ", ".join("{:,} {}".format(v, k)
                          for k, v in actions.most_common()) + ".",
              "", "| gate fired | count | share |", "|---|---:|---:|"]
    for gate, n in counts.most_common():
        lines.append("| `{}` | {:,} | {:.1f}% |".format(
            gate, n, 100.0 * n / total))
    lines.append("")
    return "\n".join(lines)


def section_hooks(hooks, beats, rng, n=5):
    lines = ["## 5. {} randomly sampled tagged hooks".format(n), ""]
    uniq = sorted(set(hooks))
    if not uniq:
        return "\n".join(lines + [
            "No tagged hooks found. This run wrote no `tags.json`.", ""])
    missing = [h for h in uniq if base_beat(h) not in beats]
    if missing:
        lines += ["**{} of {} tagged ids are not in this run's beats.jsonl "
                  "({}). The tag source is not scoped to this run.**".format(
                      len(missing), len(uniq), ", ".join(missing[:4])), ""]
    lines += ["Sampled from {:,} distinct hook beats the tagger named. This "
              "is the check on whether it picks real events or coverage "
              "shots.".format(len(uniq)), ""]
    for beat_ref in rng.sample(uniq, min(n, len(uniq))):
        lines.append("- " + beat_line(beat_ref, beats))
    kinds = collections.Counter(
        (beats.get(base_beat(h), {}).get("type") or "unknown") for h in uniq)
    lines += ["", "Type mix across all tagged hooks: "
              + ", ".join("**{} {}**".format(v, k)
                          for k, v in kinds.most_common()) + ".", ""]
    return "\n".join(lines)


def section_top10(promos, n=10):
    lines = ["## 6. Top {} of List A (cold acquisition)".format(n), ""]
    list_a = (promos or {}).get("list_a") or []
    if not list_a:
        return "\n".join(lines + ["List A is empty.", ""])
    lines += ["| # | ep | timespan | len | promo type | cluster | H/C/S/E |",
              "|---:|---:|---|---:|---|---|---|"]
    rows = list_a[:n]
    for row in rows:
        cand = row["candidate"]
        segs = cand.get("segments") or []
        if segs:
            span = "{}-{}".format(
                fmt_time(min(s["t_start"] for s in segs)),
                fmt_time(max(s["t_end"] for s in segs)))
            if len(segs) > 1:
                span += " ({} segs)".format(len(segs))
        else:
            span = "?"
        sc = cand.get("scores") or {}
        lines.append("| {} | {} | {} | {:.0f}s | {} | {} | {}/{}/{}/{} |".format(
            row.get("rank"), cand.get("episode"), span,
            float(cand.get("duration") or 0), cand.get("promo_type", ""),
            row.get("cluster_name") or "-",
            sc.get("hook", "-"), sc.get("cliffhanger", "-"),
            sc.get("standalone", "-"), sc.get("escalation", "-")))
    lines += ["", "Scores are hook / cliffhanger / standalone / escalation, "
              "each 1-5. The judge's own why-line for each:", ""]
    for row in rows:
        cand = row["candidate"]
        why = " ".join((cand.get("why") or "no why line returned").split())
        lines.append("- **#{} ep {}** - {}".format(
            row.get("rank"), cand.get("episode"), why))
    lines.append("")
    return "\n".join(lines)


def section_rejected_examples(rejections, beats, rng, n=3):
    lines = ["## 7. {} rejected candidates, quoted".format(n), ""]
    if not rejections:
        return "\n".join(lines + ["No rejections logged.", ""])
    # One example per distinct gate where possible, so the sample is informative
    # rather than three rows of the single most common failure.
    by_gate = collections.defaultdict(list)
    for row in rejections:
        by_gate[row["gate_fired"]].append(row)
    gates = sorted(by_gate, key=lambda g: -len(by_gate[g]))
    picked = [rng.choice(by_gate[g]) for g in gates[:n]]
    while len(picked) < n and len(rejections) > len(picked):
        extra = rng.choice(rejections)
        if extra not in picked:
            picked.append(extra)

    for i, row in enumerate(picked, 1):
        first, _, last = (row.get("beat_range") or "").partition("->")
        repairs = row.get("repair_attempts")
        extra = (", {} repair attempt(s)".format(repairs)
                 if repairs not in ("", "0", None) else "")
        lines += [
            "**{}. ep {} - killed by `{}`** - measured {}, limit {}, action "
            "`{}`{}".format(i, row.get("episode"), row.get("gate_fired"),
                            row.get("measured_value"), row.get("limit"),
                            row.get("action"), extra),
            "",
            "  First beat: " + beat_line(first.strip(), beats),
            "",
            "  Last beat: " + beat_line(last.strip(), beats),
            "",
        ]
    return "\n".join(lines)


def section_changed(prev, funnel, diagnostics):
    lines = ["## 8. What changed vs the previous run", ""]
    if not prev:
        return "\n".join(lines + [
            "No previous run recorded. This run is the baseline.", ""])
    old_funnel = prev.get("funnel") or {}
    watch = ["episodes", "hooks_tagged", "cliffhangers_tagged",
             "invalid_beat_references", "candidates_passing_gates",
             "candidates_shortlisted", "candidates_scored", "list_a", "list_b",
             "rejections_logged", "runtime_seconds"]
    rows = []
    for key in watch:
        old, new = old_funnel.get(key), funnel.get(key)
        if old != new and (key in funnel or key in old_funnel):
            delta = ""
            if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                diff = new - old
                delta = ("{:+,.1f}".format(diff) if isinstance(diff, float)
                         else "{:+,}".format(diff))
            rows.append((key, old, new, delta))
    old_used = (prev.get("diagnostics") or {}).get("standalone_gate_used")
    new_used = diagnostics.get("standalone_gate_used")
    if old_used != new_used:
        rows.append(("standalone_gate_used", old_used, new_used, ""))
    old_prov = prev.get("provider")
    if not rows:
        return "\n".join(lines + [
            "Every funnel counter is identical to the previous run.", ""])
    lines += ["| metric | previous | now | change |", "|---|---:|---:|---:|"]
    for key, old, new, delta in rows:
        lines.append("| {} | {} | {} | {} |".format(
            key.replace("_", " "), old, new, delta))
    if old_prov:
        lines += ["", "Previous run used provider `{}`.".format(old_prov)]
    lines.append("")
    return "\n".join(lines)


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Build RUN_REPORT.md from a run's output files.")
    ap.add_argument("--out", default="outputs", help="run output directory")
    # Tagged hook ids come from <out>/tags.json, which the run writes. They
    # are deliberately NOT read from the response cache: it persists across
    # runs and would mix episodes into the sample.
    ap.add_argument("--provider", default="(not recorded)",
                    help="provider this run used, for the settings table")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--label", default="", help="one-line note for this run")
    ap.add_argument("--seed", type=int, default=None,
                    help="sampling seed (default: derived from the run)")
    args = ap.parse_args()

    outdir = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)

    funnel_doc = load_json(os.path.join(outdir, "funnel.json"), {}) or {}
    funnel = funnel_doc.get("funnel") or {}
    diagnostics = funnel_doc.get("diagnostics") or {}
    if not funnel:
        sys.exit("no funnel.json in " + outdir + " - run the pipeline first")

    promos = load_json(os.path.join(outdir, "promos.json"), {}) or {}
    beats = load_beats(os.path.join(outdir, "beats.jsonl"))
    rejections = load_rejections(os.path.join(outdir, "rejections.csv"))

    hooks, _cliffs, tagged_eps = load_tagged_hooks(outdir)
    # Loud check: every tagged id must belong to an episode in THIS run.
    run_eps = {int(str(b.get("episode"))) for b in beats.values()}
    stray = sorted({h for h in hooks if base_beat(h) not in beats})
    if stray:
        print("WARNING: {} tagged ids are not in this run's beats.jsonl, "
              "e.g. {}".format(len(stray), ", ".join(stray[:5])), file=sys.stderr)
    if tagged_eps and run_eps and not set(map(int, tagged_eps)) <= run_eps:
        print("WARNING: tags.json covers episodes outside this run: {}".format(
            sorted(set(map(int, tagged_eps)) - run_eps)[:10]), file=sys.stderr)

    sys.path.insert(0, HERE)
    from run import load_config                   # reuse the real config loader
    cfg = load_config(args.config)

    prev = load_json(os.path.join(outdir, "run_state.json"))
    changed = None
    if prev:
        old_flat = prev.get("config_flat") or {}
        new_flat = flatten_config(cfg)
        changed = dict((k, (old_flat.get(k), new_flat[k])) for k in new_flat
                       if k in old_flat and old_flat[k] != new_flat[k])

    camera_words = int((cfg.get("llm") or {}).get("camera_words") or 14)
    tokens = estimate_tokens(beats, funnel, camera_words)

    seed = args.seed if args.seed is not None else (
        int(funnel.get("beats") or 0) * 1000 + int(funnel.get("episodes") or 0))
    rng = random.Random(seed)

    parts = [
        "# Run report - promo finder",
        "",
        "Provider `{}` - {} episodes - {}s runtime. Self-contained: every "
        "number below comes from this run's own output files.".format(
            args.provider, funnel.get("episodes", "?"),
            funnel.get("runtime_seconds", "?")),
        "",
        section_settings(args.provider, cfg, funnel, changed, tokens, args.label),
        section_funnel(funnel, diagnostics),
        section_scores(promos),
        section_rejections(rejections),
        section_hooks(hooks, beats, rng),
        section_top10(promos),
        section_rejected_examples(rejections, beats, rng),
        section_changed(prev, funnel, diagnostics),
    ]
    report = "\n".join(parts).rstrip() + "\n"

    report_path = os.path.join(outdir, "RUN_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    # Snapshot for the next run's section 7. Written last, so a crash above
    # leaves the previous snapshot intact.
    with open(os.path.join(outdir, "run_state.json"), "w", encoding="utf-8") as fh:
        json.dump({"provider": args.provider, "funnel": funnel,
                   "diagnostics": diagnostics,
                   "config_flat": flatten_config(cfg)}, fh, indent=2)

    print("wrote {}  ({} lines)".format(report_path,
                                        len(report.splitlines())))


if __name__ == "__main__":
    main()
