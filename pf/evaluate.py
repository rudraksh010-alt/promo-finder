"""Evaluation.

Three things, in increasing cost to the human:

  1. Code-only, free: the gate funnel, invalid beat references, and tag
     agreement across repeated runs.
  2. Blind A/B, ~10 minutes: each tool pick is paired with a RANDOM window
     from the same episode that passes every hard gate, shuffled and shown
     unlabelled. Chance is 50% by construction, so the control is built in
     and the result cannot be contaminated by anchoring - you cannot ratify
     the tool's pick when you do not know which one it is.
  3. Gold set, ~40 minutes: precision, recall, start offset, rank correlation.

Match rule: IoU >= 0.5 AND starts within 2s. IoU alone is far too lenient on
~123-second episodes - a random window matches 28% of the time, against 5.1%
under the strict rule. Report the baseline beside the score or the number
means nothing.
"""

from __future__ import annotations

import csv
import json
import os
import random

from .assemble import span_duration, beats_in_span
from .parse import fmt_time, to_seconds


# ---------------------------------------------------------------- matching

def iou(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def is_start_only(gold) -> bool:
    """A hook-point label: a start with no end. Evaluation tier 3."""
    return gold.get("end_s") is None


def is_match(gold, tool, cfg) -> bool:
    e = cfg["evaluation"]
    if int(gold["episode"]) != int(tool["episode"]):
        return False
    # Start-only rule: a hook point marks WHERE to open, not a whole promo,
    # so there is no span to intersect and IoU cannot apply. Same episode and
    # the start within the tolerance is the whole test. It has its own random
    # baseline, because it is a looser rule than the span match.
    if is_start_only(gold):
        return abs(gold["start_s"] - tool["start_s"]) <= e["match_start_within"]
    return (iou(gold["start_s"], gold["end_s"], tool["start_s"], tool["end_s"]) >= e["match_iou"]
            and abs(gold["start_s"] - tool["start_s"]) <= e["match_start_within"])


# ---------------------------------------------------------------- gold set

def load_gold(path):
    """Accepts the workbook's columns. Times may be m:ss.s or raw seconds."""
    def parse_t(v):
        v = (v or "").strip()
        if not v:
            return None
        return to_seconds(v) if ":" in v else float(v)

    rows = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for raw in csv.DictReader(fh):
            low = {(k or "").strip().lower(): (v or "").strip()
                   for k, v in raw.items()}
            ep = low.get("episode") or low.get("ep")
            if not ep or not ep.strip().isdigit():
                continue
            s = parse_t(low.get("start_s") or low.get("start (m:ss.s)") or low.get("start"))
            e = parse_t(low.get("end_s") or low.get("end (m:ss.s)") or low.get("end"))
            # An end time is optional. A row with a start and no end is a
            # hook-point label (tier 3): it says where to open, not where to
            # stop. Requiring an end used to drop every such row in silence,
            # so a hook-point sheet scored as an empty gold set.
            if s is None:
                continue
            rank = low.get("my rank") or low.get("my_rank") or ""
            rows.append({"episode": int(ep), "start_s": s, "end_s": e,
                         "my_rank": float(rank) if rank.replace(".", "").isdigit() else None,
                         "hook_line": low.get("hook line (the beat you open on)")
                         or low.get("hook_line", "")})
    return rows


def _spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return round(num / (dx * dy), 3) if dx and dy else None


def score_against_gold(gold, tool_rows, cfg):
    """tool_rows: [{episode,start_s,end_s,rank}] in ranked order."""
    e = cfg["evaluation"]
    matched_gold, offsets, pairs = set(), [], []
    matched_tool = set()

    # Which rule applies, and therefore which random baseline the score must
    # be read against. A mixed sheet is reported as mixed rather than scored
    # under one rule, because the two rules are not comparable.
    n_start_only = sum(1 for g in gold if is_start_only(g))
    if gold and n_start_only == len(gold):
        rule_kind = "start_only"
    elif n_start_only:
        rule_kind = "mixed"
    else:
        rule_kind = "span"
    baseline = (e.get("match_start_only_baseline", e["random_baseline"])
                if rule_kind == "start_only" else e["random_baseline"])

    for gi, g in enumerate(gold):
        for t in tool_rows:
            if is_match(g, t, cfg):
                matched_gold.add(gi)
                matched_tool.add(t["rank"])
                offsets.append(abs(g["start_s"] - t["start_s"]))
                if g.get("my_rank") is not None:
                    pairs.append((g["my_rank"], t["rank"]))
                break

    # Divide by the promos that actually exist, not a hardcoded 10. With a
    # List A of 7, dividing by 10 understated precision by 30% and reported
    # precision_ranks_11_20 as 0.0 "control" when no such ranks existed.
    n_top = min(10, len(tool_rows))
    n_bottom = max(0, min(20, len(tool_rows)) - 10)
    top10 = sum(1 for r in matched_tool if r <= 10)
    bottom = sum(1 for r in matched_tool if 11 <= r <= 20)
    p10 = top10 / n_top if n_top else 0.0
    out = {
        "gold_labelled": len(gold),
        "tool_promos": len(tool_rows),
        "gold_matched": len(matched_gold),
        "match_rule_kind": rule_kind,
        "match_rule": (
            f"same episode and start within {e['match_start_within']}s "
            f"(start-only, no IoU)" if rule_kind == "start_only" else
            f"IoU >= {e['match_iou']} and start within {e['match_start_within']}s"
            + (" for rows with an end; start-only rule for rows without"
               if rule_kind == "mixed" else "")),
        "mean_start_offset_s": round(sum(offsets) / len(offsets), 2) if offsets else None,
        "max_start_offset_s": round(max(offsets), 2) if offsets else None,
    }

    # The start-only rule reports start offsets and NOTHING ELSE. It has no
    # span to intersect, so it will match a hook point against any promo in
    # the episode: a start that is badly wrong can still land within
    # tolerance of a neighbouring promo. That makes precision, recall and any
    # lift over baseline misleading under this rule, while the offset it was
    # labelled to measure stays sound.
    if rule_kind == "start_only":
        out["reported"] = ("start offsets only: precision, recall and lift "
                           "are not meaningful under a rule with no span")
        return out

    out.update({
        "recall": round(len(matched_gold) / len(gold), 3) if gold else None,
        "precision_at_10": round(p10, 3),
        "precision_at_10_denominator": n_top,
        "precision_ranks_11_20": (round(bottom / n_bottom, 3) if n_bottom
                                  else "n/a (fewer than 11 promos)"),
        "random_baseline": baseline,
        "lift_over_random": round(p10 / baseline, 2) if baseline else None,
        "rank_correlation": _spearman([p[0] for p in pairs], [p[1] for p in pairs]),
    })
    return out


# ---------------------------------------------------------------- blind A/B

def random_valid_window(beats, cfg, rng):
    """A window that passes every hard gate, chosen at random. The control."""
    g = cfg["gates"]
    ordered = sorted(beats, key=lambda b: b.t_start)
    for _ in range(400):
        i = rng.randrange(len(ordered))
        t0 = ordered[i].t_start
        target = rng.uniform(g["min_duration"], g["max_duration"])
        ends = [b.t_end for b in ordered
                if g["min_duration"] <= b.t_end - t0 <= g["max_duration"]]
        if not ends:
            continue
        t1 = min(ends, key=lambda e: abs((e - t0) - target))
        d = span_duration(beats, t0, t1)
        if g["min_duration"] <= d <= g["max_duration"]:
            return round(t0, 2), round(t1, 2)
    return None


def render_clip(beats, t0, t1, limit=14):
    lines = []
    for b in beats_in_span(beats, t0, t1):
        if b.type == "sfx":
            continue
        who = f"{b.speaker}: " if b.type == "dialogue" and b.speaker else ""
        kind = "" if b.type == "dialogue" else "[visual] "
        lines.append(f"{kind}{who}{b.text}")
    if len(lines) > limit:
        lines = lines[:limit] + [f"... (+{len(lines) - limit} more beats)"]
    return "\n".join(lines)


def make_blind_ab(list_a, beats_by_ep, cfg, outdir, seed=11):
    """Write a blind comparison sheet plus a separate answer key."""
    rng = random.Random(seed)
    n = cfg["evaluation"]["blind_ab_pairs"]
    pairs, key = [], []

    for i, row in enumerate(list_a[:n], start=1):
        c = row["candidate"]
        beats = beats_by_ep[c.episode]
        ctrl = random_valid_window(beats, cfg, rng)
        if not ctrl:
            continue
        tool = (c.segments[0].t_start, c.segments[0].t_end)
        if rng.random() < 0.5:
            left, right, tool_side = tool, ctrl, "LEFT"
        else:
            left, right, tool_side = ctrl, tool, "RIGHT"
        pairs.append({
            "pair": i, "episode": c.episode,
            "option_left": render_clip(beats, *left),
            "option_right": render_clip(beats, *right),
            "left_span": f"{fmt_time(left[0])}-{fmt_time(left[1])}",
            "right_span": f"{fmt_time(right[0])}-{fmt_time(right[1])}",
            "your_pick (L/R)": "",
        })
        key.append({"pair": i, "episode": c.episode, "tool_side": tool_side,
                    "candidate_id": c.candidate_id})

    os.makedirs(outdir, exist_ok=True)
    sheet = os.path.join(outdir, "blind_ab_sheet.md")
    with open(sheet, "w", encoding="utf-8") as fh:
        fh.write("# Blind A/B\n\nFor each pair, write L or R for the one you would "
                 "post. One side is the tool's pick, the other is a random window "
                 "from the same episode that passes every hard gate. You are not "
                 "told which. Chance is 50%.\n\n")
        for p in pairs:
            fh.write(f"\n---\n\n## Pair {p['pair']} (episode {p['episode']})\n\n"
                     f"**L**\n\n```\n{p['option_left']}\n```\n\n"
                     f"**R**\n\n```\n{p['option_right']}\n```\n\n"
                     f"Your pick (L/R): ____\n")

    with open(os.path.join(outdir, "blind_ab_answers.csv"), "w",
              encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair", "your_pick"])
        w.writeheader()
        for p in pairs:
            w.writerow({"pair": p["pair"], "your_pick": ""})

    with open(os.path.join(outdir, "blind_ab_key.json"), "w", encoding="utf-8") as fh:
        json.dump(key, fh, indent=2)
    return sheet, len(pairs)


def score_blind_ab(answers_csv, key_json):
    with open(key_json, encoding="utf-8") as fh:
        key = {str(k["pair"]): k["tool_side"] for k in json.load(fh)}
    wins = total = 0
    with open(answers_csv, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            pick = (row.get("your_pick") or "").strip().upper()
            if pick not in ("L", "R"):
                continue
            total += 1
            side = "LEFT" if pick == "L" else "RIGHT"
            if key.get(str(row["pair"]).strip()) == side:
                wins += 1
    return {"pairs_judged": total, "tool_preferred": wins,
            "tool_preference_rate": round(wins / total, 3) if total else None,
            "chance": 0.5}
