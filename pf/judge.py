"""Stage 5 - Judge.

One batched call per episode. The episode's beat text is sent ONCE and the
candidates are referenced as id ranges inside it. Per-candidate calls would
cost 2-4x more tokens for the same information: measured on the reference
pack, per-candidate judging at 10 candidates/episode comes to ~369k tokens
against ~187k batched.

The judge model is configured separately from the tag model so the two can
differ. A judge that did not make the pick cannot prefer its own work.
"""

from __future__ import annotations

import os

from .llm import call
from .assemble import beats_in_span

PROMPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "prompts", "judge.txt")

DIMS = ("hook", "cliffhanger", "standalone", "escalation")


def render_candidate(cand, beats, label):
    """A candidate as the judge sees it: real text, no timestamps."""
    lines = [f"{label}: {cand.promo_type}, {len(cand.segments)} segment(s)"]
    for n, seg in enumerate(cand.segments, start=1):
        if len(cand.segments) > 1:
            lines.append(f"  -- segment {n} --")
        for b in beats_in_span(beats, seg.t_start, seg.t_end):
            if b.type == "sfx":
                continue
            who = f"{b.speaker}: " if b.type == "dialogue" and b.speaker else ""
            mods = f" ({', '.join(b.modifiers)})" if b.modifiers else ""
            kind = {"camera": "[visual]", "action": "[action]",
                    "dialogue": ""}.get(b.type, "")
            lines.append(f"  {kind}{mods} {who}{b.text}".rstrip())
    return "\n".join(lines)


def judge_episode(meta, beats, candidates, cfg, log):
    """Score one episode's shortlist. Returns (candidates, stats)."""
    if not candidates:
        return [], {"episode": meta["episode"], "judged": 0, "unscored": 0}

    with open(PROMPT, encoding="utf-8") as fh:
        template = fh.read()

    labels = {f"cand_{i}": c for i, c in enumerate(candidates, start=1)}
    rendered = "\n\n".join(render_candidate(c, beats, lbl)
                           for lbl, c in labels.items())
    prompt = template.format(episode=meta["episode"],
                             logline=meta.get("logline", ""),
                             candidates=rendered)
    raw = call(prompt, model=cfg["judge_model"], provider=cfg.get("provider"),
               temperature=cfg.get("temperature", 0.0))

    stats = {"episode": meta["episode"], "judged": 0, "unscored": 0,
             "unknown_ids": 0}
    for row in raw.get("scores") or []:
        cid = row.get("candidate_id")
        cand = labels.get(cid)
        if cand is None:
            stats["unknown_ids"] += 1
            log.append({"stage": "judge", "episode": meta["episode"],
                        "gate_fired": "candidate_id_not_found",
                        "measured_value": cid, "limit": "must be a listed cand_n",
                        "action": "reject", "repair_attempts": 0})
            continue
        # A dimension the model did not return is MISSING, not a 1. Clamping
        # it to 1 made a dropped `standalone` indistinguishable from a
        # genuine "incomprehensible to a cold viewer", and `standalone` is
        # the List A gate. So a candidate with any missing dimension is left
        # unscored, which keeps it out of both lists, and is counted.
        scores, missing = {}, []
        for d in DIMS:
            raw_v = row.get(d)
            if raw_v is None or str(raw_v).strip() == "":
                missing.append(d)
                continue
            try:
                scores[d] = max(1, min(5, int(raw_v)))
            except (TypeError, ValueError):
                missing.append(d)
        if missing:
            stats["missing_dims"] = stats.get("missing_dims", 0) + 1
            log.append({"stage": "judge", "episode": meta["episode"],
                        "beat_range": cand.candidate_id,
                        "gate_fired": "judge_dimension_missing",
                        "measured_value": ",".join(missing),
                        "limit": "all of " + ", ".join(DIMS),
                        "action": "reject", "repair_attempts": 0})
            continue
        cand.scores = scores
        cand.why = (row.get("why") or "").strip()
        try:
            cand.scores["rank_within_episode"] = int(row.get("rank_within_episode", 99))
        except (TypeError, ValueError):
            cand.scores["rank_within_episode"] = 99
        stats["judged"] += 1

    scored = []
    for c in candidates:
        if c.scores:
            scored.append(c)
        else:
            stats["unscored"] += 1
            log.append({"stage": "judge", "episode": meta["episode"],
                        "beat_range": c.candidate_id,
                        "gate_fired": "no_score_returned",
                        "measured_value": "", "limit": "judge must score every candidate",
                        "action": "reject", "repair_attempts": 0})
    return scored, stats
