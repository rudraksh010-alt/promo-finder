"""Stage 3 - Assemble, gate and repair.

Code owns time. The tagger named beats; this module turns them into candidate
promos, enforces every hard constraint by arithmetic, and logs what it killed.

Two facts about this input shape drive the whole module:

  1. The timeline is LAYERED. 52% of beats in the reference pack start before
     the previous beat ends, because camera runs in parallel with dialogue.
     So a promo's duration is  max(end) - min(start)  over its beats, NEVER
     the sum of beat durations. Summing overstates by 2-3x and would silently
     reject every good candidate.

  2. A beat is not the atomic unit of editing. A 3.8s action line can hold
     three visual ideas and an editor wants the second one, so window starts
     may land on a clause (#n) and code enumerates 0/1/2-beat lead-ins.

Gates are HARD because they come from the brief (20-60s, hook within 3s).
Preferences are GRADIENT, so a 44s promo is not treated like a 20s one and a
hook at 0.2s beats one at 2.9s. A repaired candidate carries a penalty, so
repair can never make a candidate outrank one that was right first time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Segment:
    t_start: float
    t_end: float
    beat_ids: list = field(default_factory=list)


@dataclass
class Candidate:
    candidate_id: str
    episode: int
    segments: list                      # [Segment]
    hook_id: str
    cliff_id: str
    hook_offset: float                  # seconds from promo start to the hook
    duration: float
    beat_ids: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    lead_in_beats: int = 0
    repair_attempts: int = 0
    repairs: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    promo_type: str = ""
    cluster_id: str = ""
    why: str = ""

    def to_dict(self):
        d = asdict(self)
        d["segments"] = [asdict(s) if not isinstance(s, dict) else s
                         for s in self.segments]
        return d


def _resolve(bid, beat_by_id, clause_by_id):
    """A selectable id is either a beat or a clause of one."""
    if bid in clause_by_id:
        c, parent = clause_by_id[bid]
        return c["t_start"], c["t_end"], parent
    b = beat_by_id[bid]
    return b.t_start, b.t_end, b


def index_beats(beats):
    beat_by_id = {b.beat_id: b for b in beats}
    clause_by_id = {}
    for b in beats:
        for c in b.clauses:
            clause_by_id[c["id"]] = (c, b)
    return beat_by_id, clause_by_id


def beats_in_span(beats, t0, t1):
    """Every beat overlapping [t0, t1]. A time range carries what plays in it."""
    return [b for b in beats if b.t_end > t0 + 1e-6 and b.t_start < t1 - 1e-6]


def span_duration(beats, t0, t1):
    """Duration of the promo, which is the segment that ships: t1 - t0.

    This used to measure to the last beat's END inside the window, which is
    NOT what the editor is handed. Because t1 carries the cliffhanger tail
    pad, the two differed by exactly the pad: measured on run 02, 6 of 7
    promos reported a duration 2.00s below their own printed timespan. The
    20-60s gate was therefore validating a shorter window than the cut, so a
    candidate with 58.1s of content emitted a 60.1s promo and still passed.

    `beats` is kept in the signature so both callers keep one definition of
    duration; the span is fully determined by its own endpoints.
    """
    return t1 - t0


def classify(cand: Candidate, beats_by_ep) -> str:
    """Promo type is derived by code, not judged by a model."""
    segs = cand.segments
    eps = {cand.episode}
    if len(segs) == 1:
        beats = beats_in_span(beats_by_ep[cand.episode],
                              segs[0].t_start, segs[0].t_end)
        scenes = {b.scene_no for b in beats}
        return "single-scene hook clip" if len(scenes) <= 1 else "cross-scene hook clip"
    mean_len = sum(s.t_end - s.t_start for s in segs) / len(segs)
    if len(segs) >= 3 and mean_len < 4.0:
        return "montage"
    return "multi-segment hook clip"


def build_candidates(meta, beats, tagged, cfg, log):
    """Enumerate, gate and repair candidates for one episode."""
    g = cfg["gates"]
    min_d, max_d = g["min_duration"], g["max_duration"]
    hook_window = g["hook_within"]
    tail_pad = g["cliffhanger_tail_pad"]
    max_lead = g.get("max_lead_in_beats", 2)

    beat_by_id, clause_by_id = index_beats(beats)
    ordered = sorted(beats, key=lambda b: (b.t_start, b.t_end))
    pos = {b.beat_id: i for i, b in enumerate(ordered)}

    hooks = [h for h in tagged["hooks"] if h in beat_by_id or h in clause_by_id]
    cliffs = [c for c in tagged["cliffhangers"] if c in beat_by_id or c in clause_by_id]

    # Microdrama episodes end on a cliffhanger by construction, so the final
    # beat is always a candidate ending even when the tagger did not name it.
    # Cliffhangers are the scarce side of the hook x cliffhanger pairing
    # (measured: 152 tagged across 50 episodes, ~3 per episode), so the
    # scarcer side caps the whole pipeline.
    last_beat = max(beats, key=lambda b: b.t_end)
    if last_beat.beat_id not in cliffs:
        cliffs.append(last_beat.beat_id)

    out, seen, n = [], set(), 0
    for h in hooks:
        h_start, _, h_parent = _resolve(h, beat_by_id, clause_by_id)
        h_idx = pos[h_parent.beat_id]

        for lead in range(0, max_lead + 1):
            start_beat = ordered[max(0, h_idx - lead)]
            t0 = start_beat.t_start if lead else h_start
            hook_offset = round(h_start - t0, 2)
            if hook_offset > hook_window + 1e-6:
                continue                      # lead-in pushed the hook out of reach

            for c in cliffs:
                _, c_end, _ = _resolve(c, beat_by_id, clause_by_id)
                t1 = round(c_end + tail_pad, 2)
                if t1 <= t0:
                    continue

                duration = round(span_duration(beats, t0, t1), 2)
                repairs, attempts = [], 0

                # --- repair: over-long, trim back to a beat boundary under max.
                # The kept boundary must leave room for the tail pad, because
                # the emitted segment is t1 - t0 and t1 carries that pad.
                if duration > max_d and attempts < g["max_repair_attempts"]:
                    inside = sorted(beats_in_span(beats, t0, t1), key=lambda b: b.t_end)
                    fits = [b.t_end for b in inside
                            if (b.t_end + tail_pad) - t0 <= max_d]
                    if fits:
                        t1 = round(max(fits) + tail_pad, 2)
                        duration = round(span_duration(beats, t0, t1), 2)
                        attempts += 1
                        repairs.append("trimmed_to_beat_boundary")

                inside = beats_in_span(beats, t0, t1)

                # --- repair: nothing may play after the cliffhanger.
                # Clip the tail pad at the first beat that starts after the
                # cliffhanger ends, rather than rejecting. The pad exists so
                # the cliffhanger can land, but on a layered timeline (52% of
                # beats start before the previous ends) a new camera beat
                # almost always starts inside it, so the old code rejected
                # what R7 calls mechanically fixable. The old "repair" set t1
                # to c_end + tail_pad, the value it already held, so it
                # changed nothing and the candidate was rejected anyway.
                after = [b for b in inside if b.t_start > c_end + 1e-6]
                if after and attempts < g["max_repair_attempts"]:
                    first_after = min(b.t_start for b in after)
                    t1 = round(min(c_end + tail_pad, first_after), 2)
                    inside = beats_in_span(beats, t0, t1)
                    after = [b for b in inside if b.t_start > c_end + 1e-6]
                    duration = round(span_duration(beats, t0, t1), 2)
                    attempts += 1
                    repairs.append("clipped_tail_at_next_beat")
                if after:
                    log.append(_reject(meta, h, c, "beat_after_cliffhanger",
                                       len(after), "0 beats after", attempts))
                    continue

                # --- gates on duration, checked AFTER every repair that can
                # move t1, so a repaired candidate can never carry a stale
                # duration past the gate.
                if duration < min_d:
                    log.append(_reject(meta, h, c, "duration_too_short",
                                       duration, f">= {min_d}s", attempts))
                    continue
                if duration > max_d:
                    log.append(_reject(meta, h, c, "duration_too_long",
                                       duration, f"<= {max_d}s", attempts))
                    continue

                # --- gate: never cut a line of dialogue in half
                clipped = [b for b in inside
                           if b.type == "dialogue" and (b.t_start < t0 - 1e-6
                                                        or b.t_end > t1 + 1e-6)]
                if clipped:
                    log.append(_reject(meta, h, c, "dialogue_clipped_mid_line",
                                       clipped[0].beat_id, "whole beat or none",
                                       attempts))
                    continue

                key = (round(t0, 2), round(t1, 2))
                if key in seen:
                    continue
                seen.add(key)

                n += 1
                tags = sorted({t for b in [h, c] for t in tagged["tags"].get(b, [])})
                out.append(Candidate(
                    candidate_id=f"e{meta['episode']:03d}_cand_{n}",
                    episode=meta["episode"],
                    segments=[Segment(round(t0, 2), round(t1, 2),
                                      [b.beat_id for b in inside])],
                    hook_id=h, cliff_id=c, hook_offset=hook_offset,
                    duration=duration,
                    beat_ids=[b.beat_id for b in inside],
                    tags=tags, lead_in_beats=lead,
                    repair_attempts=attempts, repairs=repairs,
                ))

    for cand in out:
        cand.promo_type = classify(cand, {meta["episode"]: beats})
    return out


def _reject(meta, hook, cliff, gate, measured, limit, attempts):
    return {"stage": "assemble", "episode": meta["episode"],
            "beat_range": f"{hook}->{cliff}", "gate_fired": gate,
            "measured_value": measured, "limit": limit,
            "action": "reject", "repair_attempts": attempts}


def preference_scores(cand: Candidate, cfg) -> dict:
    """Gradient preferences. Gates are pass/fail; these are smooth."""
    p = cfg["preferences"]
    lo, hi = p["ideal_duration"]
    d = cand.duration
    if lo <= d <= hi:
        fit = 1.0
    elif d < lo:
        floor_ = cfg["gates"]["min_duration"]
        fit = p["duration_fit_floor"] + (1 - p["duration_fit_floor"]) * \
            max(0.0, (d - floor_) / max(1e-6, lo - floor_))
    else:
        ceil_ = cfg["gates"]["max_duration"]
        fit = p["duration_fit_floor"] + (1 - p["duration_fit_floor"]) * \
            max(0.0, (ceil_ - d) / max(1e-6, ceil_ - hi))
    hook_bonus = max(0.0, 1.0 - cand.hook_offset / cfg["gates"]["hook_within"])
    penalty = p["repair_penalty"] * cand.repair_attempts
    return {"duration_fit": round(fit, 3),
            "hook_offset_bonus": round(hook_bonus, 3),
            "repair_penalty": round(penalty, 3)}


def shortlist(cands, cfg):
    """Cap candidates per episode before the judge pass, cheapest-first.

    Dedupes lead-in variants of the same promo: three variants of one window
    must not occupy three shortlist slots.
    """
    cap = cfg["shortlist_per_episode"]
    ranked = sorted(
        cands,
        key=lambda c: (-preference_scores(c, cfg)["duration_fit"],
                       -preference_scores(c, cfg)["hook_offset_bonus"],
                       c.repair_attempts, c.candidate_id),
    )
    kept, used = [], []
    thresh = cfg["preferences"]["dedupe_overlap"]
    for c in ranked:
        s = c.segments[0]
        dup = False
        for k in used:
            inter = max(0.0, min(s.t_end, k[1]) - max(s.t_start, k[0]))
            union = max(s.t_end, k[1]) - min(s.t_start, k[0])
            if union > 0 and inter / union >= thresh:
                dup = True
                break
        if dup:
            continue
        kept.append(c)
        used.append((s.t_start, s.t_end))
        if len(kept) >= cap:
            break
    return kept
