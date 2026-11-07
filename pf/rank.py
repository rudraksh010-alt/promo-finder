"""Stage 9 - Rank.

Two lists, because payoff strength and standalone clarity are in tension and
forcing them into one ordering makes one of them lose silently.

  List A - cold acquisition. Standalone clarity is a GATE, not a score.
           This list is the answer to "which would you post first".
  List B - retargeting. The clarity gate is relaxed; raw payoff wins. This is
           where a late-series twist belongs, labelled for spoiler risk.

Episode position is NOT an input. Early episodes tend to top List A because
they need no context, so the funnel ordering falls out of the clarity gate
instead of being hardcoded - which also means it still works on a new show
whose episode 40 happens to be self-contained, and it needs no assumption
about where the paywall sits.

Within a cluster the order is ordered tie-breaks, not invented weights:
there is no performance data yet, so any weighting would be made up. Weights
get learned from CTR once promos are live.
"""

from __future__ import annotations

from .assemble import preference_scores


def _sort_key(cand, cfg, which="a"):
    """Ordered tie-breaks. R6 OVERRIDE: cliffhanger is no longer an input.

    R6 locked hook -> cliffhanger -> escalation -> duration fit. Measured on
    the 50-episode run, cliffhanger scored 5 for 51 of 66 candidates (77%,
    mean 4.45), so it was spending a tie-break slot on a near-constant.

    The two lists differ because standalone is informative in exactly one of
    them. List A gates on standalone, so every candidate in it already scored
    at or above the threshold and the dimension is near-constant there. List B
    holds the candidates that FAILED that gate, where standalone varies and
    separates them. So:

        List A:  hook -> escalation -> duration_fit
        List B:  hook -> standalone -> escalation -> duration_fit
    """
    p = preference_scores(cand, cfg)
    s = cand.scores or {}
    penalty = p["repair_penalty"]
    head = [-(s.get("hook", 0) - penalty)]
    if which == "b":
        head.append(-s.get("standalone", 0))
    return tuple(head + [
        -s.get("escalation", 0),
        -p["duration_fit"],
        -p["hook_offset_bonus"],
        cand.episode,
        cand.candidate_id,
    ])


def spoiler_risk(cand, ctx_by_id) -> str:
    """A label, never a ranking input. Guessing a spoiler score would be
    inventing precision; labelling it is honest and still useful."""
    vals = [ctx_by_id.get(i) for i in (cand.hook_id, cand.cliff_id)
            if ctx_by_id.get(i)]
    if not vals:
        return "unknown"
    worst = max(vals)
    return "low" if worst <= 2 else ("medium" if worst <= 3 else "high")


def apply_gate(candidates, cfg):
    """Relax the clarity gate stepwise until List A is usable. Returns the
    threshold actually used and the count admitted at every level, which is a
    free line for the writeup."""
    r = cfg["ranking"]
    floor = int(r.get("standalone_gate_floor", 1))
    table = {}
    chosen, passing = r["standalone_gate"], []
    for t in range(r["standalone_gate"], floor - 1, -1):
        admitted = [c for c in candidates if (c.scores or {}).get("standalone", 0) >= t]
        table[t] = len(admitted)
        if len(admitted) >= r["min_list_a"] and not passing:
            chosen, passing = t, admitted
    if not passing:
        # min_list_a was never reached. Stop at the floor and report a short
        # List A rather than relaxing to 1, which would disable the gate and
        # admit promos the judge called incomprehensible to a cold viewer.
        chosen = floor
        passing = [c for c in candidates
                   if (c.scores or {}).get("standalone", 0) >= floor]
    return chosen, passing, table


def round_robin(candidates, cfg):
    """Order by taking the best of each cluster in turn.

    Cluster order is set by the strongest hook inside it. The single best hook
    overall can override to #1, so round-robin cannot bury the best promo in
    the pack behind a weaker cluster.
    """
    by_cluster = {}
    for c in candidates:
        by_cluster.setdefault(c.cluster_id or "unclustered", []).append(c)
    for members in by_cluster.values():
        members.sort(key=lambda c: _sort_key(c, cfg))

    order = sorted(by_cluster,
                   key=lambda cid: _sort_key(by_cluster[cid][0], cfg))
    out, i = [], 0
    while True:
        added = False
        for cid in order:
            if i < len(by_cluster[cid]):
                out.append(by_cluster[cid][i])
                added = True
        if not added:
            break
        i += 1

    if cfg["ranking"].get("top_hook_override") and candidates:
        best = min(candidates, key=lambda c: _sort_key(c, cfg))
        if out and out[0] is not best:
            out.remove(best)
            out.insert(0, best)
    return out


def rank(candidates, clusters, ctx_by_id, cfg):
    """Returns (list_a, list_b, diagnostics)."""
    gate, passing, table = apply_gate(candidates, cfg)
    failing = [c for c in candidates if c not in passing]

    list_a = round_robin(passing, cfg)
    pure = sorted(passing, key=lambda c: _sort_key(c, cfg))
    pure_pos = {c.candidate_id: i + 1 for i, c in enumerate(pure)}

    r = cfg["ranking"]
    failing.sort(key=lambda c: _sort_key(c, cfg, which="b"))
    keep = max(1, int(len(failing) * r["list_b_top_fraction"])) if failing else 0
    list_b = failing[:min(keep, r["list_b_max"])]

    rows_a = []
    for i, c in enumerate(list_a, start=1):
        rows_a.append({
            "rank": i,
            "pure_rank": pure_pos.get(c.candidate_id),
            "cluster": c.cluster_id,
            "cluster_name": (clusters.get(c.cluster_id) or {}).get("name", ""),
            "spoiler_risk": spoiler_risk(c, ctx_by_id),
            "candidate": c,
        })
    rows_b = [{"rank": i, "spoiler_risk": spoiler_risk(c, ctx_by_id),
               "cluster": c.cluster_id,
               "cluster_name": (clusters.get(c.cluster_id) or {}).get("name", ""),
               "candidate": c}
              for i, c in enumerate(list_b, start=1)]

    top10 = [r_["candidate"].cluster_id for r_ in rows_a[:10]]
    diagnostics = {
        "standalone_gate_used": gate,
        "gate_admission_table": table,
        "list_a_size": len(rows_a),
        "list_b_size": len(rows_b),
        "clusters_in_top10": len(set(top10)),
        "pure_rank_top5_clusters": len({c.cluster_id for c in pure[:5]}),
        "rank_divergence": sum(
            1 for r_ in rows_a[:10]
            if r_["pure_rank"] and abs(r_["pure_rank"] - r_["rank"]) >= 3),
    }
    return rows_a, rows_b, diagnostics
