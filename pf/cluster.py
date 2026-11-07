"""Stage 7+8 - Cluster, then name.

Code does the measurable part: beat overlap, tag overlap, shared speakers,
shared location. That is deterministic, free and exact.

One small LLM call then does only what code cannot: merge groups that are the
same moment type despite sharing nothing measurable, and name each group in
words an editor would use. A long series repeats its emotional beats - in the
reference pack the most common motif recurred in a quarter of the episodes -
so two promos built on the same motif from far-apart episodes share no beats,
no characters and no location, and code sees 0% similarity where an editor
sees a duplicate.

The clustering is also the diversity mechanism for ranking: List A is built
round-robin across clusters, which turns "post a varied set first" from an
assertion into a mechanism.
"""

from __future__ import annotations

import os

from .llm import call
from .assemble import beats_in_span

PROMPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "prompts", "cluster.txt")


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _features(cand, beats_by_ep):
    beats = beats_by_ep[cand.episode]
    speakers, locations = set(), set()
    for seg in cand.segments:
        for b in beats_in_span(beats, seg.t_start, seg.t_end):
            if b.type == "dialogue" and b.speaker:
                speakers.add(b.speaker)
            if b.scene_heading:
                locations.add(b.scene_heading.strip().upper())
    return {"beats": set(cand.beat_ids), "tags": set(cand.tags),
            "speakers": speakers, "locations": locations}


def similarity(fa, fb, cfg) -> float:
    c = cfg["clustering"]
    loc = 1.0 if fa["locations"] & fb["locations"] else 0.0
    return (c["beat_overlap_weight"] * _jaccard(fa["beats"], fb["beats"])
            + c["tag_overlap_weight"] * _jaccard(fa["tags"], fb["tags"])
            + c["speaker_overlap_weight"] * _jaccard(fa["speakers"], fb["speakers"])
            + c["location_weight"] * loc)


def cluster(candidates, beats_by_ep, cfg, log):
    """Single-link clustering by the code-computed similarity matrix."""
    if not candidates:
        return {}
    feats = [_features(c, beats_by_ep) for c in candidates]
    thresh = cfg["clustering"]["merge_threshold"]

    parent = list(range(len(candidates)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if similarity(feats[i], feats[j], cfg) >= thresh:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups = {}
    for i, cand in enumerate(candidates):
        groups.setdefault(find(i), []).append(cand)

    clusters = {}
    for n, (_, members) in enumerate(sorted(groups.items(),
                                            key=lambda kv: -len(kv[1])), start=1):
        cid = f"cluster_{n}"
        for m in members:
            m.cluster_id = cid
        clusters[cid] = {"cluster_id": cid, "members": members, "name": ""}
    return clusters


def name_clusters(clusters, beats_by_ep, cfg, log):
    """One call: name each group, and merge semantic duplicates code can't see."""
    if not clusters:
        return clusters
    if not cfg["clustering"].get("llm_naming", True):
        for cid, c in clusters.items():
            c["name"] = f"group {cid.split('_')[1]}"
        return clusters

    blocks = []
    for cid, c in clusters.items():
        rep = max(c["members"], key=lambda m: sum(
            v for k, v in (m.scores or {}).items() if k != "rank_within_episode"))
        beats = beats_in_span(beats_by_ep[rep.episode],
                              rep.segments[0].t_start, rep.segments[0].t_end)
        text = " / ".join(b.text[:70] for b in beats
                          if b.type in ("dialogue", "action"))[:420]
        blocks.append(
            f"{cid}: {len(c['members'])} promo(s), episodes "
            f"{sorted({m.episode for m in c['members']})}, "
            f"tags {sorted({t for m in c['members'] for t in m.tags})}\n"
            f"  representative: {text}")

    with open(PROMPT, encoding="utf-8") as fh:
        template = fh.read()
    raw = call(template.format(clusters="\n\n".join(blocks)),
               model=cfg["llm"]["cluster_model"],
               provider=cfg["llm"].get("provider"),
               temperature=cfg["llm"].get("temperature", 0.0))

    merges = {}
    for row in raw.get("clusters") or []:
        cid = row.get("cluster_id")
        if cid not in clusters:
            log.append({"stage": "cluster", "gate_fired": "cluster_id_not_found",
                        "measured_value": cid, "limit": "must be a listed cluster",
                        "action": "reject", "repair_attempts": 0})
            continue
        clusters[cid]["name"] = (row.get("name") or "").strip() or cid
        tgt = row.get("merge_with")
        if tgt and tgt in clusters and tgt != cid:
            merges[cid] = tgt

    for src, tgt in merges.items():
        # Follow merge chains, with a hard cap on the walk. A cycle that does
        # not contain `src` (a->b, b->c, c->d, d->b) would loop forever here,
        # and `merges` comes from the model, so an unseen pack can produce
        # one. Hanging is the worst failure shape for a live run.
        for _ in range(len(merges) + 1):
            if tgt not in merges or merges[tgt] == src:
                break
            tgt = merges[tgt]
        if src not in clusters or tgt not in clusters or src == tgt:
            continue
        for m in clusters[src]["members"]:
            m.cluster_id = tgt
        clusters[tgt]["members"].extend(clusters[src]["members"])
        del clusters[src]

    for cid, c in clusters.items():
        if not c["name"]:
            c["name"] = cid
    return clusters
