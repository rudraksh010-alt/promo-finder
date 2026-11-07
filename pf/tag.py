"""Stage 2 - Tag.

One call per episode. The model sees beats WITHOUT timestamps (they are not
its business, and stripping them removes ~half the tokens) and returns beat
ids only.

Any id the model returns that is not in the beat table is a hallucinated
reference. It is REJECTED, never repaired: there is no correct way to guess
which beat `e001b999` meant, and fuzzy-matching it would invent data. The
count of rejected references is the project's hallucination metric, so
repairing them would destroy the number that proves the design works.
"""

from __future__ import annotations

import os

from .llm import call

PROMPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "prompts", "tag.txt")

TYPE_ABBR = {"camera": "CAM", "dialogue": "DLG", "action": "ACT", "sfx": "SFX"}
VALID_TAGS = {"humiliation", "reversal", "betrayal", "accident", "reveal",
              "threat", "violence", "confession", "quotable", "emotional_peak",
              "separation", "rescue"}


def render_beats(beats, camera_words: int = 14, drop_sfx: bool = True) -> str:
    """Beat list as the tagger sees it: ids, no times, camera prose trimmed.

    Measured on the reference pack: the raw script is 169,247 tokens, this
    representation is ~87,000, and trimming camera prose brings it to ~77,000.
    """
    lines = []
    for b in beats:
        if drop_sfx and b.type == "sfx":
            continue
        kind = TYPE_ABBR[b.type]
        who = f" {b.speaker}" if b.type == "dialogue" and b.speaker else ""
        mods = f" ({', '.join(b.modifiers)})" if b.modifiers else ""
        if b.clauses:
            parts = " | ".join(f"{c['text']} #{i}"
                               for i, c in enumerate(b.clauses, start=1))
            lines.append(f"[{b.beat_id}] {kind}{who}{mods}: {parts}")
        else:
            text = b.text
            if b.type == "camera":
                words = text.split()
                if len(words) > camera_words:
                    text = " ".join(words[:camera_words]) + " ..."
            lines.append(f"[{b.beat_id}] {kind}{who}{mods}: {text}")
    return "\n".join(lines)


def _valid_id_set(beats) -> set:
    ids = set()
    for b in beats:
        ids.add(b.beat_id)
        for c in b.clauses:
            ids.add(c["id"])
    return ids


def tag_episode(meta, beats, cfg, log):
    """Tag one episode. Returns (tags_dict, stats)."""
    with open(PROMPT, encoding="utf-8") as fh:
        template = fh.read()
    prompt = template.format(
        episode=meta["episode"],
        logline=meta.get("logline", ""),
        beats=render_beats(beats, camera_words=cfg.get("camera_words", 14)),
    )
    raw = call(prompt,
               model=cfg["tag_model"],
               provider=cfg.get("provider"),
               temperature=cfg.get("temperature", 0.0))

    valid = _valid_id_set(beats)
    stats = {"episode": meta["episode"], "returned": 0, "invalid_refs": 0,
             "hooks": 0, "cliffhangers": 0, "bad_tags": 0}

    def keep(ids):
        out = []
        for i in ids or []:
            stats["returned"] += 1
            if i in valid:
                out.append(i)
            else:
                stats["invalid_refs"] += 1
                log.append({"stage": "tag", "episode": meta["episode"],
                            "gate_fired": "beat_id_not_found",
                            "measured_value": i, "limit": "must exist in beat table",
                            "action": "reject", "repair_attempts": 0})
        return out

    hooks = keep(raw.get("hooks"))
    cliffs = keep(raw.get("cliffhangers"))

    tags = {}
    for bid, tlist in (raw.get("tags") or {}).items():
        if bid not in valid:
            continue
        clean = [t for t in (tlist or []) if t in VALID_TAGS]
        stats["bad_tags"] += len(tlist or []) - len(clean)
        if clean:
            tags[bid] = clean[:3]

    ctx = {}
    for bid, v in (raw.get("context_dependency") or {}).items():
        if bid in valid:
            try:
                ctx[bid] = max(1, min(5, int(v)))
            except (TypeError, ValueError):
                pass

    stats["hooks"], stats["cliffhangers"] = len(hooks), len(cliffs)
    return {"episode": meta["episode"], "hooks": hooks, "cliffhangers": cliffs,
            "tags": tags, "context_dependency": ctx}, stats
