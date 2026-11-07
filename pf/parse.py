"""Stage 1 - Parse.

Turns a folder of screenplay .txt files into a flat beat table.

This is the ONLY place timestamps enter the system. Every downstream stage
refers to beats by id; no model ever sees or emits a time.

Fails loudly: if the input does not look like the expected format, this
raises ParseError naming the file, the line number and the offending line,
rather than silently producing an empty or partial table.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, asdict, field

LABEL_RE = re.compile(r"^\*\*(.+?):\*\*$")
TIME_RE = re.compile(r"^Start:\s*\((\d+:\d+(?:\.\d+)?)\)\s*\|\s*End:\s*\((\d+:\d+(?:\.\d+)?)\)$")
# Field labels are matched case-insensitively on purpose: 1 of the 50 episodes
# in the reference pack (ep 6) writes them in uppercase. Tolerating case is also
# the cheapest thing that helps on an unseen pack.
SCENE_RE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
HEADING_RE = re.compile(r"^Scene heading:\s*(.+)$", re.I)
SSTART_RE = re.compile(r"^Scene start:\s*(\d+:\d+(?:\.\d+)?)\s*$", re.I)
SEND_RE = re.compile(r"^Scene end:\s*(\d+:\d+(?:\.\d+)?)\s*$", re.I)
EPNUM_RE = re.compile(r"^EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", re.I)
LOGLINE_RE = re.compile(r"^EPISODE LOGLINE:\s*(.*)$", re.I)
CONT_RE = re.compile(r"^CONTINUITY NOTES:\s*(.*)$", re.I)
START_MARK = "--- SCRIPT START ---"

# Clause splitting for sub-beat resolution on visual beats.
CLAUSE_SPLIT_RE = re.compile(
    r",(?=\s)|;|\s+(?:and then|before|while|as)\s+", re.I
)
MIN_CLAUSE_CHARS = 14

TYPE_MAP = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx", "SFX": "sfx"}


class ParseError(Exception):
    """Raised when the input does not match the expected screenplay format."""


def to_seconds(stamp: str) -> float:
    mm, ss = stamp.split(":")
    return int(mm) * 60 + float(ss)


def fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m}:{s:04.1f}"


@dataclass
class Beat:
    beat_id: str
    episode: int
    scene_no: int
    scene_heading: str
    t_start: float
    t_end: float
    type: str            # camera | dialogue | action | sfx
    speaker: str         # canonical
    speaker_raw: str
    modifiers: list = field(default_factory=list)
    offscreen: str = ""  # "" | O.S. | V.O.
    text: str = ""
    clauses: list = field(default_factory=list)  # [{id, t_start, t_end, text}]


def _split_label(label: str):
    """'LADY ALPHA (Cold, Venomous)' -> ('LADY ALPHA', ['Cold','Venomous'], '')."""
    mods, offscreen = [], ""
    for grp in re.findall(r"\(([^)]*)\)", label):
        for part in grp.split(","):
            p = part.strip()
            if not p:
                continue
            if p.upper().replace(" ", "") in {"O.S.", "OS", "OFFSCREEN", "OFF-SCREEN"}:
                offscreen = "O.S."
            elif p.upper().replace(" ", "") in {"V.O.", "VO", "VOICEOVER", "VOICE-OVER"}:
                offscreen = "V.O."
            else:
                mods.append(p)
    base = re.sub(r"\s*\(.*$", "", label).strip()
    return base, mods, offscreen


def build_alias_map(raw_speakers) -> dict:
    """Collapse speaker-label variants onto one canonical name.

    The reference pack used 67 distinct labels for roughly 40 characters
    (ALPHA / LADY ALPHA, BETA / LADY BETA, ...). Without this, standalone-
    clarity scoring treats one person as several strangers.

    Rule: strip honorifics and group by the remainder. Merge a group ONLY when
    its members differ by the presence of a title, never by a different title.

    LADY ALPHA / ALPHA  -> same person, merge.
    LADY GAMMA / LORD GAMMA -> a married couple sharing a surname. Two
    different titles means two different people, so the group is left alone.
    Without this guard the map silently deletes a character.

    Names here are placeholders: the input pack is confidential.
    """
    honorifics = r"^(LADY|LORD|SIR|MISS|MRS|MR|DAME|DUKE|DUCHESS|THE)\s+"
    groups = {}
    for name, count in raw_speakers.items():
        key, titles = name.upper().strip(), []
        while (m := re.match(honorifics, key)):
            titles.append(m.group(1))
            key = key[m.end():].strip()
        # "THE" is an article, not a gendered title, so it cannot distinguish people.
        titles = [t for t in titles if t != "THE"]
        groups.setdefault(key, []).append((count, name, tuple(titles)))

    alias = {}
    for key, members in groups.items():
        distinct_titles = {t for _, _, t in members if t}
        if len(distinct_titles) > 1:
            for _, name, _ in members:      # ambiguous: keep every name as itself
                alias[name] = name
            continue
        canonical = max(members)[1]          # highest count wins
        for _, name, _ in members:
            alias[name] = canonical
    return alias


def _clauses_for(beat_id: str, text: str, t0: float, t1: float):
    """Split a visual beat's description into clauses with proportional times.

    A 3.8s action line can hold three distinct visual ideas and an editor wants
    the second one. Clause times are ESTIMATED by character length - they are
    not in the source - so they are approximate by design.
    """
    parts = [p.strip() for p in CLAUSE_SPLIT_RE.split(text) if p and p.strip()]
    parts = [p for p in parts if len(p) >= MIN_CLAUSE_CHARS]
    if len(parts) < 2:
        return []
    total = sum(len(p) for p in parts)
    span = t1 - t0
    out, cursor = [], t0
    for i, p in enumerate(parts, start=1):
        width = span * (len(p) / total)
        end = t1 if i == len(parts) else round(cursor + width, 2)
        out.append({"id": f"{beat_id}#{i}", "t_start": round(cursor, 2),
                    "t_end": end, "text": p})
        cursor = end
    return out


def parse_episode(path: str, strict: bool = True):
    """Parse one screenplay file into (episode_meta, [Beat]). Raises ParseError."""
    raw = open(path, encoding="utf-8", errors="replace").read()
    lines = raw.split("\n")

    def die(lineno, msg, offending=""):
        raise ParseError(
            f"{os.path.basename(path)}:{lineno}: {msg}"
            + (f"\n    offending line: {offending!r}" if offending else "")
        )

    meta = {"episode": None, "logline": "", "continuity": "",
            "source_file": os.path.basename(path)}
    for i, line in enumerate(lines[:12], start=1):
        if (m := EPNUM_RE.match(line)):
            meta["episode"] = int(m.group(1))
        elif (m := LOGLINE_RE.match(line)):
            meta["logline"] = m.group(1).strip()
        elif (m := CONT_RE.match(line)):
            meta["continuity"] = m.group(1).strip()

    if meta["episode"] is None:
        if strict:
            die(1, "no 'EPISODE NUMBER:' header found in the first 12 lines")
        m = re.search(r"episode[-_ ]?0*(\d+)", os.path.basename(path), re.I)
        meta["episode"] = int(m.group(1)) if m else 0

    if strict and START_MARK not in raw:
        die(1, f"missing {START_MARK!r} marker")

    beats, scene_no, heading = [], 0, ""
    scenes, idx = [], 0
    scene_start = scene_end = None

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if (m := SCENE_RE.match(line)):
            scene_no = int(m.group(1))
            heading, scene_start, scene_end = "", None, None
        elif (m := HEADING_RE.match(line)):
            heading = m.group(1).strip()
        elif (m := SSTART_RE.match(line)):
            scene_start = to_seconds(m.group(1))
        elif (m := SEND_RE.match(line)):
            scene_end = to_seconds(m.group(1))
            if scene_start is not None:
                if scene_end < scene_start:
                    die(i + 1, f"scene {scene_no} ends ({scene_end}s) before it starts "
                               f"({scene_start}s)", line)
                scenes.append({"scene_no": scene_no, "heading": heading,
                               "t_start": scene_start, "t_end": scene_end})
        elif (m := LABEL_RE.match(line)):
            if i + 1 >= len(lines):
                die(i + 1, "beat label is the last line; no timestamp line follows", line)
            tm = TIME_RE.match(lines[i + 1].strip())
            if not tm:
                die(i + 2,
                    "beat label is not followed by a parsable "
                    "'Start: (M:SS.ss) | End: (M:SS.ss)' line",
                    lines[i + 1])
            t0, t1 = to_seconds(tm.group(1)), to_seconds(tm.group(2))
            if t1 < t0:
                die(i + 2, f"beat ends ({t1}s) before it starts ({t0}s)", lines[i + 1])
            body = lines[i + 2].strip() if i + 2 < len(lines) else ""
            if LABEL_RE.match(body) or TIME_RE.match(body):
                body = ""
            base, mods, offs = _split_label(m.group(1))
            btype = TYPE_MAP.get(base.upper(), "dialogue")
            idx += 1
            bid = f"e{meta['episode']:03d}b{idx:03d}"
            beats.append(Beat(
                beat_id=bid, episode=meta["episode"], scene_no=scene_no,
                scene_heading=heading, t_start=t0, t_end=t1, type=btype,
                speaker="" if btype != "dialogue" else base,
                speaker_raw="" if btype != "dialogue" else base,
                modifiers=mods, offscreen=offs, text=body,
                clauses=_clauses_for(bid, body, t0, t1) if btype in ("camera", "action") else [],
            ))
            i += 2
        i += 1

    if strict:
        if not scenes:
            die(1, "no complete scene block found "
                   "(need SCENE n / Scene heading / Scene start / Scene end)")
        if len(beats) < 20:
            die(1, f"only {len(beats)} beats extracted; expected at least 20. "
                   "The file format is probably not what this parser expects.")
        starts = [b.t_start for b in beats]
        regressions = sum(1 for a, b in zip(starts, starts[1:]) if b < a - 0.01)
        if regressions > 0.05 * len(beats):
            die(1, f"{regressions} of {len(beats)} beat starts go backwards "
                   "(>5%); beats are probably out of order")

    meta["scenes"] = scenes
    meta["duration"] = max((b.t_end for b in beats), default=0.0)
    meta["n_beats"] = len(beats)
    return meta, beats


def parse_pack(folder: str, strict: bool = True):
    """Parse every .txt in a folder. Returns (episodes, beats) sorted by episode."""
    paths = sorted(glob.glob(os.path.join(folder, "*.txt")))
    if not paths:
        raise ParseError(f"no .txt files found in {folder!r}")

    episodes, all_beats = [], []
    for p in paths:
        meta, beats = parse_episode(p, strict=strict)
        episodes.append(meta)
        all_beats.extend(beats)

    raw_speakers = {}
    for b in all_beats:
        if b.type == "dialogue" and b.speaker_raw:
            raw_speakers[b.speaker_raw] = raw_speakers.get(b.speaker_raw, 0) + 1
    alias = build_alias_map(raw_speakers)
    for b in all_beats:
        if b.type == "dialogue":
            b.speaker = alias.get(b.speaker_raw, b.speaker_raw)

    order = {m["source_file"]: m["episode"] for m in episodes}
    episodes.sort(key=lambda m: m["episode"])
    all_beats.sort(key=lambda b: (b.episode, b.t_start))
    return episodes, all_beats, alias


def write_beats(beats, path: str):
    with open(path, "w", encoding="utf-8") as fh:
        for b in beats:
            fh.write(json.dumps(asdict(b), ensure_ascii=False) + "\n")


def load_beats(path: str):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(Beat(**json.loads(line)))
    return out
