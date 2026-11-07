# Promo Finder

Finds 20–60 second promo clips in microdrama screenplays, ranks them, and says
where it picks badly.

## The one design decision everything follows from

**The model never sees or emits a timestamp.**

The screenplay is parsed into a table of beats, each with an id and a real
start and end read from the file. The model works only in beat ids: it tags
which beats are hooks and which are cliffhangers. Code turns those ids into
candidate windows, computes durations, enforces every hard constraint, and
maps ids back to times.

A hallucinated timestamp is therefore not unlikely, it is **structurally
impossible**. The measured count of invalid beat references is reported in
every run (`invalid_beat_references` in the funnel).

Two facts about this input drive the rest:

- **The timeline is layered.** 52% of beats start before the previous beat
  ends, because camera runs in parallel with dialogue. Duration is
  `max(end) − min(start)` over the window, never the sum of beat durations.
  Summing overstates by 2–3× and silently rejects every good candidate.
- **A beat is not the atomic unit of editing.** A 3.8s action line can hold
  three visual ideas and an editor wants the second one, so visual beats are
  split into clauses with proportional times, and code enumerates 0/1/2-beat
  lead-ins before the hook.

## Quickstart

No API key needed to see the whole thing run:

```bash
python run.py --input /path/to/screenplays --provider mock
```

With a real model — set one key and the provider auto-detects:

```bash
export GEMINI_API_KEY=...        # or GROQ_API_KEY / OPENROUTER_API_KEY
                                 # / ANTHROPIC_API_KEY / OPENAI_API_KEY
python run.py --input /path/to/screenplays
```

Useful flags: `--limit 5` (first five episodes), `--out some/dir`,
`--no-cluster-naming`, `--provider groq`.

Python 3.10+. No required third-party packages; `pyyaml` is used if present
and a built-in fallback parses the config otherwise.

## Pipeline

| # | Stage | Owner | Notes |
|---|-------|-------|-------|
| 1 | Parse | code | hard assertions, fails loudly on an unexpected format |
| 2 | Tag | LLM | one call per episode, beats shown **without** times |
| 3 | Assemble, gate, repair | code | enumerate windows, enforce the brief's constraints |
| 4 | Judge | LLM | one call per episode, scoring all of its candidates together |
| 5 | Cluster | code | beat / tag / speaker / location overlap |
| 6 | Name clusters | LLM | one call: name groups, merge semantic duplicates |
| 7 | Rank | code | two lists, ordered tie-breaks |
| 8 | Output | code | JSON, CSV, HTML, rejection log, blind A/B sheet |
| 9 | Approve | human | the editor accepts or rejects the shortlist |

The tag model and the judge model are configured **separately**, so the model
that scores a pick need not be the one that made it.

## Gates vs preferences

Hard gates come from the brief and are enforced by arithmetic:
duration 20–60s · the hook lands within 3s · nothing plays after the
cliffhanger · dialogue beats are never cut mid-line · every beat id exists.

Preferences are gradients, so a 44s promo is not treated like a 20s one and a
hook at 0.2s beats one at 2.9s even though both pass. A repaired candidate
carries a penalty, so **repair can never make a candidate outrank one that was
right the first time**.

Repair policy: *repair what is mechanically fixable, count what is diagnostic,
never repair a reference.* An over-long window is trimmed to a beat boundary;
a beat after the cliffhanger is truncated away. An invalid beat id is rejected
and counted, because there is no correct way to guess what it meant.

## Two ranked lists

Payoff strength and standalone clarity pull in opposite directions, so forcing
them into one ordering makes one lose silently.

- **List A, cold acquisition.** Standalone clarity is a *gate*, relaxed
  stepwise (4 → 3 → 2 → 1) until the list holds at least ten. The threshold
  used and the count admitted at each level are reported. **This list answers
  "which would you post first."** Order is round-robin across clusters, so a
  varied set goes out first; the single best hook can override to #1.
- **List B, retargeting.** Top 20% by hook among the clarity-gate failures,
  capped at five, each labelled for spoiler risk.

Episode position is **not** an input. Early episodes tend to top List A
because they need no context, so the funnel ordering falls out of the clarity
gate rather than being hardcoded — which keeps working on a new show whose
episode 40 happens to be self-contained, and needs no assumption about where
the paywall sits.

Within a cluster the order is ordered tie-breaks (hook → cliffhanger →
escalation → duration fit), not weights. There is no performance data yet, so
any weighting would be invented. Weights get learned from CTR once promos run.

## Evaluation

Three tiers, in increasing cost to the human.

**1. Code-only, free.** Written on every run: the gate funnel, invalid beat
references, and the rejection log broken down by which gate fired.

**2. Blind A/B, ~10 minutes.** Each run writes
`outputs/blind_ab_sheet.md`: each tool pick paired with a **random window from
the same episode that passes every hard gate**, shuffled and unlabelled.
Chance is 50% by construction, so the control is built in and anchoring is
impossible — you cannot ratify the tool's pick when you do not know which it
is. Fill in `outputs/blind_ab_answers.csv`, then:

```bash
python run.py --score-blind-ab
```

**3. Gold set, ~40 minutes.** Label promos before running anything, then:

```bash
python run.py --eval-gold gold_set.csv
```

Reports recall, precision@10, precision on ranks 11–20 as a control, mean and
max start-offset error, and rank correlation.

A match is **IoU ≥ 0.5 AND starts within 2 seconds**. IoU alone is far too
lenient here: episodes average ~123s and promos are 20–60s, so any two windows
in one episode overlap heavily by construction, and a random window matches
28% of the time. Under the strict rule a random window matches **5.1%**. That
baseline is reported next to precision, because the score means nothing
without it.

## Confidentiality

**The input pack must not be committed.** `.gitignore` excludes `data/`,
`outputs/`, `*.jsonl` and the cache. Point `--input` at a folder outside the
repo, or drop the pack into `data/` where git will ignore it.

## Config

Everything tunable is in `config.yaml`: provider and models, every gate
threshold, the gradient preferences, shortlist cap, clustering weights,
ranking thresholds and the evaluation match rule. Changing model or provider
is a config change, never a code change.

Responses are cached on disk by a hash of prompt and model, so re-ranking
never re-calls the model. Retries are capped at **1** — a retry storm is what
trips a free tier, so a failure is counted rather than hammered.

## Cost

Measured on a 50-episode pack (169,247 tokens of raw script):

| | tokens |
|---|---|
| Raw script (what a naive approach sends) | 169,247 |
| Beat list the tagger sees (no timestamps) | 87,484 |
| Trimmed camera prose | 76,547 |
| Judge pass input, 47 episodes with candidates, 4 per episode | ~150,000 |

The judge makes **one call per episode**, scoring all of that episode's
candidates together, which saves the per-call overhead and one copy of the
prompt template per candidate.

It does **not** save the beat text, and an earlier version of this section
claimed it did, quoting a saving of ~187,000 tokens against ~369,000. That
comparison assumed the episode's beats were sent once with candidates
referenced as id ranges. They are not: `render_candidate()` renders each
candidate's own span, so candidates whose windows overlap send the same beat
text more than once. The figure above is what the judge pass actually costs,
measured by rebuilding every episode's prompt: 3,190 input tokens on average,
5,230 on the largest episode.

That duplication is the binding constraint on a free tier, not the total
cost. A single request whose input plus `max_tokens` exceeds the provider's
per-request ceiling is refused outright rather than rate-limited, so pacing
cannot rescue it: at 8 candidates per episode one prompt reached 8,651
tokens against a ceiling of 8,000. `shortlist_per_episode` is set to keep
the worst episode inside it.

**Known future work:** send the episode's beat text once and reference each
candidate as a range of beat ids inside it. That removes the duplication, and
would make the original claim true.

## Known limits

- Timestamps are **shortlist-accurate, not frame-accurate**. The tool finds
  the moment; the editor trims the cut.
- Clause times inside a beat are *estimated* by proportional character length,
  since the source gives times per beat, not per clause.
- The script is not the screen: render quality, framing, line delivery, audio
  and comedy timing are invisible here and are exactly where a visual-led hook
  carries more risk than a dialogue-led one.
- Scoring is one model's judgment anchored to a written rubric, and the gold
  set is one editor's labels. Neither is a consensus standard.
