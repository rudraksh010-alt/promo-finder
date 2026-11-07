#!/usr/bin/env python3
"""Promo finder - orchestrator.

    python run.py --input <folder-of-screenplays>
    python run.py --input <folder> --provider mock      # no API key needed
    python run.py --eval-gold gold_set.csv              # score against labels
    python run.py --score-blind-ab                      # score the blind A/B

Pipeline:
    parse (code) -> tag (LLM) -> assemble + gate + repair (code)
    -> judge (LLM) -> cluster (code) -> name clusters (LLM)
    -> rank (code) -> output (code)

The model never sees or emits a timestamp. It names beat ids; code owns time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pf import parse as P
from pf import tag as T
from pf import assemble as A
from pf import judge as J
from pf import cluster as C
from pf import rank as R
from pf import report as REP
from pf import evaluate as E


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        return _mini_yaml(text)


def _mini_yaml(text):
    """Loads the subset of YAML this config uses, so the tool has no
    third-party dependency. Nested maps, scalars, inline lists, comments."""
    import ast

    lines = []
    for raw in text.split("\n"):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        lines.append(raw.split("  #")[0].rstrip())

    root = {}
    stack = [(-1, root)]
    for i, line in enumerate(lines):
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        key, val = key.strip(), val.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if val == "":
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            if nxt is not None and (len(nxt) - len(nxt.lstrip())) > indent:
                node = {}
                parent[key] = node
                stack.append((indent, node))
            else:
                parent[key] = None          # e.g. "provider:" left blank
        else:
            try:
                parent[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                parent[key] = {"true": True, "false": False,
                               "null": None, "~": None}.get(val.lower(), val)
    return root


def main():
    ap = argparse.ArgumentParser(description="Find promos in microdrama screenplays.")
    ap.add_argument("--input", help="folder of screenplay .txt files")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--provider", help="override llm.provider (use 'mock' for no key)")
    ap.add_argument("--out", help="override output_dir")
    ap.add_argument("--limit", type=int, help="only process the first N episodes")
    ap.add_argument("--no-cluster-naming", action="store_true")
    ap.add_argument("--eval-gold", help="score a gold-set CSV against promos.json")
    ap.add_argument("--score-blind-ab", action="store_true")
    ap.add_argument("--hook-points-template", metavar="EPISODES",
                    help="write outputs/hook_points_template.csv for these "
                         "episodes (e.g. 1,4,7,12) and stop. Evaluation tier "
                         "3: mark only where you would START, no end points")
    ap.add_argument("--lenient-parse", action="store_true",
                    help="downgrade parser assertions (diagnostic only)")
    ap.add_argument("--check", action="store_true",
                    help="list the models this key can call, then try one tiny "
                         "call with each configured model, and stop")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(args.config if os.path.isabs(args.config)
                      else os.path.join(here, args.config))
    if args.provider:
        cfg["llm"]["provider"] = args.provider
    if args.no_cluster_naming:
        cfg["clustering"]["llm_naming"] = False

    # Provider request knobs live in config.yaml like everything else tunable.
    # They are applied to the model adapter here, in one place, so no stage
    # has to thread them through its own call.
    from pf import llm as _llm
    if cfg["llm"].get("max_tokens"):
        _llm.MAX_TOKENS = int(cfg["llm"]["max_tokens"])
    if cfg["llm"].get("max_tokens_by_model"):
        _llm.MAX_TOKENS_BY_MODEL = dict(cfg["llm"]["max_tokens_by_model"])
    if "reasoning_effort" in cfg["llm"]:
        _llm.REASONING_EFFORT = cfg["llm"]["reasoning_effort"] or ""
    if cfg["llm"].get("reasoning_effort_by_model"):
        _llm.REASONING_EFFORT_BY_MODEL = dict(
            cfg["llm"]["reasoning_effort_by_model"])
    if cfg["llm"].get("min_call_interval_seconds") is not None:
        _llm.MIN_CALL_INTERVAL = float(cfg["llm"]["min_call_interval_seconds"])

    outdir = args.out or os.path.join(here, cfg.get("output_dir", "outputs"))
    os.makedirs(outdir, exist_ok=True)

    if args.check:
        from pf.llm import detect_provider, list_models, call, ENV_KEYS, LLMError
        prov = cfg["llm"].get("provider") or detect_provider()
        print(f"provider: {prov}")
        if prov == "mock":
            print("No API key found. Set one of: "
                  + ", ".join(v for v in ENV_KEYS.values() if v))
            return
        key = os.environ.get(ENV_KEYS[prov], "")
        try:
            available = list_models(prov, key)
        except Exception as exc:                       # noqa: BLE001
            print(f"could not list models: {exc}")
            available = []
        if available:
            print(f"\n{len(available)} models this key can call:")
            for m in available:
                print(f"  {m}")
        configured = {k: cfg["llm"][k] for k in
                      ("tag_model", "judge_model", "cluster_model")}
        print("\nconfigured in config.yaml:")
        for role, model in configured.items():
            mark = "ok " if (not available or model in available) else "NOT LISTED"
            print(f"  {role:<14} {model:<34} {mark}")
        print("\ntest call with each configured model:")
        for role, model in configured.items():
            try:
                # Groq's json_object response_format requires the literal word
                # "json" somewhere in the messages, so the probe says it too.
                call('Reply with exactly this json and nothing else: {"ok": true}',
                     model=model, provider=prov, use_cache=False)
                print(f"  {role:<14} {model:<34} WORKS")
            except LLMError as exc:
                print(f"  {role:<14} {model:<34} FAILED\n      {exc}")
        return

    if args.hook_points_template:
        import csv as _csv
        eps = [int(x) for x in
               args.hook_points_template.replace(" ", "").split(",") if x]
        path = os.path.join(outdir, "hook_points_template.csv")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["episode", "start (m:ss.s)", "hook_line", "notes"])
            for ep in eps:
                w.writerow([ep, "", "", ""])
        print(f"wrote {path} for episodes {eps}")
        print("Mark only where you would START, one row per promo you would "
              "cut; add rows as needed. Leave the end time out entirely.")
        print("Score it with:  python run.py --eval-gold " + path)
        print("It will be scored under the start-only rule (same episode, "
              "start within "
              f"{cfg['evaluation']['match_start_within']}s, no IoU) against a "
              f"random baseline of "
              f"{cfg['evaluation'].get('match_start_only_baseline')}.")
        return

    if args.score_blind_ab:
        res = E.score_blind_ab(os.path.join(outdir, "blind_ab_answers.csv"),
                               os.path.join(outdir, "blind_ab_key.json"))
        print(json.dumps(res, indent=2))
        return

    if args.eval_gold:
        with open(os.path.join(outdir, "promos.json"), encoding="utf-8") as fh:
            promos = json.load(fh)
        tool_rows = [{"episode": r["candidate"]["episode"],
                      "start_s": r["candidate"]["segments"][0]["t_start"],
                      "end_s": r["candidate"]["segments"][0]["t_end"],
                      "rank": r["rank"]}
                     for r in promos["list_a"]]
        res = E.score_against_gold(E.load_gold(args.eval_gold), tool_rows, cfg)
        with open(os.path.join(outdir, "gold_metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print(json.dumps(res, indent=2))
        return

    if not args.input:
        ap.error("--input is required unless using --eval-gold or --score-blind-ab")

    t_start = time.time()

    # Resolve the provider up front and refuse to run on the mock unless the
    # mock was asked for. A silent fall-back to mock is the worst failure this
    # tool could have: it produces a complete, plausible set of outputs that
    # are entirely synthetic.
    from pf.llm import detect_provider, ENV_KEYS, LLMError
    provider = cfg["llm"].get("provider") or detect_provider()
    if provider == "mock" and (args.provider or "").lower() != "mock":
        keys = ", ".join(v for v in ENV_KEYS.values() if v)
        sys.exit(
            "\nNo API key found, so the pipeline would run on the MOCK model and\n"
            "produce synthetic results that look real.\n\n"
            f"  Set one of: {keys}\n"
            '  PowerShell:  $env:GEMINI_API_KEY = "your-key"\n'
            "  bash:        export GEMINI_API_KEY=your-key\n\n"
            "Or pass --provider mock if you meant to exercise the pipeline "
            "without a model.\n"
        )

    # PREPARED, INACTIVE. A per-role provider, so one stage can be moved to
    # another provider when a daily token budget stalls the run: free tiers
    # meter tokens per day PER MODEL, and the tag pass alone needs most of
    # one model's day. With no `provider_by_role` in config.yaml, every role
    # gets the provider above and behaviour is byte-for-byte unchanged.
    by_role = cfg["llm"].get("provider_by_role") or {}
    for role, prov in by_role.items():
        need = ENV_KEYS.get(prov)
        if prov != "mock" and need and not os.environ.get(need):
            sys.exit(f"\nprovider_by_role sets {role!r} to {prov!r}, which "
                     f"needs {need} in the environment.\n")
    if by_role:
        print(f"      per-role providers: "
              + ", ".join(f"{r}={p}" for r, p in sorted(by_role.items())))

    def llm_for(role):
        """cfg['llm'] with this role's provider resolved."""
        chosen = by_role.get(role)
        return dict(cfg["llm"], provider=chosen) if chosen else cfg["llm"]

    print(f"[1/7] parsing {args.input}")
    episodes, beats, alias = P.parse_pack(args.input, strict=not args.lenient_parse)
    if args.limit:
        keep = {m["episode"] for m in episodes[:args.limit]}
        episodes = [m for m in episodes if m["episode"] in keep]
        beats = [b for b in beats if b.episode in keep]
    beats_by_ep = {}
    for b in beats:
        beats_by_ep.setdefault(b.episode, []).append(b)
    merges = {k: v for k, v in alias.items() if k != v}
    print(f"      {len(episodes)} episodes, {len(beats)} beats, "
          f"{sum(len(b.clauses) for b in beats)} clauses, "
          f"{len(merges)} speaker aliases merged")
    P.write_beats(beats, os.path.join(outdir, "beats.jsonl"))

    # One episode's model call failing used to abort the whole run, losing
    # every episode already done. Both observed failures were real: an
    # oversized judge prompt (HTTP 413) and the daily token ceiling (HTTP 429
    # after the retry). LLMError is caught rather than only LLMClientError,
    # because the 413 raises the latter but the exhausted-budget 429 raises
    # the former, and catching only one would leave the other aborting.
    log, tag_stats, all_cands = [], [], []
    failed_eps = {}

    def note_failure(stage, episode, exc):
        failed_eps[episode] = f"{stage}: {type(exc).__name__}"
        log.append({"stage": stage, "episode": episode, "beat_range": "",
                    "gate_fired": "episode_failed",
                    "measured_value": " ".join(str(exc).split())[:300],
                    "limit": "the model call must succeed",
                    "action": "reject", "repair_attempts": 0})
        print(f"      WARNING: episode {episode} failed at {stage} "
              f"({type(exc).__name__}); continuing")

    def too_many_failed():
        """More than a third of episodes failing is not a bad episode, it is
        a broken run. Say so loudly and stop rather than publish a shortlist
        drawn from whatever happened to get through."""
        if len(failed_eps) * 3 > len(episodes):
            sys.exit(
                f"\n*** {len(failed_eps)} of {len(episodes)} EPISODES FAILED "
                f"their model call. That is more than a third, so this run is\n"
                f"    not trustworthy and has been stopped.\n\n"
                f"    Failed: {sorted(failed_eps)}\n"
                f"    Last reason: {list(failed_eps.values())[-1]}\n\n"
                f"    Completed calls are cached, so fixing the cause and "
                f"re-running resumes rather than restarts.\n")

    print(f"[2/7] tagging ({len(episodes)} calls, "
          f"provider={llm_for('tag').get('provider') or provider})")
    tagged_all = {}
    for m in episodes:
        try:
            tg, st = T.tag_episode(m, beats_by_ep[m["episode"]],
                                   llm_for("tag"), log)
        except LLMError as exc:
            note_failure("tag", m["episode"], exc)
            continue
        tagged_all[m["episode"]] = tg
        tag_stats.append(st)
    too_many_failed()

    # Write THIS run's tag output. The report samples the hooks that were
    # tagged, and it must read them from here. Recovering them from the
    # response cache instead mixes in other runs: the cache is keyed by
    # prompt hash and persists across runs, so a 5-episode report sampled
    # hooks from all 50 episodes of an earlier run.
    with open(os.path.join(outdir, "tags.json"), "w", encoding="utf-8") as fh:
        json.dump({"episodes": sorted(tagged_all),
                   "tags": {str(k): v for k, v in tagged_all.items()}},
                  fh, indent=2, ensure_ascii=False)

    print("[3/7] assembling and gating")
    per_ep_generated = {}
    # Only episodes that actually got tagged. An episode whose tag call failed
    # has no beat ids to assemble from, and is already counted as failed.
    for m in [e for e in episodes if e["episode"] in tagged_all]:
        cands = A.build_candidates(m, beats_by_ep[m["episode"]],
                                   tagged_all[m["episode"]], cfg, log)
        per_ep_generated[m["episode"]] = len(cands)
        all_cands.append((m, A.shortlist(cands, cfg)))
    generated = sum(per_ep_generated.values())
    shortlisted = sum(len(c) for _, c in all_cands)
    print(f"      {generated} candidates passed the gates, "
          f"{shortlisted} shortlisted, {len(log)} rejections logged")

    # An episode that yields nothing used to pass in silence. Say so, and say
    # it loudly if it is happening across the pack: that means the pack, the
    # gates or the tagger is wrong, not that the show has no good moments.
    empty_eps = sorted(ep for ep, n in per_ep_generated.items() if n == 0)
    if empty_eps:
        shown = ", ".join(str(e) for e in empty_eps[:12])
        more = f" (+{len(empty_eps) - 12} more)" if len(empty_eps) > 12 else ""
        print(f"      WARNING: {len(empty_eps)} of {len(episodes)} episodes "
              f"produced 0 candidates: {shown}{more}")
        if len(empty_eps) * 3 > len(episodes):
            print(f"      *** MORE THAN A THIRD OF EPISODES YIELDED NOTHING "
                  f"({len(empty_eps)}/{len(episodes)}). Check the pack format, "
                  f"the duration gates and the tagger before trusting this "
                  f"run. ***")

    print(f"[4/7] judging ({len(all_cands)} batched calls)")
    scored = []
    for m, cands in all_cands:
        try:
            s, _ = J.judge_episode(m, beats_by_ep[m["episode"]], cands,
                                   llm_for("judge"), log)
        except LLMError as exc:
            note_failure("judge", m["episode"], exc)
            continue
        scored.extend(s)
    too_many_failed()
    print(f"      {len(scored)} candidates scored")

    print("[5/7] clustering")
    clusters = C.cluster(scored, beats_by_ep, cfg, log)
    clusters = C.name_clusters(clusters, beats_by_ep,
                               dict(cfg, llm=llm_for("cluster")), log)
    print(f"      {len(clusters)} groups")

    print("[6/7] ranking")
    ctx = {}
    for tg in tagged_all.values():
        ctx.update(tg.get("context_dependency") or {})
    list_a, list_b, diag = R.rank(scored, clusters, ctx, cfg)
    print(f"      List A {len(list_a)} (standalone gate "
          f"{diag['standalone_gate_used']}), List B {len(list_b)}")

    invalid = sum(1 for r in log if r.get("gate_fired") == "beat_id_not_found")
    missing_dims = sum(1 for r in log
                       if r.get("gate_fired") == "judge_dimension_missing")
    funnel = {
        "episodes": len(episodes),
        "episodes_failed": len(failed_eps),
        "episodes_with_no_candidates": len(empty_eps),
        "beats": len(beats),
        "hooks_tagged": sum(s["hooks"] for s in tag_stats),
        "cliffhangers_tagged": sum(s["cliffhangers"] for s in tag_stats),
        "invalid_beat_references": invalid,
        "candidates_passing_gates": generated,
        "candidates_shortlisted": shortlisted,
        "candidates_scored": len(scored),
        "judge_dimension_missing": missing_dims,
        "list_a": len(list_a),
        "list_b": len(list_b),
        "rejections_logged": len(log),
        "runtime_seconds": round(time.time() - t_start, 1),
    }

    print("[7/7] writing outputs")
    REP.write_outputs(list_a, list_b, clusters, diag, funnel, cfg, outdir, beats_by_ep)
    REP.write_rejection_log(log, outdir)
    sheet, n_pairs = E.make_blind_ab(list_a, beats_by_ep, cfg, outdir)
    with open(os.path.join(outdir, "funnel.json"), "w", encoding="utf-8") as fh:
        json.dump({"funnel": funnel, "diagnostics": diag,
                   "gate_admission_table": diag["gate_admission_table"]}, fh, indent=2)

    print()
    for k, v in funnel.items():
        print(f"  {k:<32} {v}")
    print(f"\n  outputs -> {outdir}")
    print(f"  blind A/B sheet ({n_pairs} pairs) -> {sheet}")


if __name__ == "__main__":
    main()
