"""Embeddings and retrieval (RAG) tested against ground truth: the solver library and our own docs.

Usage: uv run python scripts/rag_eval.py LIBRARY_DIR DOCS_DIR OUTDIR [N_QUERIES]
Needs Ollama on 127.0.0.1:11434 with the embedding models below and the chat models in CHAT_MODELS.

A. Spot retrieval. Every flop decision in the solver library (solver_library.py) becomes a text description
   (board, hand, made hand, draws, decision point, stack depth). The same 25 held-out flops as ml_flop.py are the
   queries; the other 75 are the store. For each embedding model, the 5 nearest stored spots (cosine) predict the
   solver's top action by averaging their strategies. Compared with: the decision point's average strategy (what
   random retrieval converges to), 5 nearest neighbours on ml_flop.py's hand-made features, and the trained MLP.
B. LLM with RAG. A sample of the queries goes to each chat model three ways: alone; with the 5 spots retrieved by
   the best embedding model and the solver's strategy for each; with the 5 found on hand-made features. Scored on
   agreement with the solver's top action.
C. Question answering over docs/. The write-ups are cut into chunks and embedded; fixed questions with answers
   stated in the docs are asked without context and with the top 4 chunks. Retrieval is scored on whether a
   retrieved chunk contains the answer, answering on whether the reply does.
"""

import glob
import gzip
import json
import os
import random
import re
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_flop  # noqa: E402

OLLAMA = "http://127.0.0.1:11434"
EMBED_MODELS = ["nomic-embed-text", "mxbai-embed-large", "bge-m3"]
CHAT_MODELS = ["qwen2.5vl:32b", "hf.co/TheDrummer/Tiger-Gemma-9B-v3-GGUF:Q4_K_M"]
SLOT_NAMES = ["check/call", "bet 2/3 pot", "all-in", "fold"]
NODE_TEXT = {
    "oop_open": "first to act, out of position, no bet yet",
    "ip_after_check": "in position, opponent checked",
    "ip_facing_bet": "in position, facing a two-thirds-pot bet",
}
QUESTIONS = [  # (question, regex the answer must contain); every answer is stated in docs/
    ("At 10 big blinds heads-up, what share of hands does the small blind shove in our push/fold equilibrium?", r"59"),
    ("How many solver solves was the flop model trained on?", r"400"),
    ("What top-action agreement did the extra-trees flop model reach on held-out flops?", r"82\.5"),
    (
        "Which model's self-learning copy beat its frozen twin at poker past the multiple-testing correction?",
        r"(?i)tiger-?gemma[ -]?9b",
    ),
    ("How long does the optimised TexasSolver build take for the flop solve on 16 threads?", r"41"),
    ("What VPIP did Pluribus play in the hand-history study?", r"26"),
    ("How many Pluribus spots did the backtest sample?", r"300"),
    ("What 3-bet frequency did Pluribus show in the hand-history study?", r"8\.6"),
    ("Which models were retired and deleted from the lab?", r"(?i)mistral"),
    ("What was the Windows VM's evaluation edition?", r"(?i)enterprise"),
    ("At short stacks with the solver acting, in how many of 88 sessions were the solver seats ahead?", r"56"),
    ("What exploitability did the textbook river game reach in the solver demo?", r"0\.06"),
]


def post(path, body, timeout=600):
    req = urllib.request.Request(OLLAMA + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def embed(model, texts, batch=64):
    out = []
    for k in range(0, len(texts), batch):
        out += post("/api/embed", {"model": model, "input": texts[k : k + batch]})["embeddings"]
    v = np.array(out, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def chat(model, prompt, n=80):
    try:
        r = post(
            "/api/chat",
            {
                "model": model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0, "num_predict": n},
            },
        )
        return r["message"]["content"]
    except OSError:
        return ""


def describe(board, hand, node, spot):
    import eval7

    b = [eval7.Card(c) for c in board]
    h = [eval7.Card(hand[:2]), eval7.Card(hand[2:])]
    cat = eval7.handtype(eval7.evaluate(h + b))
    suits = [c[1] for c in board]
    texture = {1: "monotone", 2: "two-tone", 3: "rainbow"}[len(set(suits))]
    ranks = sorted((ml_flop.RANKS.index(c[0]) for c in board), reverse=True)
    paired = "paired" if len(set(ranks)) < 3 else "unpaired"
    hr = sorted((ml_flop.RANKS.index(hand[0]), ml_flop.RANKS.index(hand[2])), reverse=True)
    pair = next((["top", "middle", "bottom"][k] for k, r in enumerate(ranks) if r in hr), None)
    flush_draw = max((suits + [hand[1], hand[3]]).count(s) for s in "cdhs") == 4
    allr = sorted(set(ranks) | set(hr))
    run = best = 1
    for x, y in zip(allr, allr[1:]):
        run = run + 1 if y == x + 1 else 1
        best = max(best, run)
    over = sum(r > ranks[0] for r in hr)
    made = cat + (f" ({pair} pair)" if cat == "Pair" and pair else "")
    if cat == "Pair" and hr[0] == hr[1] and hr[0] > ranks[0]:
        made = "Pair (overpair)"
    draws = ", ".join(x for x, ok in (("flush draw", flush_draw), ("straight draw", best >= 4)) if ok) or "no draw"
    depth = "deep (stack 9.5x the pot)" if spot.startswith("spr9.5") else "shallow (stack 4.5x the pot)"
    ranges = "wide against wide" if spot.endswith("_wide") and "tight" not in spot else "wide against a tight range"
    return (
        f"Flop {' '.join(board)} ({texture}, {paired}, high card {ml_flop.RANKS[ranks[0]]}). "
        f"Hand {hand[:2]} {hand[2:]}: {made}; {draws}; {over} overcard(s). "
        f"Decision: {NODE_TEXT[node]}. Stacks {depth}. Ranges {ranges}."
    )


def load_spots(libdir):
    rows = []
    for path in sorted(glob.glob(os.path.join(libdir, "solves", "*.json.gz"))):
        sid = os.path.basename(path)[: -len(".json.gz")]
        flop_s, spot = sid.split("_", 1)
        board = [flop_s[i : i + 2] for i in (0, 2, 4)]
        tree = json.load(gzip.open(path, "rt"))
        nodes = [("oop_open", tree)]
        for name, child in (tree.get("childrens") or {}).items():
            if child.get("node_type") == "action_node":
                nodes.append(("ip_after_check" if name.startswith("CHECK") else "ip_facing_bet", child))
        for node_name, node in nodes:
            acts = node.get("actions", [])
            for hand, probs in ((node.get("strategy") or {}).get("strategy") or {}).items():
                y = np.zeros(4)
                for a, p in zip(acts, probs):
                    y[ml_flop.slot(a)] += p
                rows.append({"flop": flop_s, "board": board, "hand": hand, "node": node_name, "spot": spot, "y": y})
    return rows


def mix_text(y):
    return ", ".join(f"{SLOT_NAMES[k]} {v:.0%}" for k, v in enumerate(y) if v >= 0.005)


def parse_choice(text):
    low = (text or "").lower()
    m = re.search(r"answer\s*[:=]\s*\**\s*(check|call|bet|all-in|all in|fold)", low)
    w = (
        m.group(1)
        if m
        else next((x for x in ("all-in", "all in", "fold", "bet", "check", "call") if x in low), "check")
    )
    return {"check": 0, "call": 0, "bet": 1, "all-in": 2, "all in": 2, "fold": 3}[w]


def part_ab(libdir, outdir, n_queries, rep):
    from sklearn.model_selection import GroupShuffleSplit

    rows = load_spots(libdir)
    G = np.array([r["flop"] for r in rows])
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(G, G, G))  # = ml_flop.py
    rng = random.Random(7)
    store = [rows[k] for k in rng.sample(list(tr), min(6000, len(tr)))]
    queries = [rows[k] for k in rng.sample(list(te), n_queries)]
    Ys, Yq = np.array([r["y"] for r in store]), np.array([r["y"] for r in queries])
    truth = Yq.argmax(1)
    agree = lambda P: round(float((P.argmax(1) == truth).mean()), 4)  # noqa: E731
    res = {"store": len(store), "queries": len(queries), "retrieval": {}}
    node_mean = {n: Ys[[r["node"] == n for r in store]].mean(0) for n in ml_flop.NODES}
    res["retrieval"]["decision-point average (random retrieval)"] = agree(
        np.array([node_mean[r["node"]] for r in queries])
    )
    feats = lambda rs: np.array([ml_flop.features(r["board"], r["hand"], r["node"], r["spot"]) for r in rs])  # noqa: E731
    Fs, Fq = feats(store), feats(queries)
    knn_feat = [np.argsort(((Fs - f) ** 2).sum(1))[:5] for f in Fq]
    res["retrieval"]["5-NN on hand-made features"] = agree(np.array([Ys[ix].mean(0) for ix in knn_feat]))
    try:
        from joblib import load

        mlp = load(os.path.join(os.path.dirname(outdir.rstrip("/")) or ".", "mlflop", "mlp.joblib"))
        res["retrieval"]["trained MLP (ml_flop.py)"] = agree(ml_flop.normalise(mlp.predict(Fq)))
    except (OSError, ImportError):
        pass
    ds, dq = [describe(r["board"], r["hand"], r["node"], r["spot"]) for r in store], None
    dq = [describe(r["board"], r["hand"], r["node"], r["spot"]) for r in queries]
    knn_emb = {}
    for m in EMBED_MODELS:
        t0 = time.time()
        Es, Eq = embed(m, ds), embed(m, dq)
        ix = [np.argsort(-(Es @ q))[:5] for q in Eq]
        knn_emb[m] = ix
        res["retrieval"][f"5-NN on {m} embeddings"] = agree(np.array([Ys[k].mean(0) for k in ix]))
        res.setdefault("embed_seconds", {})[m] = round(time.time() - t0, 1)
        print(m, res["retrieval"][f"5-NN on {m} embeddings"], flush=True)
    best = max(EMBED_MODELS, key=lambda m: res["retrieval"][f"5-NN on {m} embeddings"])
    res["best_embedding"] = best
    rep["A_spot_retrieval"] = res
    # B: the LLM alone, with retrieved solved spots (best embedding), with feature-retrieved spots
    sample = rng.sample(range(len(queries)), min(120, len(queries)))
    ask = (
        "You are playing No-Limit Hold'em. {d}\nOptions: check/call, bet (two-thirds of the pot), all-in"
        "{f}.\n{ctx}Which single action does a game-theory solver take most often here? "
        "End with 'Answer: <action>'."
    )
    out = {}
    for model in CHAT_MODELS:
        for flow, ixs in (("alone", None), (f"RAG ({best})", knn_emb[best]), ("RAG (features)", knn_feat)):
            hits = 0
            for q in sample:
                r = queries[q]
                ctx = ""
                if ixs is not None:
                    ctx = (
                        "Similar spots solved by TexasSolver:\n"
                        + "\n".join(f"- {ds[k]} Solver: {mix_text(Ys[k])}." for k in ixs[q])
                        + "\n"
                    )
                fold = ", fold" if r["node"] == "ip_facing_bet" else ""
                hits += parse_choice(chat(model, ask.format(d=dq[q], f=fold, ctx=ctx), 200)) == truth[q]
            out[f"{model.split('/')[-1]} | {flow}"] = round(hits / len(sample), 4)
            print(model, flow, out[f"{model.split('/')[-1]} | {flow}"], flush=True)
    rep["B_llm_with_rag"] = {"queries": len(sample), "agreement_with_solver_top_action": out}


def part_c(docs_dir, rep):
    chunks = []
    for f in sorted(glob.glob(os.path.join(docs_dir, "*.md"))):
        for sec in re.split(r"(?m)^(?=## )", open(f).read()):
            for k in range(0, len(sec), 1200):
                if sec[k : k + 1200].strip():
                    chunks.append(f"[{os.path.basename(f)}] " + sec[k : k + 1200])
    res = {"chunks": len(chunks), "retrieval_hit": {}, "answer_correct": {}}
    best, best_hit, top = None, -1, {}
    for m in EMBED_MODELS:
        E, Q = embed(m, chunks), embed(m, [q for q, _ in QUESTIONS])
        top[m] = [np.argsort(-(E @ q))[:4] for q in Q]
        hit = sum(any(re.search(rx, chunks[k]) for k in ix) for ix, (_, rx) in zip(top[m], QUESTIONS))
        res["retrieval_hit"][m] = f"{hit}/{len(QUESTIONS)}"
        if hit > best_hit:
            best, best_hit = m, hit
    res["best_embedding"] = best
    for model in CHAT_MODELS:
        for flow in ("alone", f"RAG ({best})"):
            ok = 0
            for (q, rx), ix in zip(QUESTIONS, top[best]):
                ctx = (
                    ""
                    if flow == "alone"
                    else "Use these notes from our research:\n" + "\n---\n".join(chunks[k] for k in ix) + "\n\n"
                )
                ok += bool(re.search(rx, chat(model, ctx + q + " Answer briefly.", 120)))
            res["answer_correct"][f"{model.split('/')[-1]} | {flow}"] = f"{ok}/{len(QUESTIONS)}"
            print(model, flow, ok, flush=True)
    rep["C_docs_qa"] = res


def main(libdir, docs_dir, outdir, n_queries=600):
    os.makedirs(outdir, exist_ok=True)
    rep = {}
    part_c(docs_dir, rep)
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    part_ab(libdir, outdir, n_queries, rep)
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], *(int(x) for x in sys.argv[4:5]))
