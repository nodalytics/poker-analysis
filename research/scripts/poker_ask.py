"""Ask each model whether it can play poker, then check the claim against two questions with known answers.

Usage: python3 poker_ask.py OUT.json MODEL [MODEL ...]   (talks to Ollama on 127.0.0.1:11434)

- self_assessment: free answer, read by hand.
- flush_outs: AhKh on Qh9h3d2s; exactly 9 hearts remain, so the answer must be 9.
- push_decision: 10 big blinds on the button with A7 offsuit, folded to you. Every published push/fold
  chart has this as a shove, so the answer must be all-in (not a min-raise, limp or fold).
"""

import json
import re
import sys
import urllib.request

PROMPTS = {
    "self_assessment": "Can you play Texas Hold'em poker? Be honest about what you can and cannot do well, "
    "for example strategy, maths, reading opponents, or playing live in a real game.",
    "flush_outs": "Texas Hold'em. You hold Ah Kh. The board is Qh 9h 3d 2s. How many cards left in the deck "
    "give you a flush on the river? Reply with the number first.",
    "push_decision": "Tournament, 9 players at the table. You are on the button with 10 big blinds and hold "
    "Ace-Seven offsuit. Everyone folds to you. What is your action: fold, limp, min-raise, or all-in? "
    "Reply with the action first, then one sentence why.",
}


def ask(model, prompt):
    body = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0, "seed": 7, "num_predict": 350},
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["message"]["content"]


def grade(name, text):
    head = text.strip()[:120].lower()
    if name == "flush_outs":
        # The final answer wins when the model works through it first (Qwen ends with \\boxed{9}).
        final = re.findall(r"boxed\{(\d+)\}|\*\*(\d+)\*\*", text)
        if final:
            return [x for x in final[-1] if x][0] == "9"
        m = re.search(r"\d+", head)
        return m is not None and m.group() == "9"
    if name == "push_decision":
        first = re.split(r"[.,\n]", head)[0]
        return bool(re.search(r"all[- ]?in|shove|push|jam", first)) and not re.search(r"\bfold\b|limp|min", first)
    return None


if __name__ == "__main__":
    out, models = sys.argv[1], sys.argv[2:]
    results = []
    for m in models:
        row = {"model": m, "answers": {}}
        for name, prompt in PROMPTS.items():
            text = ask(m, prompt)
            row["answers"][name] = {"text": text, "correct": grade(name, text)}
            print(f"{m} {name}: correct={row['answers'][name]['correct']} | {' '.join(text.split())[:160]}", flush=True)
        results.append(row)
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
