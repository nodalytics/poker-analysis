"""Context management for small local models: chunked tool output, conversation compaction, memory compression.

- Chunks: a tool result longer than CHUNK characters is stored and returned one page at a time, ending in a note
  telling the model to call `more_output` with the id for the next page. Nothing is silently cut off.
- Compaction: when a conversation's estimated size passes COMPACT_AT of the model's context, the turns between
  the task and the most recent KEEP_RECENT messages are summarised by the model itself into one progress note.
  The system prompt and the task stay verbatim; tool calls and their results are never split from each other.
- Memory compression: a learner's memory file over its cap is condensed by the model instead of losing the oldest
  lines.
Token counts are estimated at 3.5 characters a token, which errs towards compacting early.
"""

import json
import urllib.request

OLLAMA = "http://127.0.0.1:11434"
CHUNK, COMPACT_AT, KEEP_RECENT = 2500, 0.7, 6
CONTEXT = {"wizard-vicuna": 2048}  # everything else is served at 8192 (OLLAMA_CONTEXT_LENGTH)

MORE_OUTPUT_TOOL = {
    "type": "function",
    "function": {
        "name": "more_output",
        "description": "Get the next page of a long tool result. Use the id given at the end of the previous page.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    },
}


def context_tokens(model):
    return next((v for k, v in CONTEXT.items() if k in model), 8192)


def estimate(messages):
    return int(sum(len(json.dumps(m)) for m in messages) / 3.5)


def _chat(model, messages, num_predict=300):
    body = {
        "model": model,
        "stream": False,
        "messages": messages,
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    req = urllib.request.Request(
        OLLAMA + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["message"].get("content", "")


class Chunks:
    def __init__(self, size=CHUNK):
        self.size, self.store, self.n = size, {}, 0
        self.pages_served = 0

    def page(self, text):
        """First page of a result; the rest is kept for more_output."""
        if len(text) <= self.size:
            return text
        self.n += 1
        cid = f"out{self.n}"
        self.store[cid] = text[self.size :]
        return f"{text[: self.size]}\n[... {len(text) - self.size} more characters: call more_output with id {cid}]"

    def more(self, cid):
        rest = self.store.get(cid)
        if rest is None:
            return f"ERROR: no more output for id {cid}"
        self.pages_served += 1
        if len(rest) <= self.size:
            del self.store[cid]
            return rest + "\n[end of output]"
        self.store[cid] = rest[self.size :]
        return f"{rest[: self.size]}\n[... {len(rest) - self.size} more characters: call more_output with id {cid}]"


class Context:
    """Holds a conversation and compacts it before it overflows the model's window."""

    def __init__(self, model, messages, head=2):
        self.model, self.messages, self.head = model, list(messages), head
        self.limit = context_tokens(model)
        self.compactions = 0

    def add(self, msg):
        self.messages.append(msg)

    def maybe_compact(self):
        if estimate(self.messages) < COMPACT_AT * self.limit or len(self.messages) <= self.head + KEEP_RECENT:
            return False
        cut = len(self.messages) - KEEP_RECENT
        while cut > self.head and self.messages[cut].get("role") == "tool":  # keep calls with their results
            cut -= 1
        middle = self.messages[self.head : cut]
        if not middle:
            return False
        transcript = "\n".join(
            f"{m.get('role')}: "
            + (m.get("content") or json.dumps([c.get("function") for c in m.get("tool_calls", [])]))[:600]
            for m in middle
        )
        summary = _chat(
            self.model,
            [
                {
                    "role": "user",
                    "content": "Summarise this part of your own work so far in at most 600 characters: what you did, what you "
                    "found, and what is left to do. Keep exact names, paths and numbers.\n\n" + transcript,
                }
            ],
        )
        self.messages = (
            self.messages[: self.head]
            + [{"role": "user", "content": f"(Earlier steps, summarised by you: {summary.strip()[:800]})"}]
            + self.messages[cut:]
        )
        self.compactions += 1
        return True


def compress_memory(model, text, cap):
    """Condense a memory file to under `cap` characters with the model's own words; fall back to the tail."""
    if len(text) <= cap:
        return text
    try:
        out = _chat(
            model,
            [
                {
                    "role": "user",
                    "content": f"These are your own notes from past games. Rewrite them as a shorter list of the lessons that still "
                    f"matter, at most {cap} characters, one '- ' line each. Keep opponent names.\n\n{text[-6000:]}",
                }
            ],
            num_predict=400,
        )
    except OSError:
        out = ""
    out = out.strip()
    return out[:cap] if out else text[-cap:]
