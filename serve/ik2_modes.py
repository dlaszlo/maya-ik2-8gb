"""maya-ik2's drafting modes: the token ids the engine reads what the model is writing from.

With --draft the engine decides every step whether a draft is worth its time, and what it learns about that is a
mode's own (src/ik2/ik2_spec_ctl.hpp): the thinking, an answer's text, each tool's call, and code inside a fence
accept drafts very differently.  The engine works on token ids and has no tokenizer, so the server gives it the ids
of the model's own markers in four environment variables:

  IK2_MODE_START  a message's start (<|message_model|>): a tool's name follows it, then the content marker
  IK2_MODE_MARKS  the content markers (<|content_thinking|>, <|content_text|>, <|content_invoke_tool_json|>, ...)
  IK2_MODE_FENCE  the tokens with ``` in them (a fence opens or closes)
  IK2_MODE_DRAFT  the content markers drafting is on in (the tool calls: <|content_invoke_tool_json|>, ..._text).
                  With it the switch is a fixed one: drafts in a tool call's content and inside a code fence, none in
                  the thinking or an answer's prose.  Set it to "" in a config's env and every mode drafts when its
                  own acceptance says so.

A config's own "env" wins: a key it sets is left as it is (an empty value turns the modes off).
"""
import json
import os

START = ("<|message_model|>",)
FENCE = "```"


def mode_env(tokenizer_dir: str) -> dict:
    """The three variables for the tokenizer in tokenizer_dir (its vocab.json); {} when it has no content markers."""
    try:
        with open(os.path.join(tokenizer_dir, "vocab.json"), encoding="utf-8") as f:
            vocab = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(vocab, dict):
        return {}
    marks = sorted(i for t, i in vocab.items()
                   if t.startswith("<|content_") and t.endswith("|>") and isinstance(i, int))
    if not marks:
        return {}
    starts = [vocab[t] for t in START if isinstance(vocab.get(t), int)]
    fences = sorted(i for t, i in vocab.items() if FENCE in t and isinstance(i, int))
    def ids(v):
        return ",".join(str(i) for i in v)

    drafts = sorted(i for t, i in vocab.items()
                    if t.startswith("<|content_invoke_tool") and t.endswith("|>") and isinstance(i, int))
    return {"IK2_MODE_START": ids(starts), "IK2_MODE_MARKS": ids(marks), "IK2_MODE_FENCE": ids(fences),
            "IK2_MODE_DRAFT": ids(drafts)}


def export(env: dict, cfg: dict) -> None:
    """Adds the variables to the engine's environment when the config drafts (--draft in its args)."""
    args = [str(a) for a in (cfg.get("args") or [])]
    if "--draft" not in args or not cfg.get("tokenizer"):
        return
    tok = cfg["tokenizer"]
    if not os.path.isabs(tok) and cfg.get("cwd"):
        tok = os.path.join(cfg["cwd"], tok)
    for k, v in mode_env(tok).items():
        env.setdefault(k, v)
