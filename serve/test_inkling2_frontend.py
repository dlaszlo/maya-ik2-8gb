"""serve/test_inkling2_frontend.py - Inkling-2-Small in the frontend and the server: thinkingmachines' own chat
template detected and rendered at every level a client names (its effort_map's names, thinking off as "none"), the
messages it refuses as clients send them (a tool call's arguments as a JSON string, None content) taken, its answers
- typed segments, not tags - read into reasoning, answer and tool calls however the stream is cut, its end of turn
a stop, a required tool call opened in its own form, and pictures refused in plain words (maya-ik2 has no vision
tower yet).

    python -m unittest serve.test_inkling2_frontend

Offline, no GPU: the template is tools/ik2/chat_template.jinja, the tokenizer a small stand-in with the template's
marks as single tokens, the engine a scripted one.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import string
import threading
import time
import unittest
import unittest.mock
import urllib.request
from pathlib import Path

from serve import server
from serve.frontend import ChatTemplate, InklingRewriter, OutputParser, effort_kwargs, inkling_messages, \
    openai_to_messages
from serve.server import ByteTokenizer, Service

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "tools" / "ik2" / "chat_template.jinja"
PUBLIC_SHA256 = "e9cfd8e75d1272633c2e72b3cc76e99567e2bc8912b960483aed483ad35d54c7"
USER, MODEL, SYSTEM, TOOL = "<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>"
THINK, TEXT, CALL, END = "<|content_thinking|>", "<|content_text|>", "<|content_invoke_tool_json|>", "<|end_message|>"
EOS = "<|content_model_end_sampling|>"
EFFORT = "<|message_system|><|content_text|>Thinking effort level: "
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "The weather of a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}]
ASK = [{"role": "user", "content": "Weather in Paris?"}]
PICTURE = {"role": "user", "content": [{"type": "text", "text": "What is this?"},
                                       {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}


def public() -> ChatTemplate:
    return ChatTemplate(PUBLIC)


def call(name, args) -> str:
    """A tool call as the model writes it (and its template renders a past one)."""
    return MODEL + name + CALL + json.dumps({"name": name, "args": args}, separators=(",", ":")) + END


class InkTok(ByteTokenizer):
    """ByteTokenizer with the template's marks as one token each; its eos the end of sampling (the GGUF's)."""
    SPECIALS = ByteTokenizer.SPECIALS + [USER, MODEL, SYSTEM, TOOL, THINK, TEXT, CALL, END, EOS, "<|content_xml|>",
                                         "<|content_image|>"]
    special_tokens = {s: 256 + k for k, s in enumerate(SPECIALS)}
    special_ids = {"eos": special_tokens[EOS]}


class Scripted:
    """Request k's answer is scripts[k] (the last one again after that), then the end of sampling.  Records every
    prompt.  Like maya-ik2: no logprobs, no logit bias, one conversation."""

    def __init__(self, tok, *scripts):
        self.tok, self.max_context, self.scripts = tok, 32768, scripts
        self.prompts, self.sampling = [], []
        self.info = {"engine": "ik2", "engine_kind": "ik2"}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        self.sampling.append(dict(sampling or {}))
        text = self.scripts[min(len(self.prompts), len(self.scripts)) - 1]
        for t in (self.tok.encode(text, parse_special=True) + [self.tok.special_tokens[EOS]])[:max_new]:
            yield t


def service(*scripts):
    tok = InkTok()
    eng = Scripted(tok, *scripts)
    return Service(eng, tok, public()), eng, tok


def answer(svc, req):
    msgs, tools, kw = openai_to_messages(req)
    ids, thinking, max_new = svc.prepare(msgs, tools, kw, 500)
    sampling = {k: kw.get(k) for k in ("_prefill", "_blocked", "_think_resume")}
    events, done = [], None
    for kind, x in svc.run(ids, thinking, tools, max_new, sampling, threading.Event()):
        if kind == "event":
            events.append(x)
        elif kind == "done":
            done = x
    return {"content": "".join(e.text for e in events if e.kind == "content"),
            "reasoning": "".join(e.text for e in events if e.kind == "reasoning"),
            "calls": [(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"], "done": done}


class Template(unittest.TestCase):
    def test_public_copy(self):   # (line endings aside: a checkout may rewrite them)
        raw = PUBLIC.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(raw).hexdigest(), PUBLIC_SHA256)

    def test_detected(self):
        t = public()
        self.assertTrue(t.inkling)
        self.assertFalse(t.minimax or t.deepseek or t.qwen or t.hy4 or t.glm_effort or t.glm_dsa)

    def test_other_templates_are_not(self):
        for path in ("serve/chat_template.jinja", "serve/glm5_chat_template.jinja", "tools/mm3/chat_template.jinja",
                     "serve/fixtures/hy4_flash/chat_template.jinja"):
            self.assertFalse(ChatTemplate(ROOT / path).inkling, path)

    def test_levels(self):
        # its own names reach it as they are; thinking off is its "none" (0); none named: its default, 0.9
        t = public()
        for word, num in (("none", "0"), (False, "0"), ("off", "0"), ("minimal", "0"), ("low", "0.2"),
                          ("medium", "0.7"), ("high", "0.9"), ("xhigh", "0.99"), ("max", "0.99"), ("auto", "0.9"),
                          (None, "0.9")):
            with self.subTest(word=word):
                out = t.render(ASK, **effort_kwargs(word))
                self.assertIn(EFFORT + num + END, out)
                self.assertFalse(t.opens_thinking(out))   # (the model opens its own segments)

    def test_render(self):
        out = public().render([{"role": "system", "content": "Be terse."}] + ASK)
        self.assertEqual(out, SYSTEM + TEXT + "Be terse." + END + EFFORT + "0.9" + END + USER + TEXT +
                         "Weather in Paris?" + END)

    def test_history_as_clients_send_it(self):
        # a past call's arguments as a JSON string (OpenAI's form) and its content None: the template refuses both
        # as given - they go in as an object and ""; the same prompt as the canonical form
        t = public()
        flat = [x["function"] for x in TOOLS]
        tc = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"days": 2, "city": "Paris"}'}}
        given = ASK + [{"role": "assistant", "content": None, "reasoning_content": "Look it up.", "tool_calls": [tc]},
                       {"role": "tool", "tool_call_id": "c1", "content": '{"temp": 21}'}]
        canon = json.loads(json.dumps(given))
        canon[1]["content"] = ""
        canon[1]["tool_calls"][0]["function"]["arguments"] = {"city": "Paris", "days": 2}
        out = t.render(given, tools=flat)
        self.assertEqual(out, t.template.render(messages=canon, tools=TOOLS, add_generation_prompt=True))
        self.assertIn('<|message_system|>tool_declare<|content_xml|>[{"description":"The weather of a city."', out)
        self.assertIn(MODEL + THINK + "Look it up." + END + call("get_weather", {"city": "Paris", "days": 2}) + EOS,
                      out)
        self.assertIn(TOOL + "get_weather" + TEXT + '{"temp": 21}' + END, out)

    def test_messages(self):
        got = inkling_messages([{"role": "developer", "content": "x"},
                                {"role": "assistant", "content": None, "reasoning": "r", "tool_calls": [
                                    {"function": {"name": "f", "arguments": ""}},
                                    {"function": {"name": "g", "arguments": "not json"}}]}])
        self.assertEqual(got[0], {"role": "system", "content": "x"})
        self.assertEqual((got[1]["content"], got[1]["reasoning_content"]), ("", "r"))
        self.assertEqual([c["function"]["arguments"] for c in got[1]["tool_calls"]], [{}, {"_raw": "not json"}])


class Parser(unittest.TestCase):
    OUT = {
        "think+text": (MODEL + THINK + "Let me see." + END + MODEL + TEXT + "The answer is 4." + END + EOS,
                       "Let me see.", "The answer is 4.", []),
        "text only": (MODEL + TEXT + "Hi there!" + END + EOS, "", "Hi there!", []),
        "plain": ("Just words.", "", "Just words.", []),
        "think+call": (MODEL + THINK + "Need it." + END +
                       call("get_weather", {"city": "a <b> \"c\"", "days": 3, "opts": {"x": [1, 2]}}) + EOS,
                       "Need it.", "", [("get_weather", {"city": "a <b> \"c\"", "days": 3, "opts": {"x": [1, 2]}})]),
        "text+call": (MODEL + TEXT + "Looking." + END + call("get_weather", {"city": "Rome"}) + EOS,
                      "", "Looking.", [("get_weather", {"city": "Rome"})]),
    }

    def feed(self, text, thinking, step):
        p, events = OutputParser(thinking=thinking, tools=[t["function"] for t in TOOLS], stream_tools=True,
                                 inkling=True), []
        for i in range(0, len(text), step):
            events += p.feed(text[i:i + step])
        events += p.finish()
        return ("".join(e.text for e in events if e.kind == "reasoning"),
                "".join(e.text for e in events if e.kind == "content").strip(),
                [(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"])

    def test_however_cut(self):
        for name, (text, reasoning, content, calls) in self.OUT.items():
            for thinking in (False, True):
                for step in (1, 2, 3, 7, 1000):
                    with self.subTest(name=name, thinking=thinking, step=step):
                        self.assertEqual(self.feed(text, thinking, step), (reasoning, content, calls))

    def test_loose_call_json(self):   # (issue #2: the comma and the quote before "args" left out)
        body = '{"name":"get_weather"args":{"city":"Paris","days":2}}'
        text = MODEL + THINK + "Let me do that." + END + MODEL + "get_weather" + CALL + body + END + EOS
        for step in (1, 3, 1000):
            with self.subTest(step=step):
                self.assertEqual(self.feed(text, True, step),
                                 ("Let me do that.", "", [("get_weather", {"city": "Paris", "days": 2})]))
        for loose in ('{"name":"get_weather",args":{"city":"Rome"}}', '{"name": "get_weather" "args": {"city": "Rome"}',
                      '{"name":"get_weather"args:{"city":"Rome"}}', '{"name":"get_weather"\u602c":{"city":"Rome"}}',
                      '{"name":"get_weather"\u55dc":"city":"Rome"}}', '{"name":"get_weather"{"city":"Rome"}}'):
            with self.subTest(body=loose):
                self.assertEqual(self.feed(MODEL + "get_weather" + CALL + loose + END + EOS, False, 1000),
                                 ("", "", [("get_weather", {"city": "Rome"})]))
        # the name alone (its message ended, or the answer at an end token): the call with no arguments
        for end in (END + EOS, ""):
            self.assertEqual(self.feed(MODEL + "get_weather" + CALL + '{"name":"get_weather"' + end, False, 4),
                             ("", "", [("get_weather", {})]))

    def test_unreadable_call_stays_text(self):
        for body in ('{"name":"get_weather"args":{"city":"Pa', '{"name":"get_weather"args":{"city":"Paris"}} and more',
                     '{"name":"get_weather"args":"Paris"}', 'not json at all'):
            with self.subTest(body=body):
                r = InklingRewriter(thinking=False)
                out = r.feed(MODEL + "get_weather" + CALL + body + END) + r.finish()
                self.assertEqual((out, r.repaired), (body, 0))
        r = InklingRewriter(thinking=False)   # (cut off inside the name: text)
        out = r.feed(MODEL + "get_weather" + CALL + '{"name":"get_wea') + r.finish()
        self.assertEqual(out, '{"name":"get_wea')

    def test_unfinished_call_is_text(self):
        r = InklingRewriter(thinking=False)
        out = r.feed(MODEL + "get_weather" + CALL + '{"name":"get_weather","args":{"ci') + r.finish()
        self.assertEqual(out, '{"name":"get_weather","args":{"ci')


# a file as a call's argument: quotes, a backslash, `<`, line ends - and no line twice (the loop guard reads a block
# written three times as a loop, also inside this template's calls)
_LETTERS = random.Random(2000)
FILE = "import os\n\n\n" + "".join(
    "def %s():\n    print(\"<b>%s</b>\", os.sep, '\\\\')\n" % tuple("".join(_LETTERS.choices(string.ascii_lowercase, k=k))
                                                                 for k in (6, 24)) for _ in range(8))


class LiveCalls(unittest.TestCase):
    """A tool call as it is written (the parser's stream_tools): the model wrote a whole file as a call's arguments
    - 6,000 tokens, three minutes - and the parser gave no event until the call's <|end_message|>: no tool_start,
    "reading the prompt" in the status all the while, no byte for a streaming client.  Now the call is announced as
    soon as its name and the opening of its arguments are read, and a string argument goes out as it is written.
    (The rewriter by itself, as in Parser above, still reads a call whole: an unreadable one is text there.)"""

    def walk(self, text, step=1, thinking=False, **kw):
        """-> (the parser, [(characters fed so far, that feed's events)], the events of its finish)."""
        p, seen = OutputParser(thinking=thinking, stream_tools=True, inkling=True, **kw), []
        for i in range(0, len(text), step):
            seen.append((min(len(text), i + step), p.feed(text[i:i + step])))
        return p, seen, p.finish()

    @staticmethod
    def summary(events):
        """-> (content, [(name, arguments, complete)], [the streamed arguments of each call, as one text])."""
        streamed = {}
        for e in events:
            if e.kind == "tool_start":
                streamed[e.call.id] = ""
            elif e.kind == "tool_args":
                streamed[e.call.id] += e.text
        return ("".join(e.text for e in events if e.kind == "content").strip(),
                [(e.call.name, e.call.arguments, e.complete) for e in events if e.kind == "tool_call"],
                list(streamed.values()))

    @staticmethod
    def loads(text):
        try:
            return json.loads(text)
        except ValueError:
            return None

    def test_a_call_is_announced_and_streamed_while_it_is_written(self):
        args = {"path": "src/main.py", "content": FILE, "mode": 420}
        body = json.dumps({"name": "write_file", "args": args}, separators=(",", ":"))
        text = MODEL + "write_file" + CALL + body + END + EOS
        opened = text.index('"args":{') + len('"args":{')
        half = text.index('"content":"') + len(FILE) // 2
        for step in (1, 3, 7):
            with self.subTest(step=step):
                p, seen, last = self.walk(text, step)
                first = next(at for at, evs in seen if evs)
                self.assertLessEqual(first, opened + step)             # announced at the arguments' opening
                early = [e for at, evs in seen if at <= half for e in evs]
                self.assertEqual([e.kind for e in early][:2], ["tool_start", "tool_args"])
                self.assertEqual(early[0].call.name, "write_file")
                sent = "".join(e.text for e in early if e.kind == "tool_args")
                self.assertTrue(sent.startswith('{"path":"src/main.py","content":"import os'), sent[:60])
                self.assertGreater(len(sent), len(FILE) // 3)          # the file's first half is on its way
                writing = sum(1 for at, evs in seen if at <= half and any(e.kind == "tool_args" for e in evs))
                self.assertGreater(writing, len(FILE) // (4 * step))   # piece by piece, not in a few lumps
                events = [e for _, evs in seen for e in evs] + last
                content, calls, streamed = self.summary(events)
                self.assertEqual((content, calls), ("", [("write_file", args, True)]))
                self.assertEqual(json.loads(streamed[0]), args)        # what was streamed is the call's arguments
                self.assertEqual({e.call.id for e in events if e.call is not None}, {early[0].call.id})
                self.assertEqual(p.malformed, [])

    def test_what_went_out_can_be_closed_at_any_point(self):
        # an escape goes out only whole (a "\" or a "\u00" alone, closed there, would be no JSON), `<` as its escape
        args = {"a": "x\\y\"z\n<|é  \U0001F600", "b": [1, {"c": "}]"}], "d": "</arg_value></tool_call>"}
        for ascii_only in (False, True):
            body = json.dumps({"name": "f", "args": args}, ensure_ascii=ascii_only)
            _, seen, _ = self.walk(MODEL + "f" + CALL + body + END, 1)
            sent, closed = "", 0
            for _, evs in seen:
                for e in evs:
                    if e.kind == "tool_args":
                        sent += e.text
                        self.assertNotIn("<", e.text)
                if sent and not sent.endswith(("}", ":")):
                    got = next((v for v in (self.loads(sent + end) for end in ('"}', "}")) if v is not None), None)
                    self.assertIsNotNone(got, sent)
                    closed += 1
            self.assertEqual(json.loads(sent), args)
            self.assertGreater(closed, 20)

    BODIES = (
        '{"name":"f","args":{"city":"Rome","days":2}}',
        '{ "name" : "f" ,\n  "args" : {\n    "city" : "Rome" ,\n    "days" : 2\n  }\n}',
        '{"name":"f","args":{}}',
        '{"name":"f","args":{"a":"","b":null,"c":true,"d":-1.5e3,"e":[],"f":{},"g":[[1,"]"],{"h":"}"}]}}',
        '{"name":"f","args":{"t":"caf\\u00e9 \\ud83d\\ude00 \\/ \\"q\\" \\\\ \\n\\t"}}',
        '{"name":"f","args":{"x":"a <b> </tool_call> <arg_key> c"}}',
        '{"name":"f","args":{"k":1,"k":2,"s":"one","s":"two"}}',                 # (JSON's own rule: the last one)
        '{"name":"f","args":{"a":1},"id":"call_7"}',                             # (a key after the arguments)
        '{"name":"f","args":{"a":"b"}',                                          # (the closing brace left out)
        '{"name":"f"args":{"city":"Rome","days":2}}',                            # (the loose opening that streams)
        '{"name":"f" args : {"city":"Rome"}}',
        '{"name":"","args":{"city":"Rome"}}',                                    # (the name from the message's head)
    )

    def test_streamed_or_whole_the_same_call(self):
        # every call that streams is the call the same body gives when it is read whole (stream_tools off: the
        # parser as it was), however the stream is cut - and what was streamed is that call's arguments
        for body in self.BODIES:
            text = MODEL + THINK + "Hm." + END + MODEL + "f" + CALL + body + END + MODEL + TEXT + "Done." + END + EOS
            whole = OutputParser(thinking=True, stream_tools=False, inkling=True)
            want = self.summary(whole.feed(text) + whole.finish())
            self.assertEqual((want[0], len(want[1])), ("Done.", 1), body)
            for step in (1, 2, 5, 1000):
                with self.subTest(body=body, step=step):
                    p, seen, last = self.walk(text, step, thinking=True)
                    events = [e for _, evs in seen for e in evs] + last
                    content, calls, streamed = self.summary(events)
                    self.assertEqual((content, calls), want[:2])
                    self.assertEqual(json.loads(streamed[0]), calls[0][1])
                    self.assertEqual("".join(e.text for e in events if e.kind == "reasoning"), "Hm.")
                    if step == 1:                            # announced before its end, not with it
                        at = next(at for at, evs in seen if any(e.kind == "tool_start" for e in evs))
                        self.assertLess(at, text.index(body) + len(body))

    def test_the_other_loose_shapes_read_whole(self):
        # a stray token where ,"args": belongs, no "args" at all, the name alone: read at the call's end, as before
        for body in ('{"name":"f"怬":{"city":"Rome"}}', '{"name":"f"嗜":"city":"Rome"}}',
                     '{"name":"f"{"city":"Rome"}}', '{"name":"f"}', '{"args":{"city":"Rome"},"name":"f"}'):
            with self.subTest(body=body):
                text = MODEL + "f" + CALL + body + END + EOS
                _, seen, last = self.walk(text, 1)
                self.assertEqual([at for at, evs in seen if evs], [text.index(END) + len(END)])
                want = {} if body == '{"name":"f"}' else {"city": "Rome"}
                self.assertEqual(self.summary([e for _, evs in seen for e in evs] + last)[1], [("f", want, True)])

    def test_an_announced_call_that_does_not_read_ends_incomplete(self):
        # it cannot become the answer's text any more: what went out stands, closed, and the call is marked
        # incomplete - the parser's rule for a call that does not parse (#613); the text after it is the answer's
        for body, sent in (('{"name":"f","args":{"city":"Pa', {"city": "Pa"}),
                           ('{"name":"f"args":{"city":"Paris"}} and more', {"city": "Paris"}),
                           ('{"name":"f","args":{"a":1,"city":"Pa\\qris"}}', {"a": 1, "city": "Pa"}),
                           ('{"name":"f","args":{"a":"x","n":12x,"b":"y"}}', {"a": "x"}),
                           ('{"name":"f","args":{"a":"x",}}', {"a": "x"}),
                           ('{"name":"f","args":{"a":"line\nbreak"}}', {"a": "line"}),
                           ('{"name":"f","args":{"a":[1,2}}', {})):
            for step in (1, 4, 1000):
                with self.subTest(body=body, step=step):
                    text = MODEL + "f" + CALL + body + END + MODEL + TEXT + "Sorry." + END + EOS
                    p, seen, last = self.walk(text, step)
                    content, calls, streamed = self.summary([e for _, evs in seen for e in evs] + last)
                    self.assertEqual((content, [c[0] for c in calls], [c[2] for c in calls]), ("Sorry.", ["f"], [False]))
                    self.assertEqual(json.loads(streamed[0]), sent)
                    self.assertEqual(len(p.malformed), 1)

    def test_the_output_ends_inside_an_announced_call(self):
        # (max_tokens, a stop: no <|end_message|>) - the parser closes what went out, as for any other model's call
        cut = MODEL + "f" + CALL + '{"name":"f","args":{"n":3,"city":"Pa'
        _, seen, last = self.walk(cut, 1)
        events = [e for _, evs in seen for e in evs] + last
        self.assertEqual(self.summary(events)[2], ['{"n":3,"city":"Pa"}'])
        self.assertEqual([e.kind for e in last][-1], "tool_call")
        _, seen, last = self.walk(cut, 1, strict_tools=True)             # (/v1/responses: incomplete, never closed)
        self.assertEqual([(e.kind, e.complete) for e in last][-1], ("tool_call", False))
        # ... and a body that is whole is the call, with or without its <|end_message|>
        for tail in ("", EOS, MODEL + TEXT + "Next." + END):
            _, seen, last = self.walk(MODEL + "f" + CALL + '{"name":"f","args":{"city":"Paris"}}' + tail, 1)
            content, calls, streamed = self.summary([e for _, evs in seen for e in evs] + last)
            self.assertEqual((calls, streamed), ([("f", {"city": "Paris"}, True)], ['{"city":"Paris"}']), tail)
            self.assertEqual(content, "Next." if "Next." in tail else "")

    def test_two_calls(self):
        text = call("f", {"city": "Rome"}) + call("g", {"q": "x" * 40, "n": 1}) + EOS
        _, seen, last = self.walk(text, 3)
        events = [e for _, evs in seen for e in evs] + last
        self.assertEqual(self.summary(events)[1:], ([("f", {"city": "Rome"}, True), ("g", {"q": "x" * 40, "n": 1}, True)],
                                                    ['{"city":"Rome"}', json.dumps({"q": "x" * 40, "n": 1},
                                                                                   separators=(",", ":"))]))
        kinds = [e.kind for e in events if e.kind in ("tool_start", "tool_call")]
        self.assertEqual(kinds, ["tool_start", "tool_call", "tool_start", "tool_call"])


class Serving(unittest.TestCase):
    def test_stop(self):
        svc, _, tok = service("x")
        self.assertIn(tok.special_tokens[EOS], svc.stop_ids)
        self.assertEqual(svc.eos_ids, {tok.special_tokens[EOS]})

    def test_thinking_and_answer(self):
        svc, eng, tok = service(MODEL + THINK + "Thinking." + END + MODEL + TEXT + "Hello." + END)
        got = answer(svc, {"messages": ASK})
        self.assertEqual((got["reasoning"], got["content"], got["calls"]), ("Thinking.", "Hello.", []))
        self.assertTrue(tok.decode(eng.prompts[0]).endswith(USER + TEXT + "Weather in Paris?" + END))

    def test_effort(self):
        svc, eng, tok = service(MODEL + TEXT + "Hello." + END)
        answer(svc, {"messages": ASK, "reasoning_effort": "none"})
        answer(svc, {"messages": ASK, "reasoning_effort": "low"})
        svc.default_effort = "xhigh"
        answer(svc, {"messages": ASK})
        said = [tok.decode(p) for p in eng.prompts]
        for text, num in zip(said, ("0", "0.2", "0.99")):
            self.assertIn(EFFORT + num + END, text)

    def test_tool_call(self):
        svc, _, _ = service(MODEL + THINK + "Look it up." + END + call("get_weather", {"city": "Paris", "days": 2}))
        got = answer(svc, {"messages": ASK, "tools": TOOLS})
        self.assertEqual(got["calls"], [("get_weather", {"city": "Paris", "days": 2})])
        self.assertEqual(got["reasoning"], "Look it up.")

    def test_required_call_opens_in_its_own_form(self):
        svc, eng, tok = service('{"name":"get_weather","args":{"city":"Rome"}}' + END)
        got = answer(svc, {"messages": ASK, "tools": TOOLS,
                           "tool_choice": {"type": "function", "function": {"name": "get_weather"}}})
        want = tok.encode(MODEL + "get_weather" + CALL, parse_special=True)
        self.assertEqual(eng.prompts[0][-len(want):], want)
        self.assertIn(EFFORT + "0" + END, tok.decode(eng.prompts[0]))   # (thinking off for a required call)
        self.assertEqual(got["calls"], [("get_weather", {"city": "Rome"})])

    def test_pictures_are_refused_in_plain_words(self):
        svc, eng, _ = service("x")
        msgs, tools, kw = openai_to_messages({"messages": [PICTURE]})
        with self.assertRaises(ValueError) as e:
            svc.prepare(msgs, tools, kw, 100)
        self.assertIn("on maya-ik2 reads text only without the image encoder", str(e.exception))
        self.assertEqual(eng.prompts, [])

    def test_logprobs_and_bias_are_refused(self):   # (an engine whose INFO names neither)
        svc, _, _ = service("x")
        with self.assertRaises(ValueError):
            svc.check_request({"messages": ASK, "logprobs": True})
        with self.assertRaises(ValueError):
            svc.check_request({"messages": ASK, "logit_bias": {"65": 5}})


WRITE = [{"type": "function", "function": {
    "name": "write_file", "description": "Write a file.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                   "required": ["path", "content"]}}}]
WRITE_ARGS = {"path": "src/main.py", "content": FILE}
# the call as the template writes it (it streams), and in a loose shape the parser reads only at its end (no "args"
# key: nothing of it can go out while it is written)
WRITTEN = call("write_file", WRITE_ARGS)
HELD = MODEL + "write_file" + CALL + '{"name":"write_file"' + json.dumps(WRITE_ARGS) + "}" + END


class Slow(Scripted):
    """Scripted, a token every `delay` seconds: an answer that takes a while to write."""
    delay = 0.001

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        for t in super().generate(ids, max_new, sampling, cancel, embeddings):
            time.sleep(self.delay)
            yield t


class WhileItIsWritten(unittest.TestCase):
    """What a client and the status see while a long tool call is written: the call as it comes (LiveCalls, through
    the server), a keep-alive every MAYA_STREAM_PING_S whenever tokens come and nothing of them does - a call held to
    its end, by its shape or by the loop guard - on every stream, and a phase that is true ("reading the prompt" used
    to stand there for the whole call)."""

    def setUp(self):
        self.env = unittest.mock.patch.dict(os.environ, {"MAYA_STREAM_PING_S": "0.05"})
        self.env.start()
        self.tok = InkTok()

    def tearDown(self):
        self.env.stop()

    def run_it(self, script, **sampling):
        """-> [(kind, x, the status' phase, its generated)] of Service.run, and the engine."""
        eng = Slow(self.tok, script)
        svc = Service(eng, self.tok, public())
        msgs, tools, kw = openai_to_messages({"messages": ASK, "tools": WRITE, "reasoning_effort": "none"})
        ids, thinking, max_new = svc.prepare(msgs, tools, kw, 4000)
        items = []
        for kind, x in svc.run(ids, thinking, tools, max_new, sampling, threading.Event()):
            items.append((kind, x, svc.status.get("phase"), svc.status.get("generated")))
        return items, svc

    def test_a_held_call_keeps_the_stream_alive_and_the_phase_true(self):
        items, _ = self.run_it(HELD)
        first = next(i for i, it in enumerate(items) if it[0] == "event")
        before = items[:first]
        self.assertGreaterEqual(len(before), 5, len(before))             # ~0.7 s of tokens, a ping every 0.05 s
        self.assertEqual({(kind, x) for kind, x, _, _ in before}, {("ping", None)})
        self.assertEqual({phase for _, _, phase, _ in before}, {"generating"})
        self.assertGreater(before[-1][3], before[0][3])                  # (tokens were coming all the while)
        self.assertEqual([x.call.arguments for kind, x, _, _ in items if kind == "event" and x.kind == "tool_call"],
                         [WRITE_ARGS])
        self.assertEqual(items[-1][0], "done")                           # (and no ping after the end)

    def test_the_progress_line_says_generating(self):
        import contextlib
        import io
        eng = Slow(self.tok, HELD)
        svc = Service(eng, self.tok, public())
        msgs, tools, kw = openai_to_messages({"messages": ASK, "tools": WRITE, "reasoning_effort": "none"})
        ids, thinking, max_new = svc.prepare(msgs, tools, kw, 4000)
        lines = io.StringIO()
        with contextlib.redirect_stdout(lines):
            for kind, x in svc.run(ids, thinking, tools, max_new, {}, threading.Event()):
                if kind == "ping":
                    svc._progress(0.0)
        said = [line for line in lines.getvalue().splitlines() if " of max " in line]
        self.assertGreaterEqual(len(said), 5)
        self.assertTrue(all(line.startswith("[maya] generating: ") for line in said), said[:3])

    def test_the_phase_from_the_first_token(self):
        from serve.frontend import Event, ToolCall
        svc, stat = Service(Scripted(self.tok, "x"), self.tok, public()), server.RequestStat()
        svc.status.update(busy=True, phase="reading the prompt", tail="", tool=None)
        svc._note(1, [], stat)                                           # a token, no event of it yet
        self.assertEqual(svc.status["phase"], "generating")
        svc._note(2, [Event("tool_start", call=ToolCall("write_file", {}))], stat)
        svc._note(3, [], stat)
        self.assertEqual((svc.status["phase"], svc.status["tool"]), ("writing a tool call: write_file", "write_file"))

    def test_a_call_that_streams_is_named_while_it_is_written(self):
        items, _ = self.run_it(WRITTEN)
        total = items[-1][1]["completion_tokens"]
        start = next(it for it in items if it[0] == "event")
        self.assertEqual((start[1].kind, start[1].call.name), ("tool_start", "write_file"))
        self.assertLess(start[3], 60)                                    # announced at its opening (of ~700 tokens)
        self.assertEqual(start[2], "writing a tool call: write_file")
        args = [it for it in items if it[0] == "event" and it[1].kind == "tool_args"]
        self.assertGreater(sum(1 for it in args if it[3] < total // 2), 100)   # its arguments, as they are written
        self.assertEqual(json.loads("".join(it[1].text for it in args)), WRITE_ARGS)
        self.assertNotIn("reading the prompt", {it[2] for it in items})
        self.assertNotIn("generating", {it[2] for it in items if it[3] > 60})

    def test_a_call_the_loop_guard_holds_back_stays_held(self):
        # calls the loop guard blocks in this answer (`_blocked`): every call's events wait for the answer's end
        # (run()'s `pending`) - nothing of this one goes out early, and the pings keep the stream alive
        items, _ = self.run_it(WRITTEN, _blocked={("other_tool", "{}"): (None, [], False)})
        total = items[-1][1]["completion_tokens"]
        events = [it for it in items if it[0] == "event"]
        self.assertEqual([it[1].kind for it in events][:2] + [events[-1][1].kind], ["tool_start", "tool_args", "tool_call"])
        self.assertEqual({it[3] for it in events}, {total - 1})          # all of them at the end (- the end token)
        self.assertEqual(events[-1][1].call.arguments, WRITE_ARGS)
        self.assertGreaterEqual(sum(1 for it in items if it[0] == "ping"), 5)

    # ------------------------------------------------------------------------------------------ the three streams
    def stream(self, script, path, body):
        """-> the SSE blocks of a streamed answer, as they were sent (a keep-alive is the block ": keep-alive")."""
        httpd = server.serve(Service(Slow(self.tok, script), self.tok, public()), port=0)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}",
                                         data=json.dumps({"model": "m", "stream": True, **body}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode()
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertTrue(raw.endswith("\n\n"), raw[-80:])
        return raw[:-2].split("\n\n")

    def data(self, blocks):
        """The blocks' JSON, in order: every block is a keep-alive, [DONE], or one event whose data parses - a
        keep-alive never stands inside one."""
        out = []
        for b in blocks:
            if b in (": keep-alive", "data: [DONE]"):
                continue
            lines = b.split("\n")
            self.assertLessEqual(len(lines), 2, b[:200])
            self.assertTrue(lines[-1].startswith("data: "), b[:200])
            if len(lines) == 2:
                self.assertTrue(lines[0].startswith("event: "), b[:200])
            out.append(json.loads(lines[-1][6:]))
        return out

    STREAMS = (
        ("/v1/chat/completions", {"messages": ASK, "tools": WRITE, "reasoning_effort": "none", "max_tokens": 4000}),
        ("/v1/messages", {"messages": ASK, "max_tokens": 4000, "tools": [
            {"name": "write_file", "input_schema": WRITE[0]["function"]["parameters"]}]}),
        ("/v1/responses", {"input": "Weather in Paris?", "max_output_tokens": 4000, "tools": [
            {"type": "function", "name": "write_file", "parameters": WRITE[0]["function"]["parameters"]}]}),
    )

    def arguments(self, path, events):
        """The call's arguments as the stream's events carry them, and how many events brought a piece of them."""
        if path == "/v1/chat/completions":
            pieces = [tc["function"]["arguments"] for e in events for tc in e["choices"][0]["delta"].get("tool_calls") or []]
            names = [tc["function"].get("name") for e in events for tc in e["choices"][0]["delta"].get("tool_calls") or []]
            self.assertEqual((names[0], set(names[1:]) - {None}), ("write_file", set()))   # the name once, first
            self.assertEqual(events[-1]["choices"][0]["finish_reason"], "tool_calls")
        elif path == "/v1/messages":
            pieces = [e["delta"]["partial_json"] for e in events if e.get("delta", {}).get("type") == "input_json_delta"]
            self.assertEqual([e["content_block"]["name"] for e in events if e["type"] == "content_block_start"
                              and e["content_block"]["type"] == "tool_use"], ["write_file"])
            self.assertEqual(events[-1]["type"], "message_stop")
        else:
            pieces = [e["delta"] for e in events if e["type"] == "response.function_call_arguments.delta"]
            self.assertEqual(events[-1]["type"], "response.completed")
            self.assertEqual(json.loads(events[-1]["response"]["output"][-1]["arguments"]), WRITE_ARGS)
        return json.loads("".join(pieces)), len(pieces)

    def test_keep_alives_on_every_stream_while_a_held_call_is_written(self):
        for path, body in self.STREAMS:
            with self.subTest(path=path):
                blocks = self.stream(HELD, path, body)
                alive = [i for i, b in enumerate(blocks) if b == ": keep-alive"]
                self.assertGreaterEqual(len(alive), 5, (path, blocks[:4]))
                events = self.data(blocks)
                got, pieces = self.arguments(path, events)
                self.assertEqual(got, WRITE_ARGS)
                last = max(i for i, b in enumerate(blocks) if b not in (": keep-alive", "data: [DONE]"))
                self.assertLess(alive[-1], last)                         # none after the stream's end

    def test_the_call_streams_on_every_stream(self):
        for path, body in self.STREAMS:
            with self.subTest(path=path):
                events = self.data(self.stream(WRITTEN, path, body))
                got, pieces = self.arguments(path, events)
                self.assertEqual(got, WRITE_ARGS)
                self.assertGreater(pieces, 200)                          # (the held call above: a handful, at its end)

    def test_not_streamed_the_same_call(self):
        httpd = server.serve(Service(Scripted(self.tok, WRITTEN), self.tok, public()), port=0)
        try:
            path, body = self.STREAMS[0]
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}",
                                         data=json.dumps({"model": "m", **body}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.loads(r.read())
        finally:
            httpd.shutdown()
            httpd.server_close()
        calls = out["choices"][0]["message"]["tool_calls"]
        self.assertEqual([(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls],
                         [("write_file", WRITE_ARGS)])
        self.assertEqual(out["choices"][0]["finish_reason"], "tool_calls")


class Defaults(unittest.TestCase):
    CFG = {"exe": "/opt/maya/build/maya-ik2", "args": ["--model", "/models/Inkling-2-Small-UD-Q3_K_XL-00001-of-00005.gguf",
                                                       "--max-context", "32768"], "model_name": "inkling-2-small"}

    def test_models_own(self):
        Q = server.maya_quants
        self.assertEqual(Q.family_of(self.CFG), "ik2")
        self.assertEqual(Q.model_defaults(self.CFG), ({"temperature": 1.0, "top_p": 1.0}, "auto"))

    def test_config_effort(self):
        cfg = server.checked_config_effort(dict(self.CFG), public(), True)
        self.assertEqual(cfg["reasoning_effort"], "auto")


if __name__ == "__main__":
    unittest.main()
