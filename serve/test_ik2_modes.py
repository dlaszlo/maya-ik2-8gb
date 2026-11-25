"""serve/ik2_modes.py: the marker ids a drafting maya-ik2 gets from the tokenizer."""
import json
import os
import tempfile
import unittest

from serve import ik2_modes


def vocab_dir(vocab) -> str:
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    return d


INKLING = {"a": 1, "```": 7, "Ġ```": 9, "<|message_model|>": 100, "<|content_text|>": 101,
           "<|content_thinking|>": 103, "<|content_invoke_tool_json|>": 110, "<|end_message|>": 120}


class ModeEnv(unittest.TestCase):
    def test_inkling_markers(self):
        env = ik2_modes.mode_env(vocab_dir(INKLING))
        self.assertEqual(env, {"IK2_MODE_START": "100", "IK2_MODE_MARKS": "101,103,110", "IK2_MODE_FENCE": "7,9",
                               "IK2_MODE_DRAFT": "110"})

    def test_no_markers_no_modes(self):
        self.assertEqual(ik2_modes.mode_env(vocab_dir({"a": 1, "```": 2})), {})
        self.assertEqual(ik2_modes.mode_env("/nonexistent/tokenizer"), {})

    def test_export_only_when_drafting(self):
        tok = vocab_dir(INKLING)
        env = {}
        ik2_modes.export(env, {"args": ["--model", "m.gguf"], "tokenizer": tok})
        self.assertEqual(env, {})
        ik2_modes.export(env, {"args": ["--model", "m.gguf", "--draft", "mtp.safetensors"], "tokenizer": tok})
        self.assertEqual(env["IK2_MODE_MARKS"], "101,103,110")

    def test_config_env_wins(self):
        tok = vocab_dir(INKLING)
        env = {"IK2_MODE_MARKS": ""}          # (the config turned the modes off)
        ik2_modes.export(env, {"args": ["--draft", "x"], "tokenizer": tok})
        self.assertEqual(env["IK2_MODE_MARKS"], "")

    def test_relative_tokenizer_from_cwd(self):
        tok = vocab_dir(INKLING)
        env = {}
        ik2_modes.export(env, {"args": ["--draft", "x"], "tokenizer": os.path.basename(tok),
                               "cwd": os.path.dirname(tok)})
        self.assertEqual(env["IK2_MODE_START"], "100")


if __name__ == "__main__":
    unittest.main()
