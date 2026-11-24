import unittest
from serve.server import StrataEngine


class InklingSamplingTransport(unittest.TestCase):
    def keys(self, kind, **sampling):
        engine = object.__new__(StrataEngine)
        engine.info = {"engine_kind": kind}
        return engine.request_sampling_keys(sampling).split()

    def test_ik2_zero_is_preserved(self):
        self.assertIn("top_k=0", self.keys("ik2", temperature=1.0, top_k=0))

    def test_ik2_wide_top_k_is_preserved(self):
        for value in (65, 100, 1000, 2000):
            self.assertIn(f"top_k={value}", self.keys("ik2", temperature=1.0, top_k=value))

    def test_ik2_absent_key_keeps_engine_default(self):
        self.assertEqual(self.keys("ik2", temperature=1.0), ["temperature=1.0"])

    def test_other_engines_keep_the_existing_limit(self):
        for kind in ("glm", "hy4", "", None):
            for value in (0, 100, 1000):
                self.assertIn("top_k=64", self.keys(kind, temperature=1.0, top_k=value))

    def test_greedy_stays_greedy(self):
        self.assertEqual(self.keys("ik2", temperature=0, top_k=1000), ["temperature=0"])

    def test_invalid_top_k_is_not_forwarded(self):
        for value in (-1, True, 2.5, "20"):
            self.assertFalse(any(k.startswith("top_k=") for k in
                                 self.keys("ik2", temperature=1.0, top_k=value)))
