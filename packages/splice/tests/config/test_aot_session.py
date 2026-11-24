# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the AOT session configuration boundary without a native compiler or GPU."""

import importlib.util
import inspect
import os
import pickle
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_PATH = Path(__file__).resolve().parents[2] / "python" / "splice" / "aot.py"
_SPEC = importlib.util.spec_from_file_location("_splice_aot_session_test", _PATH)
aot = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(aot)


class TestAotSession(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.active = False

        @contextmanager
        def loom_aot(options, *, cache, workload_identity):
            search_budget_ms = options.search_budget_ms
            shortlist_budget_ms = options.shortlist_budget_ms
            self.calls.append(
                {
                    "cache": cache,
                    "workload_identity": workload_identity,
                    "search_budget_ms": search_budget_ms,
                    "shortlist_budget_ms": shortlist_budget_ms,
                }
            )
            self.active = True
            try:
                yield
            finally:
                self.active = False

        loom = SimpleNamespace(aot=loom_aot, Compile=SimpleNamespace)
        self.enterContext(patch.dict(sys.modules, {"loom": loom}))
        self.output = self.enterContext(redirect_stdout(StringIO()))

    def test_signature_defaults(self):
        parameters = inspect.signature(aot.aot_session).parameters
        self.assertEqual(aot.DEFAULT_SEARCH_BUDGET_MS, 250)
        self.assertEqual(aot.DEFAULT_SHORTLIST_BUDGET_MS, 2000)
        self.assertEqual(parameters["search_budget_ms"].default, aot.DEFAULT_SEARCH_BUDGET_MS)
        self.assertEqual(parameters["shortlist_budget_ms"].default, aot.DEFAULT_SHORTLIST_BUDGET_MS)

    def test_defaults_and_context_lifetime(self):
        with aot.aot_session():
            self.assertTrue(self.active)
        self.assertFalse(self.active)
        self.assertEqual(
            self.calls,
            [
                {
                    "cache": None,
                    "workload_identity": None,
                    "search_budget_ms": 250,
                    "shortlist_budget_ms": 2000,
                }
            ],
        )
        self.assertIn("search=250ms shortlist=2000ms", self.output.getvalue())

    def test_explicit_budgets(self):
        with aot.aot_session(search_budget_ms=12.5, shortlist_budget_ms=70):
            pass
        self.assertEqual(self.calls[0]["search_budget_ms"], 12.5)
        self.assertEqual(self.calls[0]["shortlist_budget_ms"], 70)

    def test_none_is_unlimited_independently(self):
        for search, shortlist in ((None, None), (None, 70), (12.5, None)):
            with self.subTest(search=search, shortlist=shortlist):
                with aot.aot_session(search_budget_ms=search, shortlist_budget_ms=shortlist):
                    pass
                self.assertEqual(self.calls[-1]["search_budget_ms"], search)
                self.assertEqual(self.calls[-1]["shortlist_budget_ms"], shortlist)
        self.assertIn("search=unlimited shortlist=unlimited", self.output.getvalue())

    def test_budget_environment_is_ignored(self):
        for value in ("17", "not-a-number"):
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ, {"SPLICE_AOT_BUDGET_MS": value, "SPLICE_TUNE_BUDGET_MS": value}
                ),
            ):
                with aot.aot_session():
                    pass
                self.assertEqual(self.calls[-1]["search_budget_ms"], 250)
                self.assertEqual(self.calls[-1]["shortlist_budget_ms"], 2000)
                with aot.aot_session(search_budget_ms=None, shortlist_budget_ms=None):
                    pass
                self.assertIsNone(self.calls[-1]["search_budget_ms"])
                self.assertIsNone(self.calls[-1]["shortlist_budget_ms"])

    def test_old_keywords_are_rejected(self):
        for name in ("search_ms", "tune_ms", "tune_budget_ms"):
            with (
                self.subTest(name=name),
                self.assertRaises(TypeError),
                aot.aot_session(**{name: 10}),
            ):
                pass
        self.assertEqual(self.calls, [])

    def test_quiet_unlimited_session(self):
        with aot.aot_session(search_budget_ms=None, shortlist_budget_ms=None, verbose=False):
            pass
        self.assertEqual(self.output.getvalue(), "")

    def test_cache_round_trip_with_unlimited_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            identity_file = Path(directory) / "arch.yaml"
            identity_file.write_text("model: example\n")
            cache_path = str(Path(directory) / "aot.pkl")
            for iteration in range(2):
                with aot.aot_session(
                    identity_files=[identity_file],
                    extra="shape",
                    cache_path=cache_path,
                    search_budget_ms=None,
                    shortlist_budget_ms=None,
                ):
                    self.assertEqual(self.calls[-1]["cache"], {} if iteration == 0 else {"op": 1})
                    self.calls[-1]["cache"]["op"] = 1
                self.assertEqual(
                    self.calls[-1]["workload_identity"],
                    aot.workload_identity([identity_file], "shape"),
                )
            with open(cache_path, "rb") as handle:
                self.assertEqual(pickle.load(handle), {"op": 1})
            self.assertFalse(Path(cache_path + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
