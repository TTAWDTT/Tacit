from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.emergent_ood_v0_4.response_diagnostics import (
    ResponseConfiguration, diagnose_response, main,
)
from experiments.emergent_ood_v0_4.runner import _parse_choice


FIXTURE = Path(__file__).parent / "fixtures/response_contract/offline.json"


class ResponseDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.record = json.loads(FIXTURE.read_text())
        self.config = ResponseConfiguration(**self.record["configuration"])
        self.ids = self.record["candidate_ids"]

    def diagnose(self, payload):
        return diagnose_response(payload, candidate_ids=self.ids, configuration=self.config)

    def payload(self, content, finish="stop"):
        return {"model": "synthetic-model", "choices": [
            {"message": {"content": content}, "finish_reason": finish}]}

    def test_shapes_use_unmodified_frozen_parser(self):
        cases = [
            ("C2", "stop", "candidate_id", True, False),
            (" \nC2\t", "stop", "candidate_id", True, False),
            ("", "stop", "empty_content", False, False),
            (" \n", "length", "empty_content", False, True),
            ("<think>\n</think>\nC2", "stop", "think_wrapper", False, False),
            ("<think>synthetic unfinished", "length", "think_wrapper", False, True),
            ("C", "length", "unrecognized_text", False, True),
            ("C2", "length", "candidate_id", True, True),
            ("Answer: C2", "stop", "extra_text", False, False),
            ("C2 because...", "stop", "extra_text", False, False),
            ("C20", "stop", "unrecognized_text", False, False),
            ("C1", "stop", "candidate_id", True, False),  # legal != correct
        ]
        for content, finish, shape, valid, truncated in cases:
            with self.subTest(content=content):
                payload = self.payload(content, finish)
                before = copy.deepcopy(payload)
                result = self.diagnose(payload)
                self.assertEqual(result["content_shape"], shape)
                self.assertEqual(result["strict_format_valid"], valid)
                self.assertEqual(result["strict_format_valid"], _parse_choice(content, self.ids)[1])
                self.assertEqual(result["truncated"], truncated)
                self.assertEqual(payload, before)
                self.assertNotIn("joint_success", result)

    def test_reasoning_presence_and_type_only(self):
        for field in ("reasoning_content", "reasoning"):
            for value, kind in [(None, "null"), ("", "string"), ("PRIVATE_SENTINEL", "string"),
                                ({"PRIVATE_SENTINEL": "secret"}, "object"),
                                (["PRIVATE_SENTINEL"], "array"), (True, "boolean"), (42, "number")]:
                with self.subTest(field=field, kind=kind):
                    payload = self.payload("")
                    payload["choices"][0]["message"][field] = value
                    result = self.diagnose(payload)
                    self.assertEqual(result["reasoning_fields"][field], {"present": True, "type": kind})
                    self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))
                    self.assertFalse(result["strict_format_valid"])
        absent = self.diagnose(self.payload("C2"))["reasoning_fields"]
        self.assertEqual(absent["reasoning_content"], {"present": False, "type": "missing"})

    def test_malformed_envelopes_and_content(self):
        for payload in (None, [], {}, {"choices": []}, {"choices": [None]},
                        {"choices": [{"message": []}]}):
            self.assertEqual(self.diagnose(payload)["content_shape"], "invalid_envelope")
        missing = {"choices": [{"message": {}}]}
        self.assertEqual(self.diagnose(missing)["content_shape"], "missing_content")
        for content in (None, [], {}, 7, True):
            result = self.diagnose(self.payload(content))
            self.assertEqual(result["content_shape"], "non_text_content")
            self.assertFalse(result["strict_format_valid"])

    def test_no_raw_content_or_unknown_provider_metadata_is_echoed(self):
        payload = self.payload("<think>PRIVATE_SENTINEL</think>C2", "PRIVATE_SENTINEL")
        payload["model"] = "PRIVATE_SENTINEL"
        result = self.diagnose(payload)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))
        self.assertFalse(result["response_model_matches_requested"])
        self.assertEqual(result["finish_reason"], "other")
        self.assertEqual(result["configuration"], self.record["configuration"])

    def test_cli_is_offline_and_metadata_only(self):
        with patch("socket.socket", side_effect=AssertionError("network forbidden")):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main([str(FIXTURE)]), 0)
        self.assertEqual(json.loads(output.getvalue())["content_shape"], "candidate_id")

    def test_cli_suppresses_input_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"PRIVATE_SENTINEL":')
            error = io.StringIO()
            with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                main([str(path)])
            self.assertEqual(raised.exception.code, 2)
            self.assertNotIn("PRIVATE_SENTINEL", error.getvalue())

    def test_invalid_configuration_and_candidates_rejected(self):
        config = dict(self.record["configuration"], template_id="")
        with self.assertRaises(ValueError):
            ResponseConfiguration(**config)
        for ids in ([], "C2", ["C2", "C2"], [None]):
            with self.assertRaises(ValueError):
                diagnose_response({}, candidate_ids=ids, configuration=self.config)


if __name__ == "__main__":
    unittest.main()
