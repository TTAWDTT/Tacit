from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "emergent_ood_v0_3" / "runner.py"
SPEC = importlib.util.spec_from_file_location("emergent_ood_runner_resource_gate", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


class EmergentOODResourceGateTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
        self.report = {
            "schema": "tlu.local_resource_preflight.v1",
            "status": "eligible",
            "sampled_at_utc": (self.now - timedelta(seconds=30)).isoformat(),
            "machine_name": "test-host",
            "requested_ports": [8000, 8001],
            "listening_requested_ports": [],
            "limits": dict(module.RESOURCE_GATE_LIMITS),
            "observed": {
                "host_cpu_samples_percent": [5.0, 6.0, 7.0],
                "host_cpu_mean_percent": 6.0,
                "host_cpu_max_percent": 7.0,
                "gpu_utilization_percent": 10,
                "gpu_memory_used_mib": 1200,
                "gpu_memory_total_mib": 8192,
                "free_system_memory_mib": 8192,
            },
            "model_artifact_hashed": False,
            "model_artifact_read": False,
            "model_loaded": False,
            "service_started": False,
            "inference_requests": 0,
        }
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "preflight.json"

    def _write(self):
        self.path.write_text(json.dumps(self.report), encoding="utf-8")

    def _validate(self):
        module.validate_resource_preflight(
            self.path,
            required_ports={8000, 8001},
            machine_name="test-host",
            now=self.now,
        )

    def test_recent_passing_report_for_this_host_and_ports_is_accepted(self):
        self._write()
        self._validate()

    def test_missing_report_and_stale_report_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires --resource-preflight"):
            module.validate_resource_preflight(
                None, required_ports={8000}, machine_name="test-host", now=self.now,
            )
        self.report["sampled_at_utc"] = (self.now - timedelta(seconds=301)).isoformat()
        self._write()
        with self.assertRaisesRegex(ValueError, "five minutes"):
            self._validate()

    def test_wrong_host_or_unchecked_endpoint_port_is_rejected(self):
        self._write()
        with self.assertRaisesRegex(ValueError, "different machine"):
            module.validate_resource_preflight(
                self.path, required_ports={8000}, machine_name="other-host", now=self.now,
            )
        with self.assertRaisesRegex(ValueError, "every configured endpoint port"):
            module.validate_resource_preflight(
                self.path, required_ports={8000, 9000}, machine_name="test-host", now=self.now,
            )

    def test_rejected_or_relaxed_or_forged_measurements_are_rejected(self):
        self.report["status"] = "rejected"
        self._write()
        with self.assertRaisesRegex(ValueError, "did not pass"):
            self._validate()

        self.report["status"] = "eligible"
        self.report["limits"]["host_cpu_mean_below_percent"] = 40
        self._write()
        with self.assertRaisesRegex(ValueError, "limits do not match"):
            self._validate()

        self.report["limits"] = dict(module.RESOURCE_GATE_LIMITS)
        self.report["observed"]["gpu_utilization_percent"] = 25
        self._write()
        with self.assertRaisesRegex(ValueError, "GPU measurements exceed"):
            self._validate()

    def test_busy_port_or_preflight_after_model_start_is_rejected(self):
        self.report["listening_requested_ports"] = [8000]
        self._write()
        with self.assertRaisesRegex(ValueError, "already in use"):
            self._validate()

        self.report["listening_requested_ports"] = []
        self.report["model_loaded"] = True
        self._write()
        with self.assertRaisesRegex(ValueError, "precede model/service startup"):
            self._validate()

    def test_execute_cli_rejects_missing_preflight_before_output_or_endpoint_client(self):
        output = Path(self.temp_dir.name) / "must-not-exist.jsonl"
        argv = [
            "runner.py", "--execute", "--conditions", "full_information",
            "--receiver-model", "fake-receiver", "--receiver-tokenizer-id", "fake-tokenizer",
            "--output", str(output),
        ]
        with patch.object(sys, "argv", argv), patch.object(
            module, "_NamedClient", side_effect=AssertionError("endpoint client must not be constructed")
        ), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            module.main()
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
