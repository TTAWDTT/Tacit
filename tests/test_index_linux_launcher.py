"""Offline launcher tests: stdlib only, no tasks, model, GPU or inference."""
from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("index_linux_launcher", ROOT / "experiments/index_v0_3/run_local_capability.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)
FIXTURE = ROOT / "tests/fixtures/index_launcher_fake.py"
EVIDENCE = []


def row(**changes):
    return dict(host_cpu_percent=1, gpu_uuid="GPU-offline-fixture", gpu_util_percent=1,
                gpu_memory_used_mib=100, gpu_memory_total_mib=8000,
                available_system_memory_mib=8000, server_cpu_cores=0,
                server_working_set_mib=0) | changes


class Resources:
    cpus = [1, 1]
    idle = row()
    active = row()

    def __init__(self):
        self.values = iter(self.cpus)

    def cpu(self):
        return next(self.values)

    def sample(self, server=None):
        return self.idle if server is None else self.active


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name) / "run"

    def patches(self, stack, scenario="normal", resources=Resources):
        stack.enter_context(patch.object(launcher, "LinuxResources", resources))
        for key, value in dict(STARTUP_SECONDS=.5, RUN_SECONDS=.5, SAMPLE_SECONDS=.03,
                               POLL_SECONDS=.005, PROBE_SECONDS=.02, CLEANUP_SECONDS=.05).items():
            stack.enter_context(patch.object(launcher, key, value))
        real_sleep = time.sleep
        stack.enter_context(patch.object(launcher.time, "sleep", lambda n: real_sleep(n) if n < 1 else None))
        # Unique reserved loopback socket: never touches production port 8001.
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        stack.callback(sock.close)
        stack.enter_context(patch.object(launcher, "PORT", sock.getsockname()[1]))
        stack.enter_context(patch.object(launcher, "reserve_port", return_value=sock))
        original = launcher.Children.start

        def start(children, name, args, pass_fds=()):
            # Assert production command contract before substituting the fixture.
            if name == "server":
                self.assertEqual(args[:3], ["-m", "uvicorn", "local_chat_server:app"])
                self.assertEqual(args[args.index("--workers") + 1], "1")
                self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
                self.assertEqual(len(pass_fds), 1)
            else:
                self.assertEqual(args[0], str(launcher.RUNNER))
                self.assertEqual("--artifacts-only" in args, name == "preflight")
            return original(children, name, [str(FIXTURE), name, scenario, str(self.run_dir),
                                            *map(str, pass_fds)], pass_fds)
        stack.enter_context(patch.object(launcher.Children, "start", start))

    def run_case(self, scenario, resources=Resources, prepare=False, exception=None):
        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            self.patches(stack, scenario, resources)
            if exception:
                stack.enter_context(patch.object(launcher, "ready", side_effect=exception))
            # An unrelated process must survive every cleanup path.
            outsider = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            try:
                code = launcher.launch(self.run_dir, prepare)
                self.assertIsNone(outsider.poll())
            finally:
                outsider.terminate()
                outsider.wait()
        records = [json.loads(s) for s in (self.run_dir / "launcher.jsonl").read_text().splitlines()]
        for event in records:
            if event["event"] == "child_started":
                self.assertFalse(Path(f"/proc/{event['pid']}").exists(), event)
        descendant = self.run_dir / "descendant.pid"
        if descendant.exists():
            stat = Path(f"/proc/{descendant.read_text()}/stat")
            # Orphans can briefly remain zombies until container init reaps them.
            deadline = time.monotonic() + 1
            while stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] != "Z":
                self.assertLess(time.monotonic(), deadline, "owned descendant survived cleanup")
                time.sleep(.005)
        EVIDENCE.append({"scenario": scenario, "offline": True, "result": records[-1],
                         "children_started": [r["role"] for r in records if r["event"] == "child_started"]})
        return code, records[-1]

    def test_independent_idle_rejections(self):
        cases = [("cpu_mean", [20, 20], row(host_cpu_percent=20)),
                 ("cpu_peak", [30, 1], row()),
                 ("gpu_utilization", [1, 1], row(gpu_util_percent=25)),
                 ("gpu_memory", [1, 1], row(gpu_memory_used_mib=1800)),
                 ("system_memory", [1, 1], row(available_system_memory_mib=5999))]
        for reason, cpus, idle in cases:
            with self.subTest(reason=reason):
                self.run_dir = Path(self.temp.name) / reason
                resources = type("RejectedResources", (Resources,), {"cpus": cpus, "idle": idle})
                with patch.object(launcher.Children, "start", side_effect=AssertionError("must not spawn")):
                    # run_case normally replaces start; rejection is additionally
                    # proven by no fixture files and no child_started events.
                    code, result = self.run_case(reason, resources)
                self.assertEqual(code, 1)
                self.assertEqual(result["reason"], "idle_resource_rejection:" + reason)
                self.assertEqual(EVIDENCE[-1]["children_started"], [])
                self.assertFalse(list(self.run_dir.glob("*.pid")))

    def test_port_rejection_no_resources_or_children(self):
        with socket.socket() as sock, patch.object(launcher, "LinuxResources") as resources:
            sock.bind(("127.0.0.1", 0))
            sock.listen()
            with patch.object(launcher, "PORT", sock.getsockname()[1]), redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.launch(self.run_dir), 1)
            resources.assert_not_called()
            result = json.loads((self.run_dir / "launcher.jsonl").read_text().splitlines()[-1])
            self.assertEqual(result["reason"], "port_in_use")
            EVIDENCE.append({"scenario": "port_in_use", "offline": True, "result": result, "children_started": []})
            self.assertGreater(sock.fileno(), -1)

    def test_frozen_contract(self):
        frozen = json.loads((ROOT / "experiments/index_v0_3/preregistration.json").read_text())
        self.assertEqual(launcher.STARTUP_SECONDS, frozen["maximum_work"]["server_startup_seconds"])
        self.assertEqual(launcher.RUN_SECONDS, frozen["maximum_work"]["overall_run_seconds"])
        self.assertEqual(launcher.SAMPLE_SECONDS, 10)
        self.assertEqual(launcher.PORT, 8001)
        # The unchanged runner owns 60s requests, 24 tokens, no retries and the
        # maximum 4 + 4 * 2 calls. Detect accidental changes to that dependency.
        import hashlib
        self.assertEqual(hashlib.sha256(launcher.RUNNER.read_bytes()).hexdigest(),
                         "5594b61b4ba46a2035efbb378d4bd2c98514e052285c6505235fe1e21e87c9dd")

    def test_ipv6_port_rejection(self):
        if not socket.has_ipv6:
            self.skipTest("IPv6 unavailable")
        with socket.socket(socket.AF_INET6) as sock:
            try:
                sock.bind(("::1", 0))
            except OSError:
                self.skipTest("IPv6 loopback unavailable")
            sock.listen()
            with patch.object(launcher, "PORT", sock.getsockname()[1]):
                with self.assertRaisesRegex(launcher.Stopped, "port_in_use"):
                    launcher.reserve_port()

    def test_boundaries_and_counter_reset(self):
        self.assertEqual(launcher.idle_reasons([19.9] * 3, row(gpu_util_percent=24.9,
                         gpu_memory_used_mib=1799, available_system_memory_mib=6000)), [])
        for field, value in [("host_cpu_percent", 45), ("gpu_util_percent", 85)]:
            gate = launcher.StopGate()
            gate.check(row(**{field: value}))
            gate.check(row())
            gate.check(row(**{field: value}))
            with self.assertRaises(launcher.Stopped):
                gate.check(row(**{field: value}))
        launcher.StopGate().check(row(available_system_memory_mib=4000))
        with self.assertRaises(launcher.Stopped):
            launcher.StopGate().check(row(available_system_memory_mib=3999))

    def test_lifecycle_matrix(self):
        for scenario, reason in [("normal", None), ("descendant", None),
                ("hash_failure", "artifact_preflight_failed"),
                ("startup_failure", "server_start_failed"), ("startup_timeout", "startup_timeout"),
                ("run_timeout", "run_timeout"), ("runner_failure", "runner_failed"),
                ("server_exit", "server_exited_during_run"), ("bad_summary", "JSONDecodeError"),
                ("over_limit", "invalid_request_count")]:
            with self.subTest(scenario=scenario):
                self.run_dir = Path(self.temp.name) / scenario
                code, result = self.run_case(scenario)
                self.assertEqual(code, 0 if reason is None else 1)
                if reason:
                    self.assertEqual(result["reason"], reason)
                else:
                    self.assertEqual(result["status"], "completed")
                if scenario == "hash_failure":
                    self.assertEqual(EVIDENCE[-1]["children_started"], ["preflight"])

    def test_prepare_and_environment(self):
        code, result = self.run_case("prepare", prepare=True)
        self.assertEqual((code, result["status"]), (0, "prepared"))
        self.assertEqual(EVIDENCE[-1]["children_started"], ["preflight"])
        env = json.loads((self.run_dir / "preflight.env.json").read_text())
        for key in ("TLU_TORCH_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HUB_OFFLINE"):
            self.assertEqual(env[key], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-offline-fixture")
        self.assertGreaterEqual(env["nice"], 5)
        self.assertEqual(env["blocked_signals"], [])

    def test_runtime_resource_stops(self):
        for field, value in [("host_cpu_percent", 45), ("gpu_util_percent", 85),
                             ("available_system_memory_mib", 3999)]:
            self.run_dir = Path(self.temp.name) / field
            resources = type("BusyResources", (Resources,), {"active": row(**{field: value})})
            code, result = self.run_case("resource_stop", resources)
            self.assertEqual((code, result["reason"]), (1, "automatic_resource_stop"))

    def test_missing_idle_telemetry(self):
        class Unavailable(Resources):
            def sample(self, server=None):
                raise FileNotFoundError("offline missing nvidia-smi")
        code, result = self.run_case("missing_telemetry", Unavailable)
        self.assertEqual((code, result["reason"]), (1, "FileNotFoundError"))
        self.assertEqual(EVIDENCE[-1]["children_started"], [])

    def test_startup_resource_stop(self):
        resources = type("BusyResources", (Resources,), {"active": row(host_cpu_percent=45)})
        with patch.object(launcher, "ready", return_value=False):
            code, result = self.run_case("startup_resource_stop", resources)
        self.assertEqual((code, result["reason"]), (1, "automatic_resource_stop"))
        self.assertEqual(EVIDENCE[-1]["children_started"], ["preflight", "server"])

    def test_runtime_telemetry_error(self):
        class Unavailable(Resources):
            def sample(self, server=None):
                if server is not None:
                    raise RuntimeError("offline telemetry failure")
                return self.idle
        code, result = self.run_case("telemetry_error", Unavailable)
        self.assertEqual((code, result["reason"]), (1, "RuntimeError"))

    def test_linux_telemetry_parsing(self):
        contents = {"/proc/stat": "cpu  100 0 100 800 0 0 0 0 20 0\n",
                    "/proc/meminfo": "MemAvailable: 8192000 kB\n"}
        with patch.object(Path, "read_text", lambda p: contents[str(p)]), patch.object(
                launcher.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 0, stdout="GPU-offline, 24, 1799, 8000\n")):
            resources = launcher.LinuxResources()
            contents["/proc/stat"] = "cpu  110 0 100 890 0 0 0 0 20 0\n"
            sample = resources.sample()
            self.assertEqual(sample["host_cpu_percent"], 10)
            self.assertEqual(sample["available_system_memory_mib"], 8000)
            contents["/proc/stat"] = "cpu  110 0 100 990 0 0 0 0 20 0\n"
            with patch.object(launcher.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 0, stdout="GPU-offline, N/A, 1799, 8000\n")):
                with self.assertRaises(ValueError):
                    resources.sample()

    def test_exception_cleanup(self):
        code, result = self.run_case("exception", exception=RuntimeError("injected offline exception"))
        self.assertEqual((code, result["reason"]), (1, "RuntimeError"))

    def test_sigterm(self):
        # Deliver an actual signal only after the fake runner is registered.
        def terminate():
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if (self.run_dir / "runner.pid").exists():
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
                time.sleep(.005)
        thread = threading.Thread(target=terminate)
        thread.start()
        try:
            code, result = self.run_case("sigterm")
        finally:
            thread.join()
        self.assertEqual((code, result["reason"]), (143, "signal_SIGTERM"))


if __name__ == "__main__":
    result = unittest.main(exit=False)
    destination = os.environ.get("TACIT_LAUNCHER_TEST_REPORT")
    if destination:
        Path(destination).write_text(json.dumps({"kind": "offline_launcher_validation",
            "scientific_result": False, "tests_run": result.result.testsRun,
            "successful": result.result.wasSuccessful(), "cases": EVIDENCE}, indent=2) + "\n")
    sys.exit(0 if result.result.wasSuccessful() else 1)
