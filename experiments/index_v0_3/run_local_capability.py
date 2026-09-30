"""Linux-only supervisor for the frozen INDEX_m v0.3 pilot (stdlib only)."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments/index_v0_3/run_capability_pilot.py"
MODEL = "Qwen3-1.7B"
PORT = 8001
STARTUP_SECONDS = 120
RUN_SECONDS = 300
SAMPLE_SECONDS = 10
POLL_SECONDS = 0.2
PROBE_SECONDS = 4
CLEANUP_SECONDS = 2


class Stopped(Exception):
    def __init__(self, reason, code=1):
        super().__init__(reason)
        self.code = code


def write_row(path, **row):
    row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **row}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def cpu_ticks():
    # guest/guest_nice are already included in user/nice; do not double count.
    values = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
    return sum(values), values[3] + values[4]


class LinuxResources:
    def __init__(self):
        self.previous = cpu_ticks()
        self.server_previous = (time.monotonic(), 0.0)

    def cpu(self):
        current = cpu_ticks()
        total = current[0] - self.previous[0]
        idle = current[1] - self.previous[1]
        self.previous = current
        if total <= 0 or not 0 <= idle <= total:
            raise Stopped("cpu_telemetry_unavailable")
        return 100 * (total - idle) / total

    def sample(self, server=None):
        cpu = self.cpu()
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=2,
        )
        fields = result.stdout.splitlines()[0].split(",")
        gpu_id = fields[0].strip()
        gpu, used, total = [float(x.strip()) for x in fields[1:]]
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        available = int(memory["MemAvailable"].split()[0]) / 1024
        if (not gpu_id.startswith("GPU-") or not all(math.isfinite(v) for v in (gpu, used, total))
                or not 0 <= gpu <= 100 or not 0 <= used <= total or total <= 0 or available < 0):
            raise Stopped("invalid_resource_telemetry")
        cores = rss = 0.0
        if server is not None:
            # Fields after comm (which can contain spaces): state is field 3.
            stat = Path(f"/proc/{server.pid}/stat").read_text().rsplit(")", 1)[1].split()
            cpu_seconds = (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK")
            now = time.monotonic()
            cores = (cpu_seconds - self.server_previous[1]) / (now - self.server_previous[0])
            rss = int(stat[21]) * os.sysconf("SC_PAGE_SIZE") / 1024**2
            self.server_previous = now, cpu_seconds
        return dict(host_cpu_percent=cpu, gpu_uuid=gpu_id, gpu_util_percent=gpu,
                    gpu_memory_used_mib=used, gpu_memory_total_mib=total,
                    available_system_memory_mib=available, server_cpu_cores=cores,
                    server_working_set_mib=rss)


def idle_reasons(cpu, row):
    reasons = []
    if sum(cpu) / 3 >= 20:
        reasons.append("cpu_mean")
    if max(cpu) >= 30:
        reasons.append("cpu_peak")
    if row["gpu_util_percent"] >= 25:
        reasons.append("gpu_utilization")
    if row["gpu_memory_used_mib"] >= 1800:
        reasons.append("gpu_memory")
    if row["available_system_memory_mib"] < 6000:
        reasons.append("system_memory")
    return reasons


class StopGate:
    def __init__(self):
        self.cpu = self.gpu = 0

    def check(self, row):
        self.cpu = self.cpu + 1 if row["host_cpu_percent"] >= 45 else 0
        self.gpu = self.gpu + 1 if row["gpu_util_percent"] >= 85 else 0
        if self.cpu >= 2 or self.gpu >= 2 or row["available_system_memory_mib"] < 4000:
            raise Stopped("automatic_resource_stop")


def reserve_port():
    # Check all local addresses, including IPv6, without touching their owners.
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        path = Path(table)
        if not path.exists() and table.endswith("6"):
            continue
        for line in path.read_text().splitlines()[1:]:
            fields = line.split()
            if int(fields[1].split(":")[1], 16) == PORT and fields[3] == "0A":
                raise Stopped("port_in_use")
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", PORT))
    except OSError as exc:
        sock.close()
        raise Stopped("port_in_use") from exc
    return sock


def child_env(run_dir, gpu_uuid):
    return {**os.environ, "TLU_MODEL_PATH": str(ROOT / ".cache/models" / MODEL),
            "TLU_MODEL_NAME": MODEL, "TLU_TORCH_THREADS": "1",
            "TLU_USAGE_LOG": str(run_dir / "usage.jsonl"),
            "CUDA_CACHE_PATH": str(ROOT / ".cache/cuda"),
            "CUDA_VISIBLE_DEVICES": gpu_uuid, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
            "OPENAI_API_KEY": "local-experiment", "PYTHONIOENCODING": "utf-8",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}


class Children:
    def __init__(self, stack, run_dir, env):
        self.stack, self.run_dir, self.env = stack, run_dir, env
        self.processes = []

    def start(self, name, args, pass_fds=()):
        stdout = self.stack.enter_context((self.run_dir / f"{name}.stdout.log").open("w"))
        stderr = self.stack.enter_context((self.run_dir / f"{name}.stderr.log").open("w"))
        # Priority is reduced before importing any service/model code.
        command = [sys.executable, "-c",
                   "import os,sys,signal; signal.pthread_sigmask(signal.SIG_SETMASK, []); "
                   "os.nice(5); os.execv(sys.executable,[sys.executable,*sys.argv[1:]])",
                   *args]
        # A pending SIGTERM must not leave a spawned but unregistered child.
        old = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=self.env, stdout=stdout,
                                       stderr=stderr, start_new_session=True, pass_fds=pass_fds)
            self.processes.append(process)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)
        write_row(self.run_dir / "launcher.jsonl", event="child_started", role=name, pid=process.pid)
        return process

    def close(self):
        # Groups are created only by start_new_session here. Never discover or
        # kill by port, executable name, parent session, or GPU ownership.
        for process in reversed(self.processes):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + CLEANUP_SECONDS
        for process in reversed(self.processes):
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        # Also remove descendants if the group leader exited first.
        for process in reversed(self.processes):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def ready(timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{PORT}/v1/models", timeout=timeout) as response:
            value = json.load(response)
        return MODEL in {item.get("id") for item in value.get("data", [])}
    except (OSError, ValueError, AttributeError, TypeError):
        return False


def supervise(children, sock, resources, run_dir, runner_args):
    sock.listen(128)
    resources.server_previous = (time.monotonic(), 0.0)
    server = children.start("server", ["-m", "uvicorn", "local_chat_server:app", "--app-dir",
        "research", "--host", "127.0.0.1", "--port", str(PORT), "--fd", str(sock.fileno()),
        "--workers", "1", "--log-level", "warning"], pass_fds=(sock.fileno(),))
    gate = StopGate()
    deadline = time.monotonic() + STARTUP_SECONDS
    next_sample = time.monotonic() + SAMPLE_SECONDS
    next_probe = time.monotonic() + PROBE_SECONDS

    def sample_if_due():
        nonlocal next_sample
        if time.monotonic() >= next_sample:
            row = resources.sample(server)
            write_row(run_dir / "resources.jsonl", **row)
            gate.check(row)
            next_sample = time.monotonic() + SAMPLE_SECONDS

    while True:
        if server.poll() is not None:
            raise Stopped("server_start_failed")
        if time.monotonic() >= deadline:
            raise Stopped("startup_timeout")
        sample_if_due()
        if time.monotonic() >= next_probe:
            if ready(min(2, max(0.001, deadline - time.monotonic()))):
                if time.monotonic() >= deadline:
                    raise Stopped("startup_timeout")
                break
            next_probe = time.monotonic() + PROBE_SECONDS
        time.sleep(min(POLL_SECONDS, max(0, deadline - time.monotonic())))
    runner = children.start("runner", runner_args)
    deadline = time.monotonic() + RUN_SECONDS
    next_sample = time.monotonic() + SAMPLE_SECONDS
    while True:
        if time.monotonic() >= deadline:
            raise Stopped("run_timeout")
        if server.poll() is not None:
            raise Stopped("server_exited_during_run")
        result = runner.poll()
        if result is not None:
            if result != 0:
                raise Stopped("runner_failed")
            break
        sample_if_due()
        time.sleep(min(POLL_SECONDS, max(0, deadline - time.monotonic())))
    # Read the runner's own sanitized summary; do not parse model responses here.
    summary = json.loads((run_dir / "sanitized_result.local.json").read_text())
    if (summary["maximum_model_requests"] != 12 or type(summary["model_requests"]) is not int
            or not 0 <= summary["model_requests"] <= 12):
        raise Stopped("invalid_request_count")
    return summary["status"], summary["model_requests"]


def launch(run_dir, prepare_only=False):
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    log = run_dir / "launcher.jsonl"
    write_row(log, event="started", prepare_only=prepare_only, platform=sys.platform,
              startup_seconds=STARTUP_SECONDS, run_seconds=RUN_SECONDS, maximum_model_requests=12)
    previous_handlers = {}

    def interrupted(signum, frame):
        raise Stopped(f"signal_{signal.Signals(signum).name}", 128 + signum)

    children = None
    code, outcome = 1, {"status": "failed", "reason": "unexpected_exception"}
    with ExitStack() as stack:
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[sig] = signal.signal(sig, interrupted)
            if sys.platform != "linux":
                raise Stopped("linux_required")
            sock = stack.enter_context(reserve_port())
            resources = LinuxResources()
            cpu = []
            # First delta has a one-second baseline; samples then end 2s apart.
            time.sleep(1)
            cpu.append(resources.cpu())
            time.sleep(2)
            cpu.append(resources.cpu())
            time.sleep(2)
            row = resources.sample()
            cpu.append(row["host_cpu_percent"])
            reasons = idle_reasons(cpu, row)
            write_row(run_dir / "resources.jsonl", phase="idle", cpu_samples=cpu, **row)
            write_row(log, event="idle_gate", status="rejected" if reasons else "passed", reasons=reasons)
            if reasons:
                raise Stopped("idle_resource_rejection:" + ",".join(reasons))
            children = Children(stack, run_dir, child_env(run_dir, row["gpu_uuid"]))
            runner_args = [str(RUNNER), "--tasks", str(ROOT / ".cache/index_v0_2/tasks.jsonl"),
                           "--output-dir", str(run_dir)]
            preflight = children.start("preflight", [*runner_args, "--artifacts-only"])
            if preflight.wait() != 0:
                raise Stopped("artifact_preflight_failed")
            write_row(log, event="artifacts", status="passed")
            if prepare_only:
                outcome = {"status": "prepared", "model_requests": 0}
            else:
                status, count = supervise(children, sock, resources, run_dir, runner_args)
                outcome = {"status": "completed", "runner_status": status, "model_requests": count}
            code = 0
        except Stopped as exc:
            code, outcome = exc.code, {"status": "stopped", "reason": str(exc)}
        except Exception as exc:
            outcome = {"status": "failed", "reason": type(exc).__name__, "detail": str(exc)}
        finally:
            # Repeated signals cannot interrupt cleanup of our own process groups.
            for sig in previous_handlers:
                signal.signal(sig, signal.SIG_IGN)
            try:
                if children is not None:
                    children.close()
                write_row(log, event="finished", exit_code=code, **outcome)
            finally:
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)
    print(json.dumps({**outcome, "exit_code": code, "run_dir": str(run_dir)}), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return launch(ROOT / ".cache/index_v0_3" / f"{stamp}-{uuid.uuid4().hex[:8]}", args.prepare_only)


if __name__ == "__main__":
    raise SystemExit(main())
