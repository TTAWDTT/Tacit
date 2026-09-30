"""Offline process fixture. Never imports a model or the pilot runner."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

role, scenario, directory, *fds = sys.argv[1:]
root = Path(directory)
(root / f"{role}.pid").write_text(str(os.getpid()))
(root / f"{role}.env.json").write_text(json.dumps({
    key: os.environ.get(key) for key in (
        "TLU_TORCH_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
        "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE", "NO_PROXY")
} | {"nice": os.getpriority(os.PRIO_PROCESS, 0),
     "blocked_signals": list(signal.pthread_sigmask(signal.SIG_BLOCK, []))}))
if role == "preflight":
    sys.exit(3 if scenario == "hash_failure" else 0)
if role == "runner":
    if scenario == "runner_failure":
        sys.exit(4)
    if scenario in {"run_timeout", "sigterm", "resource_stop", "telemetry_error", "server_exit"}:
        time.sleep(10)
    if scenario == "bad_summary":
        (root / "sanitized_result.local.json").write_text("invalid")
    else:
        (root / "sanitized_result.local.json").write_text(json.dumps({
            "status": "complete", "maximum_model_requests": 12,
            "model_requests": 13 if scenario == "over_limit" else 12}))
    sys.exit(0)
if scenario == "startup_failure":
    sys.exit(5)
if scenario == "descendant":
    # Deliberately survives SIGTERM; the supervisor must kill the owned group.
    descendant = subprocess.Popen([sys.executable, "-c", "import signal,time; "
                      "signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"])
    (root / "descendant.pid").write_text(str(descendant.pid))
if scenario == "startup_timeout":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(10)
sock = socket.socket(fileno=int(fds[0]))
while True:
    connection, address = sock.accept()
    with connection:
        connection.recv(4096)
        body = json.dumps({"data": [{"id": "Qwen3-1.7B"}]}).encode()
        connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                           + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
    if scenario == "server_exit":
        time.sleep(0.08)
        sys.exit(6)
