"""Start and stop a cluster of worker and gateway processes on this machine.

Used by scripts/run_local.py, scripts/benchmark.py and scripts/chaos.py.
"""

import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class _Process:
    url: str
    proc: subprocess.Popen
    is_ours: Callable[[dict], bool]  # recognises our own /health response
    args: list[str] = field(default_factory=list)  # to restart it


@dataclass
class LocalCluster:
    workers: int = 3
    backend: str = "sim"
    model: str = "qwen2.5:0.5b"
    router: str = "consistent_hash"
    epsilon: float = 0.25
    max_failovers: int = 2
    membership: bool = True  # SWIM failure detection (see membership/swim.py)
    swim_offset: int = 1000  # a member's UDP port is its HTTP port plus this
    hedge_delay_ms: float | None = None
    gateway_port: int = 8700
    worker_base_port: int = 8701
    quiet: bool = False  # hide process output (benchmarks)
    worker_urls: list[str] = field(default_factory=list, init=False)
    _workers: list[_Process] = field(default_factory=list, init=False)
    _gateway: _Process | None = field(default=None, init=False)

    @property
    def gateway_url(self) -> str:
        return f"http://127.0.0.1:{self.gateway_port}"

    def start(self) -> None:
        try:
            worker_ports = [self.worker_base_port + i for i in range(self.workers)]
            swim_addrs = [f"127.0.0.1:{p + self.swim_offset}" for p in worker_ports]

            for i, port in enumerate(worker_ports):
                name = f"worker-{i + 1}"
                url = f"http://127.0.0.1:{port}"
                self.worker_urls.append(url)
                cmd = ["-m", "infergrid.worker", "--id", name, "--port", str(port),
                       "--backend", self.backend, "--model", self.model]
                if self.membership:
                    seeds = [a for j, a in enumerate(swim_addrs) if j != i]
                    cmd += ["--swim-port", str(port + self.swim_offset), "--seeds", ",".join(seeds)]
                self._workers.append(
                    _Process(url, self._spawn(cmd), lambda b, name=name: b.get("worker_id") == name, cmd))

            cmd = ["-m", "infergrid.gateway", "--port", str(self.gateway_port), "--workers", ",".join(self.worker_urls),
                   "--router", self.router, "--epsilon", str(self.epsilon), "--max-failovers", str(self.max_failovers)]
            if self.membership:
                cmd += ["--swim-port", str(self.gateway_port + self.swim_offset), "--seeds", ",".join(swim_addrs)]
            if self.hedge_delay_ms:
                cmd += ["--hedge-delay-ms", str(self.hedge_delay_ms)]
            self._gateway = _Process(self.gateway_url, self._spawn(cmd), lambda b: b.get("workers") == self.worker_urls)

            for p in [*self._workers, self._gateway]:
                if not _wait_until_healthy(p):
                    raise RuntimeError(f"{p.url} did not start (is the port already in use?)")
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        procs = [p.proc for p in self._workers] + ([self._gateway.proc] if self._gateway else [])
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.wait()

    def kill_worker(self, index: int) -> None:
        """Kill a worker abruptly, like a machine losing power: no graceful shutdown."""
        proc = self._workers[index].proc
        proc.kill()
        proc.wait()

    def restart_worker(self, index: int) -> bool:
        worker = self._workers[index]
        worker.proc = self._spawn(worker.args)
        return _wait_until_healthy(worker)

    def gateway_alive(self) -> bool:
        return self._gateway is not None and self._gateway.proc.poll() is None

    def exited_workers(self) -> list[tuple[str, int]]:
        return [(p.url, p.proc.returncode) for p in self._workers if p.proc.poll() is not None]

    def _spawn(self, args: list[str]) -> subprocess.Popen:
        out = subprocess.DEVNULL if self.quiet else None
        return subprocess.Popen([sys.executable, *args], cwd=ROOT, stdout=out, stderr=out)

    def __enter__(self) -> "LocalCluster":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def _wait_until_healthy(p: _Process, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and p.proc.poll() is None:
        try:
            resp = httpx.get(f"{p.url}/health", timeout=1.0)
            if resp.status_code == 200 and p.is_ours(resp.json()):
                return True
        except (httpx.TransportError, ValueError):
            pass
        time.sleep(0.2)
    return False
