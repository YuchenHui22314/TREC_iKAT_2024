"""VLLMServer — boot a vLLM OpenAI-compatible server as a SEPARATE process on ONE GPU.

GPU ISOLATION (critical — caused crashes before): the subprocess env sets
`CUDA_VISIBLE_DEVICES=<gpu_id>` so vLLM sees ONLY that physical GPU (GPU 3); inside the
subprocess it is "cuda:0". The main process (FAISS on GPUs 0,1,2 + ANCE encoder on GPU 0)
keeps full visibility but never touches GPU 3. The two never collide.

The server runs from a DEDICATED env (`vllm_bin` points at e.g.
/data/rech/huiyuche/envs/vllm_qwen3/bin/vllm) — NOT the main trec_ikat env (whose vllm
0.7.3 cannot serve Qwen3). NEVER a bare `vllm` (would resolve to the wrong env).
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass
class VLLMServerConfig:
    vllm_bin: str = "/data/rech/huiyuche/envs/vllm_qwen3/bin/vllm"
    hf_model: str = "Qwen/Qwen3-32B-AWQ"
    served_model_name: str = "qwen3-32b"
    gpu_id: int = 3
    host: str = "127.0.0.1"
    port: int = 8100
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 16
    quantization: Optional[str] = "awq_marlin"     # None -> let vLLM infer
    # the HF HUB cache dir that directly contains the models--Qwen--... dirs. With HF_HOME
    # set to /data/rech/huiyuche/huggingface, huggingface-cli lands models in .../huggingface/hub,
    # so THAT (the /hub subdir) is the real hub cache vLLM must read.
    download_dir: str = "/data/rech/huiyuche/huggingface/hub"
    log_path: str = "/part/01/Tmp/yuchen/vllm_server.log"
    startup_timeout_s: float = 1800.0
    health_poll_s: float = 5.0


class VLLMServer:
    def __init__(self, cfg: VLLMServerConfig):
        self.cfg = cfg
        self._proc: Optional[subprocess.Popen] = None
        self._log = None

    def base_url(self) -> str:
        return f"http://{self.cfg.host}:{self.cfg.port}/v1"

    def _health_url(self) -> str:
        return f"http://{self.cfg.host}:{self.cfg.port}/health"

    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    def start(self):
        c = self.cfg
        if not os.path.exists(c.vllm_bin):
            raise FileNotFoundError(
                f"vllm binary not found: {c.vllm_bin} (build the dedicated vllm_qwen3 env first)")
        argv = [
            c.vllm_bin, "serve", c.hf_model,
            "--served-model-name", c.served_model_name,
            "--gpu-memory-utilization", str(c.gpu_memory_utilization),
            "--max-model-len", str(c.max_model_len),
            "--max-num-seqs", str(c.max_num_seqs),
            "--host", c.host, "--port", str(c.port),
            "--download-dir", c.download_dir,
        ]
        if c.quantization:
            argv += ["--quantization", c.quantization]
        # GPU isolation: the subprocess sees ONLY physical GPU c.gpu_id.
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(c.gpu_id)
        env["HF_HOME"] = c.download_dir
        self._log = open(c.log_path, "w")
        print(f"[vllm_server] launching on GPU {c.gpu_id}: {' '.join(argv)}")
        print(f"[vllm_server] logging to {c.log_path}")
        self._proc = subprocess.Popen(argv, env=env, stdout=self._log, stderr=subprocess.STDOUT,
                                      start_new_session=True)

    def wait_until_ready(self):
        c = self.cfg
        t0 = time.time()
        while time.time() - t0 < c.startup_timeout_s:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited early (code {self._proc.returncode}); see {c.log_path}")
            try:
                with urllib.request.urlopen(self._health_url(), timeout=5) as r:
                    if r.status == 200:
                        print(f"[vllm_server] ready in {time.time()-t0:.0f}s at {self.base_url()}")
                        return
            except Exception:
                pass
            time.sleep(c.health_poll_s)
        self.stop()
        raise TimeoutError(f"vLLM server not ready within {c.startup_timeout_s}s; see {c.log_path}")

    def stop(self):
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                try:
                    self._proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self._log is not None:
            self._log.close()
        print("[vllm_server] stopped")
        self._proc = None
