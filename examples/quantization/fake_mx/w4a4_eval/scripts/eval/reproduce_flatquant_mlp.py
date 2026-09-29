"""Own one server and run the archived FlatQuant MLP dataset baseline (Linux)."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PARAM_SHA256 = "6d47223d8431f525d0bbcbf9e4588ee39a8b482bc91607aef27ca845abd8bb6e"
DATASETS = {"math500": (500, 0.934), "mmlu_pro": (12032, 0.7843), "livecodebench": (1055, 0.5877)}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def request(port, endpoint, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/{endpoint}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60 if data else 3) as response:
        return json.load(response)


def prepare_model_view(model, out, param):
    """Link the original model and copy the sidecar; serve installs the config."""
    if out.is_relative_to(model):
        raise ValueError("Output must be outside the original model directory")
    out.mkdir(parents=True, exist_ok=False)
    view = out / "model_view"
    view.mkdir()
    for path in model.iterdir():
        if path.name not in {"quant_model_description.json", "flatquant_params.safetensors"}:
            (view / path.name).symlink_to(path, target_is_directory=path.is_dir())
    shutil.copy2(param, view / "flatquant_params.safetensors")
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="New directory; never reuse a previous run")
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    if len(args.datasets) != len(set(args.datasets)):
        parser.error("Each dataset may be selected only once")
    if args.card < 0 or not 1 <= args.port <= 65535:
        parser.error("Card must be nonnegative and port must be between 1 and 65535")
    base = Path(__file__).resolve().parents[2]
    model, out = args.model.resolve(), args.out.resolve()
    if out.exists():
        parser.error(f"Output already exists: {out}")
    if out.is_relative_to(model):
        parser.error("Output must be outside the original model directory")
    if not (model / "config.json").is_file():
        parser.error(f"Missing model config: {model}")
    if importlib.metadata.version("evalscope") != "1.10.0":
        parser.error("This baseline requires evalscope==1.10.0")
    param = base / "params/flatquant/qwen3_5_9b_flatquant_mlp-only_w4a4.safetensors"
    config = base / "configs/qwen3_5_9b_flatquant_mlp-only_w4a4.json"
    if not config.is_file():
        parser.error(f"Missing quantization config: {config}")
    if hashlib.sha256(param.read_bytes()).hexdigest() != PARAM_SHA256:
        parser.error("Sidecar SHA mismatch (run git lfs pull; do not use a pointer or other parameters)")
    with socket.socket() as probe:
        probe.bind(("0.0.0.0", args.port))
    if "livecodebench" in args.datasets:
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
        image = subprocess.check_output(["docker", "image", "inspect", "python:3.11-slim"], text=True)
    else:
        image = None
    view = prepare_model_view(model, out, param)
    state = {
        "status": "starting",
        "datasets": {},
        "args": vars(args) | {"model": str(model), "out": str(out)},
        "param_sha256": PARAM_SHA256,
        "python": sys.executable,
        "evalscope": importlib.metadata.version("evalscope"),
        "sandbox_image": json.loads(image) if image else None,
        "modelscope_cache": os.environ.get("MODELSCOPE_CACHE"),
    }
    save(out / "status.json", state)
    env = os.environ | {"PYTHON": sys.executable}
    command = [
        "bash",
        str(base / "scripts/serve/vllm_serve.sh"),
        str(view),
        str(args.card),
        str(args.port),
        str(config),
        "eager",
    ]
    save(out / "serve-command.json", command)
    proc = None
    try:
        with (out / "serve.log").open("w") as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state["server_pid"] = proc.pid
            save(out / "status.json", state)
            deadline = time.monotonic() + 900
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"Owned server exited {proc.returncode}; see serve.log")
                try:
                    models = request(args.port, "models")
                    if any(m.get("id") == "qwen3.5" for m in models.get("data", [])):
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Server readiness exceeded 900 seconds")
                time.sleep(5)
            if "Loaded 288 fake-MX transform params" not in (out / "serve.log").read_text(errors="replace"):
                raise RuntimeError("Expected sidecar load evidence missing")
            save(out / "models.json", models)
            smoke = request(
                args.port,
                "chat/completions",
                {
                    "model": "qwen3.5",
                    "messages": [{"role": "user", "content": "1+1="}],
                    "max_tokens": 32,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            choices = smoke.get("choices") or []
            if not choices or not choices[0].get("message", {}).get("content"):
                raise RuntimeError(f"Smoke failed: {smoke}")
            save(out / "smoke.json", smoke)
            for name in args.datasets:
                state.update(status="evaluating", current_dataset=name)
                save(out / "status.json", state)
                with (out / f"{name}.log").open("w") as evaluation_log:
                    subprocess.run(
                        [sys.executable, str(base / f"scripts/eval/run_{name}.py"), str(args.port), str(out / name)],
                        stdout=evaluation_log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=86400,
                    )
                reports = list((out / name).glob("*/reports/qwen3.5/*.json"))
                if len(reports) != 1:
                    raise RuntimeError(f"Expected one {name} report, found {len(reports)}")
                report = json.loads(reports[0].read_text())
                expected, baseline = DATASETS[name]
                if report["num"] != expected:
                    raise RuntimeError(f"{name} sample count {report['num']} != {expected}")
                error_lines = [
                    line
                    for line in (out / f"{name}.log").read_text(errors="replace").splitlines()
                    if re.search(r"ERROR|Traceback|timed out|TimeoutError|Retrying|ConnectionError", line)
                ]
                save(out / f"{name}-error-candidates.json", error_lines)
                state["datasets"][name] = {
                    "report": str(reports[0]),
                    "score": report["score"],
                    "num": report["num"],
                    "within_2pt": abs(report["score"] - baseline) <= 0.0200001,
                    "error_candidate_count": len(error_lines),
                }
                save(out / "status.json", state)
            state["status"] = "completed_pending_error_and_dataset_snapshot_audit"
            state.pop("current_dataset", None)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        if proc is not None:
            # Only this session's process group; never pkill by a shared port/name.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=45)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        save(out / "status.json", state)


if __name__ == "__main__":
    main()
