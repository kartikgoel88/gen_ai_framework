#!/usr/bin/env python3
"""
Run a lightweight local LLM via Ollama (or any OpenAI-compatible endpoint).

Uses a small, accurate model by default (Qwen2.5 1.5B). Data stays on your machine.

Prerequisites:
  - Install Ollama app: https://ollama.com (or brew install ollama)

Usage:
  python scripts/run_local_llm.py --pull              # pull model then interactive
  python scripts/run_local_llm.py --serve             # start server if needed, then chat
  python scripts/run_local_llm.py --serve --pull      # ensure server + model, then chat
  python scripts/run_local_llm.py "What is 2+2?"
  python scripts/run_local_llm.py --model llama3.2:3b --stream
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Project root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Default: lightweight + good accuracy. Alternatives: llama3.2:3b, qwen2.5:3b, phi3:mini, smollm2:360m
DEFAULT_MODEL = "qwen2.5:1.5b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"


def _find_ollama_bin() -> str | None:
    """Return path to ollama executable or None if not found."""
    exe = shutil.which("ollama")
    if exe:
        return exe
    # Common install locations
    for path in (
        "/usr/local/bin/ollama",
        Path.home() / ".local/bin/ollama",
    ):
        if Path(path).is_file():
            return str(path)
    return None


def _check_ollama(base_url: str) -> bool:
    try:
        import urllib.request
        url = base_url.rstrip("/").replace("/v1", "") + "/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2) as _:
            return True
    except Exception:
        return False


def _model_available(base_url: str, model: str) -> bool:
    """Return True if the given model is already pulled (Ollama /api/tags)."""
    try:
        import urllib.request
        url = base_url.rstrip("/").replace("/v1", "") + "/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        for m in data.get("models") or []:
            name = m.get("name") or ""
            if name == model or name.startswith(model + ":"):
                return True
        return False
    except Exception:
        return False


def _run_serve(base_url: str) -> bool:
    """Start ollama serve in background. Return True if server becomes reachable."""
    bin_ = _find_ollama_bin()
    if not bin_:
        print("Ollama CLI not found. Install from https://ollama.com or run: brew install ollama", file=sys.stderr)
        return False
    if _check_ollama(base_url):
        return True
    print("Starting Ollama server in background...", file=sys.stderr)
    try:
        subprocess.Popen(
            [bin_, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        print(f"Failed to start ollama serve: {e}", file=sys.stderr)
        return False
    for _ in range(30):
        time.sleep(1)
        if _check_ollama(base_url):
            print("Ollama server is up.", file=sys.stderr)
            return True
    print("Ollama server did not become reachable in time.", file=sys.stderr)
    return False


def _run_pull(model: str) -> bool:
    """Run ollama pull <model>. Return True on success."""
    bin_ = _find_ollama_bin()
    if not bin_:
        print("Ollama CLI not found. Install from https://ollama.com or run: brew install ollama", file=sys.stderr)
        return False
    print(f"Pulling model {model}...", file=sys.stderr)
    try:
        subprocess.run([bin_, "pull", model], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ollama pull failed: {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        return False


def _create_llm(base_url: str, model: str, temperature: float):
    from src.framework.llm.local_provider import LocalLLMProvider
    return LocalLLMProvider(base_url=base_url, model=model, temperature=temperature)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a local LLM (Ollama). Lightweight default: qwen2.5:1.5b.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Optional one-shot prompt; if omitted, run interactive chat.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL}). Run 'ollama pull <model>' first.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"OpenAI-compatible API base URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response token-by-token.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip checking if Ollama is reachable.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start Ollama server in background if not reachable (requires Ollama app installed).",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull the model before running (ollama pull <model>).",
    )
    args = parser.parse_args()

    if not args.no_check and not _check_ollama(args.base_url):
        if args.serve or _find_ollama_bin():
            if not _run_serve(args.base_url):
                sys.exit(1)
        else:
            print("Ollama does not appear to be running or reachable.", file=sys.stderr)
            print(f"  Start it: ollama serve  (or run this script with --serve)", file=sys.stderr)
            print(f"  Pull model: ollama pull {args.model}  (or run this script with --pull)", file=sys.stderr)
            print(f"  Install: https://ollama.com  or  brew install ollama", file=sys.stderr)
            sys.exit(1)
    if args.pull:
        if not _run_pull(args.model):
            sys.exit(1)
    elif not args.no_check and not _model_available(args.base_url, args.model):
        print(f"Model {args.model} not found. Pulling...", file=sys.stderr)
        if not _run_pull(args.model):
            print(f"Run manually: ollama pull {args.model}", file=sys.stderr)
            sys.exit(1)

    llm = _create_llm(args.base_url, args.model, args.temperature)

    if args.prompt:
        if args.stream:
            for chunk in llm.stream_invoke(args.prompt):
                print(chunk, end="", flush=True)
            print()
        else:
            print(llm.invoke(args.prompt))
        return

    # Interactive loop
    print(f"Local LLM ({args.model}). Type a message and press Enter. Ctrl+D or 'exit' to quit.", file=sys.stderr)
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line or line.lower() in ("exit", "quit", "q"):
            break
        if args.stream:
            for chunk in llm.stream_invoke(line):
                print(chunk, end="", flush=True)
            print()
        else:
            print(llm.invoke(line))


if __name__ == "__main__":
    main()
