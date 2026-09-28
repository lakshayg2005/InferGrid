"""Run a worker: python -m infergrid.worker --id worker-1 --port 8701 --backend sim"""

import argparse

import uvicorn

from infergrid.worker.app import create_app
from infergrid.worker.backends import Backend, OllamaBackend, SimBackend, SimConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an InferGrid worker.")
    parser.add_argument("--id", default="worker-1", help="worker name shown in stats and logs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8701)
    parser.add_argument("--backend", choices=["sim", "ollama"], default="sim")
    parser.add_argument("--model", default="qwen2.5:0.5b", help="Ollama model name")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-concurrency", type=int, default=None)
    args = parser.parse_args()

    backend: Backend
    if args.backend == "sim":
        config = SimConfig()
        if args.max_concurrency:
            config.max_concurrency = args.max_concurrency
        backend = SimBackend(config)
    else:
        backend = OllamaBackend(args.model, args.ollama_url, args.max_concurrency or 2)

    print(f"{args.id}: {args.backend} backend on http://{args.host}:{args.port}")
    uvicorn.run(create_app(backend, args.id), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
