"""Chat with the cluster from the terminal. Shows which worker served each reply and its cache hits.

    python scripts/chat.py
"""

import argparse
import json

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with an InferGrid gateway.")
    parser.add_argument("--url", default="http://127.0.0.1:8700")
    parser.add_argument("--system", default="You are a helpful, concise assistant.")
    args = parser.parse_args()

    history = [{"role": "system", "content": args.system}]
    print("Type a message (Ctrl+C to quit).")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (KeyboardInterrupt, EOFError):
            return
        if not user:
            continue
        history.append({"role": "user", "content": user})

        reply = []
        body = {"messages": history, "stream": True}
        with httpx.stream("POST", f"{args.url}/v1/chat/completions", json=body, timeout=None) as resp:
            if resp.status_code != 200:
                print(f"error {resp.status_code}: {resp.read().decode()}")
                history.pop()
                continue
            print(f"[{resp.headers['x-infergrid-worker']}] ", end="", flush=True)
            usage = None
            for line in resp.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if "error" in chunk:
                    print(f"\n[stream failed: {chunk['error']['message']}]")
                    break
                text = chunk["choices"][0]["delta"].get("content", "")
                reply.append(text)
                print(text, end="", flush=True)
                usage = chunk.get("usage", usage)
        if usage:
            cached = usage["prompt_tokens_details"]["cached_tokens"]
            print(f"\n  ({usage['prompt_tokens']} prompt tokens, {cached} from cache)")
        history.append({"role": "assistant", "content": "".join(reply)})


if __name__ == "__main__":
    main()
