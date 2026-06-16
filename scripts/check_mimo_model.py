#!/usr/bin/env python3
import os
import sys

from openai import OpenAI
import httpx


def main() -> int:
    model = os.getenv("MIMO_MODEL")
    api_key = os.getenv("MIMO_API_KEY")
    base_url = os.getenv("MIMO_BASE_URL")

    print(f"MIMO_MODEL={model!r}")
    print(f"MIMO_API_KEY_SET={bool(api_key)}")
    print(f"MIMO_BASE_URL={base_url!r}")
    max_completion_tokens = int(os.getenv("MIMO_CHECK_MAX_COMPLETION_TOKENS", "1024"))

    missing = [
        name
        for name, value in [
            ("MIMO_MODEL", model),
            ("MIMO_API_KEY", api_key),
            ("MIMO_BASE_URL", base_url),
        ]
        if not value
    ]
    if missing:
        print(f"ERROR: missing env vars: {', '.join(missing)}", file=sys.stderr)
        return 2

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(trust_env=False),
    )
    try:
        kwargs = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are MiMo, an AI assistant developed by Xiaomi.",
                },
                {
                    "role": "user",
                    "content": "please introduce yourself briefly",
                }
            ],
            "temperature": 1.0,
        }
        if model.startswith("mimo"):
            kwargs["max_completion_tokens"] = max_completion_tokens
        else:
            kwargs["max_tokens"] = max_completion_tokens

        resp = client.chat.completions.create(**kwargs)
        message = resp.choices[0].message
        content = message.content or ""
        reasoning_content = getattr(message, "reasoning_content", None) or ""
        print("CALL_OK=1")
        print(f"CONTENT_LEN={len(content)}")
        print(f"REASONING_CONTENT_LEN={len(reasoning_content)}")
        print(f"RESPONSE={content!r}")
        return 0
    except Exception as exc:
        print("CALL_OK=0")
        print(f"ERROR_TYPE={type(exc).__name__}")
        print(f"ERROR={exc}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
