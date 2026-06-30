import os
import re
import time
import logging
from urllib.parse import urlparse

import httpx
from openai import OpenAI


MAX_RETRIES = 10
TIMEOUT = float(os.getenv("SHOPPINGBENCH_LLM_TIMEOUT", "180"))
CALL_DELAY = float(os.getenv("SHOPPINGBENCH_LLM_CALL_DELAY", "0"))
RETRY_DELAY = float(os.getenv("SHOPPINGBENCH_LLM_RETRY_DELAY", "3"))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger()


CONTEXT_LENGTH_RE = re.compile(
    r"maximum context length is (?P<limit>\d+) tokens.*requested (?P<requested>\d+) tokens "
    r"\((?P<prompt>\d+) in the messages, (?P<completion>\d+) in the completion\)",
    re.IGNORECASE | re.DOTALL,
)


def should_bypass_env_proxy(base_url: str | None, model_config: dict | None = None) -> bool:
    model = (model_config or {}).get("model", "")
    if isinstance(model, str) and model.startswith("mimo"):
        return True

    if not base_url:
        return False

    hostname = (urlparse(base_url).hostname or "").lower()
    return hostname in {"127.0.0.1", "localhost", "0.0.0.0", "::1"} or "mimo" in hostname


def chat_completion_stream(client: OpenAI, messages: list[dict[str, str]], model_config: dict):
    stream = client.chat.completions.create(
        messages=messages,
        **model_config,
    )

    reasoning_content = ""
    content = ""
    for event in stream:
        try:
            reasoning_content += event.choices[0].delta.reasoning_content
        except:
            pass
        try:
            content += event.choices[0].delta.content
        except:
            pass

    return reasoning_content, content


def chat_completion(client: OpenAI, messages: list[dict[str, str]], model_config: dict):
    completion = client.chat.completions.create(
        messages=messages,
        **model_config,
    )

    reasoning_content = ""
    content = ""
    try:
        reasoning_content = completion.choices[0].message.reasoning_content
    except:
        pass
    try:
        content = completion.choices[0].message.content
    except:
        pass

    return reasoning_content, content


def ask_llm(
    messages: list[dict[str, str]],
    model_config: dict,
    base_url: str = None,
    api_key: str = None,
) -> tuple[str, str]:
    success = False
    request_model_config = dict(model_config)
    model = model_config.get("model", "")
    is_mimo_model = isinstance(model, str) and model.startswith("mimo")
    resolved_base_url = base_url if base_url else (
        os.environ.get("MIMO_BASE_URL") if is_mimo_model else os.environ.get("OPENAI_BASE_URL")
    )
    resolved_api_key = api_key if api_key else (
        os.environ.get("MIMO_API_KEY") if is_mimo_model else os.environ.get("OPENAI_API_KEY")
    )

    for i in range(MAX_RETRIES):
        client = None
        try:
            if CALL_DELAY > 0:
                time.sleep(CALL_DELAY)
            http_client = (
                httpx.Client(trust_env=False)
                if should_bypass_env_proxy(resolved_base_url, model_config)
                else None
            )
            client = OpenAI(
                base_url=resolved_base_url,
                api_key=resolved_api_key,
                http_client=http_client,
                timeout=TIMEOUT,
            )

            if request_model_config.get("stream", False):
                reasoning_content, content = chat_completion_stream(
                    client, messages, request_model_config
                )
            else:
                reasoning_content, content = chat_completion(
                    client, messages, request_model_config
                )

            if reasoning_content or content:
                success = True
                break
            else:
                raise Exception("reasoning_content and content is empty")
        except Exception as e:
            match = CONTEXT_LENGTH_RE.search(str(e))
            if match and "max_tokens" in request_model_config:
                limit = int(match.group("limit"))
                prompt_tokens = int(match.group("prompt"))
                current_max_tokens = int(request_model_config.get("max_tokens") or 0)
                adjusted_max_tokens = max(32, limit - prompt_tokens - 1)
                if 0 < adjusted_max_tokens < current_max_tokens:
                    logger.warning(
                        "Reducing max_tokens from %s to %s for this request because "
                        "prompt uses %s/%s tokens.",
                        current_max_tokens,
                        adjusted_max_tokens,
                        prompt_tokens,
                        limit,
                    )
                    request_model_config["max_tokens"] = adjusted_max_tokens
                    continue
            logger.error(f"Error occurred: {e}. Retry {i+1}/{MAX_RETRIES}.")
            time.sleep(RETRY_DELAY)
        finally:
            if client is not None:
                client.close()

    if not success:
        logger.error(f"Retry {MAX_RETRIES} but can't success!")
        reasoning_content = ""
        content = ""
    return reasoning_content, content


if __name__ == "__main__":
    reasoning_content, content = ask_llm(
        messages=[{"role": "user", "content": "hi"}],
        model_config={
            "model": "gemini-2.5-flash",
            "temperature": 0,
            "max_tokens": 8192,
            "extra_body": {
                "google": {
                    "thinkingConfig": {
                        "includeThoughts": True
                    },
                    "thought_tag_marker": "think"
                }
            }
        },
    )
    print(f"reasoning_content: {reasoning_content}\ncontent: {content}")
