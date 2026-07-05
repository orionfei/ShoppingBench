import os
import sys
import time
import copy
import hashlib
import portalocker
import multiprocessing as mp
import asyncio
import ujson as json
from pathlib import Path
from tqdm import tqdm

from toolkit import tools, toolmap
from util.llm import ask_llm
from util.message import Message, USER_ROLES, ASSISTANT_ROLES
from util.harness_fsm import (
    build_harness_snapshot,
    build_harness_user_prompt,
    search_trace_markdown,
)
from util.system_prompt import build_system_prompt


MAX_STEPS = 30


def get_system_prompt(config: dict, harness_snapshot=None) -> str:
    if harness_snapshot is not None:
        return build_system_prompt(
            harness_snapshot.prompt_file,
            include_tools=harness_snapshot.include_tools,
        )
    return build_system_prompt(
        config["system_prompt_file"],
        exclude_tools=config.get("exclude_tools", []),
    )


def get_user_prompt(
    message: Message,
    history_messages: list[str],
    config: dict | None = None,
    harness_snapshot=None,
) -> str:
    user_message = message.to_string(USER_ROLES)
    if user_message:
        history_messages.append(user_message)

    assistant_message = message.to_string(ASSISTANT_ROLES)
    if assistant_message:
        history_messages.append(assistant_message)

    config = config or {}
    if config.get("history_compression") == "state_folded":
        if harness_snapshot is None:
            harness_snapshot = build_harness_snapshot(
                history_messages,
                prompt_files=config.get("harness_prompt_files"),
            )
        return build_harness_user_prompt(harness_snapshot, history_messages)

    history = "\n\n".join(history_messages)
    return f"# Dialogue Records History\n{history}"


def write_search_trace(query: str, harness_snapshot, config: dict) -> str | None:
    if config.get("history_compression") != "state_folded":
        return None
    trace_dir = config.get("harness_search_trace_dir")
    if not trace_dir:
        rollout_file = config.get("rollout_file", "rollout.jsonl")
        trace_dir = str(Path(rollout_file).with_suffix("")) + "_search_trace"
    path = Path(trace_dir)
    path.mkdir(parents=True, exist_ok=True)
    query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()[:12]
    trace_path = path / f"{query_hash}.md"
    trace_path.write_text(
        search_trace_markdown(query, harness_snapshot.search_trace),
        encoding="utf-8",
    )
    return str(trace_path)


def think(
    system_prompt: str,
    user_prompt: str,
    model_config: dict,
    base_url: str | None = None,
    api_key: str | None = None,
) -> tuple[str, str, Message]:
    reasoning_content, content = ask_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model_config=model_config,
        base_url=base_url,
        api_key=api_key
    )

    return reasoning_content, content, Message.from_string(reasoning_content, content)


def act(message: Message, allowed_tools: set[str] | None = None) -> list[dict]:
    obs = []
    tool_names = {commend["name"] for commend in message.tool_call}
    mixed_decision_actions = (
        allowed_tools is not None
        and {"find_product", "recommend_product", "terminate"}.issubset(allowed_tools)
        and "find_product" in tool_names
        and bool({"recommend_product", "terminate"} & tool_names)
    )
    lone_decision_terminate = (
        allowed_tools is not None
        and {"find_product", "recommend_product", "terminate"}.issubset(allowed_tools)
        and "terminate" in tool_names
        and "recommend_product" not in tool_names
    )
    for commend in message.tool_call:
        if mixed_decision_actions:
            obs.append(
                {
                    "tool_call_id": commend["tool_call_id"],
                    "results": {
                        "error": "mixed_decision_actions_not_allowed",
                        "tool": commend["name"],
                    },
                }
            )
            continue
        if lone_decision_terminate and commend["name"] == "terminate":
            obs.append(
                {
                    "tool_call_id": commend["tool_call_id"],
                    "results": {
                        "error": "terminate_requires_recommend_product_in_decision",
                        "tool": commend["name"],
                    },
                }
            )
            continue
        if allowed_tools is not None and commend["name"] not in allowed_tools:
            obs.append(
                {
                    "tool_call_id": commend["tool_call_id"],
                    "results": {
                        "error": "tool_not_allowed_in_current_state",
                        "tool": commend["name"],
                        "allowed_tools": sorted(allowed_tools),
                    },
                }
            )
            continue
        if commend["name"] not in toolmap:
            continue
        tool = toolmap[commend["name"]]
        obs.append(
            {
                "tool_call_id": commend["tool_call_id"],
                "results": asyncio.run(tool.execute(**commend["parameters"])) if tool.name == "web_search" else tool.execute(**commend["parameters"]),
            }
        )
    return obs


def is_terminate(
    message: Message,
    config: dict | None = None,
    allowed_tools: set[str] | None = None,
) -> bool:
    tool_names = {
        commend["name"]
        for commend in message.tool_call
        if allowed_tools is None or commend["name"] in allowed_tools
    }
    if (
        allowed_tools is not None
        and {"find_product", "recommend_product", "terminate"}.issubset(allowed_tools)
        and "find_product" in tool_names
        and bool({"recommend_product", "terminate"} & tool_names)
    ):
        return False
    if (
        allowed_tools is not None
        and {"find_product", "recommend_product", "terminate"}.issubset(allowed_tools)
        and "terminate" in tool_names
        and "recommend_product" not in tool_names
    ):
        return False
    if (config or {}).get("stop_after_recommend") and "recommend_product" in tool_names:
        return True
    if (not message.think and not message.tool_call and not message.response) or "terminate" in tool_names:
        return True
    return False


def react_loop(query: str, config: dict):
    corpus_tracker = []
    history_messages = []
    message = Message(user=query)
    #print(f"System Prompt:\n{system_prompt}")
    for step in range(1, MAX_STEPS + 1):
        harness_snapshot = None
        if config.get("history_compression") == "state_folded":
            user_message = message.to_string(USER_ROLES)
            if user_message:
                history_messages.append(user_message)

            assistant_message = message.to_string(ASSISTANT_ROLES)
            if assistant_message:
                history_messages.append(assistant_message)

            harness_snapshot = build_harness_snapshot(
                history_messages,
                prompt_files=config.get("harness_prompt_files"),
            )
            user_prompt = build_harness_user_prompt(harness_snapshot, history_messages)
            system_prompt = get_system_prompt(config, harness_snapshot)
            search_trace_file = write_search_trace(query, harness_snapshot, config)
        else:
            user_prompt = get_user_prompt(message, history_messages, config)
            system_prompt = get_system_prompt(config)
            search_trace_file = None
        message.clear()
        reasoning_content, content, message = think(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_config=config["model_config"],
            base_url=config.get("base_url", ""),
            api_key=config.get("api_key", ""),
        )
        if message.tool_call:
            message.obs = act(
                message,
                allowed_tools=harness_snapshot.include_tools if harness_snapshot else None,
            )

        corpus_tracker.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "completion": {
                    "reasoning_content": reasoning_content,
                    "content": content,
                    "message": copy.deepcopy(message.to_dict()),
                },
                "extra_info": {
                    "step": step,
                    "query": query,
                    "timestamp": int(time.time() * 1000),
                    "history_compression": config.get("history_compression", "raw"),
                    "harness_state": harness_snapshot.state_name if harness_snapshot else None,
                    "harness_search_trace_file": search_trace_file,
                },
            }
        )
        #print(f"{'*' * 20}Setps: {step}/{MAX_STEPS}{'*' * 20}\nReasoning Content: {reasoning_content}\nContent: {content}\nMessage: {json.dumps(message.to_dict(), indent=4)}\n")
        if is_terminate(
            message,
            config,
            allowed_tools=harness_snapshot.include_tools if harness_snapshot else None,
        ):
            break

    with open(config["rollout_file"], "a") as fout:
        portalocker.lock(fout, portalocker.LOCK_EX)
        fout.write(f"{json.dumps(corpus_tracker)}\n")
        fout.flush()
        portalocker.unlock(fout)


def producer(queue: mp.Queue, config: dict):
    had_queries = set()
    if os.path.exists(config["rollout_file"]):
        with open(config["rollout_file"], "r") as fin:
            portalocker.lock(fin, portalocker.LOCK_EX)
            for line in fin:
                jsonobj = json.loads(line.strip())
                query = jsonobj[0]["extra_info"]["query"]
                had_queries.add(query)
            portalocker.unlock(fin)

    with open(config["synthesize_file"], "r") as fin:
        total = sum(1 for _ in fin)
    pbar = tqdm(total=total - len(had_queries), desc="Start rolling out the remaining queries: ")
    with open(config["synthesize_file"], "r") as fin:
        for line in fin:
            jsonobj = json.loads(line.strip())
            query = jsonobj["query"]
            if query in had_queries:
                continue
            queue.put(query)
            had_queries.add(query)
            pbar.update(1)
            #print(f"Put query: {query}")
    queue.put(None)


def consumer(queue: mp.Queue, config: dict):
    while True:
        query = queue.get()
        if query is None:
            queue.put(None)
            break
        #print(f"Get query: {query}")
        react_loop(query, config)


def rollout(config: dict):
    queue = mp.Queue(config["threads"])

    # Create processes
    producer_process = mp.Process(target=producer, args=(queue, config))
    consumers = []
    for _ in range(config["threads"]):
        consumers.append(mp.Process(target=consumer, args=(queue, config)))

    # Start processes
    producer_process.start()
    for consumer_process in consumers:
        consumer_process.start()

    # Join processes
    producer_process.join()
    for consumer_process in consumers:
        consumer_process.join() 


if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file, "r") as fin:
        config = json.load(fin)
    if config["task"] not in {"knowledge", "web"}:
        config["exclude_tools"] = config.get("exclude_tools", []) + ["web_search"]
    rollout(config)
