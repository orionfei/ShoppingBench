#!/usr/bin/env python3
import sys
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "src" / "rl" / "verl" / "utils" / "reward_score" / "shoppingbench_query.py"
spec = importlib.util.spec_from_file_location("shoppingbench_query", MODULE_PATH)
assert spec and spec.loader
shoppingbench_query = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = shoppingbench_query
spec.loader.exec_module(shoppingbench_query)
_workflow_validity = shoppingbench_query._workflow_validity
_has_valid_format = shoppingbench_query._has_valid_format
compute_score = shoppingbench_query.compute_score


def assistant(*names: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "name": name,
                "parameters": _parameters_for(name),
            }
            for name in names
        ],
    }


def _parameters_for(name: str) -> dict:
    if name == "find_product":
        return {"q": "black earbuds", "page": 1}
    if name == "view_product_information":
        return {"product_ids": "p1"}
    if name == "python_execute":
        return {"code": "print('{}')"}
    if name == "budget_check":
        return {"product_ids": ["p1"], "voucher": {"type": "none"}, "budget": 100}
    if name == "recommend_product":
        return {"product_ids": "p1"}
    if name == "terminate":
        return {"status": "success"}
    return {}


def assert_valid(messages: list[dict], label: str) -> None:
    assert _workflow_validity(messages) == 1.0, label


def assert_invalid(messages: list[dict], label: str) -> None:
    assert _workflow_validity(messages) == 0.0, label


def main() -> None:
    cache_path = ROOT / "data" / "tmp" / "test_reward_empty_product_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{}\n", encoding="utf-8")
    shoppingbench_query._product_cache.cache_clear()
    shoppingbench_query.os.environ["SHOPPINGBENCH_PRODUCT_CACHE"] = str(cache_path)

    assert_valid(
        [
            assistant("find_product", "find_product"),
            assistant("view_product_information", "python_execute"),
            assistant("recommend_product", "terminate"),
        ],
        "FSM-valid SEARCH, SELECT, DECISION multi-tool turns should be valid",
    )

    assert_valid(
        [
            assistant("find_product"),
            assistant("view_product_information"),
            assistant("python_execute"),
            assistant("find_product", "find_product"),
            assistant("view_product_information", "python_execute"),
            assistant("recommend_product", "terminate"),
        ],
        "retry search after DECISION should be valid",
    )
    assert_valid(
        [
            assistant("find_product"),
            assistant("view_product_information", "budget_check"),
            assistant("recommend_product", "terminate"),
        ],
        "state-local budget_check workflow should be valid",
    )

    assert_invalid(
        [assistant("find_product", "recommend_product")],
        "DECISION search cannot mix with final actions",
    )
    assert_invalid(
        [assistant("find_product"), assistant("terminate")],
        "terminate requires recommend_product in the same turn",
    )
    assert_invalid(
        [
            assistant("find_product"),
            assistant("view_product_information", "python_execute"),
            assistant("recommend_product"),
        ],
        "recommend_product requires terminate in the same turn",
    )
    assert_invalid(
        [assistant("find_product"), assistant("python_execute")],
        "budget calculation requires prior or same-turn view",
    )
    assert_invalid(
        [
            assistant("find_product"),
            assistant("view_product_information", "python_execute"),
            assistant("view_product_information"),
        ],
        "cannot go back to SELECT tools without a retry search",
    )
    assert _has_valid_format("<think>x</think><tool_call>[]</tool_call>")
    assert not _has_valid_format("<think>x</think><tool_call>[]</tool_call><tool_call>[]</tool_call>")
    assert not _has_valid_format("<think>x</think><response>a</response><response>b</response>")
    reward = compute_score(
        "",
        '{"reward":[{"product_id":"p1"}],"voucher":{}}',
        extra_info={
            "messages": [
                {
                    "role": "assistant",
                    "content": '<think>x</think><tool_call>[{"name":"find_product","parameters":{"q":"black earbuds","page":1}}]</tool_call>',
                    "tool_call": [{"name": "find_product", "parameters": {"q": "black earbuds", "page": 1}}],
                    "obs": [{"results": []}],
                }
            ]
        },
    )
    assert reward["structured_failure_mode"] == "search_recall_gap"

    print("ShoppingBench reward workflow validity checks passed")


if __name__ == "__main__":
    main()
