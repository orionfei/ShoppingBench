#!/usr/bin/env python3
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "src" / "agent"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from util.harness_fsm import build_harness_snapshot  # noqa: E402
from util.message import Message  # noqa: E402


def user(query: str) -> str:
    return Message(user=query).to_string(["user"])


def step(calls: list[dict], obs: list[dict]) -> str:
    return Message(think="x", tool_call=calls, obs=obs).to_string(["think", "tool_call", "obs"])


def find(call_id: str, q: str, results: list, page: int = 1, **extra):
    params = {"q": q, "page": page}
    params.update(extra)
    return (
        {"name": "find_product", "parameters": params, "tool_call_id": call_id},
        {"tool_call_id": call_id, "results": results},
    )


def view(call_id: str, product_ids: str, results: list):
    return (
        {
            "name": "view_product_information",
            "parameters": {"product_ids": product_ids},
            "tool_call_id": call_id,
        },
        {"tool_call_id": call_id, "results": results},
    )


def python_calc(call_id: str, payload: dict):
    return (
        {
            "name": "python_execute",
            "parameters": {"code": "print(...)"},
            "tool_call_id": call_id,
        },
        {"tool_call_id": call_id, "results": {"success": True, "stdout": json.dumps(payload)}},
    )


def assert_state(history: list[str], expected: str, label: str):
    snapshot = build_harness_snapshot(history)
    assert snapshot.state_name == expected, (
        f"{label}: expected {expected}, got {snapshot.state_name}; state={snapshot.state}"
    )
    return snapshot


def main() -> None:
    history = [user("Need black wireless earbuds under my voucher budget")]
    assert_state(history, "CANDIDATE_SEARCH", "initial")

    call, obs = find("s0", "black wireless earbuds", [])
    history.append(step([call], [obs]))
    snapshot = assert_state(history, "CANDIDATE_SEARCH", "SEARCH->SEARCH")
    assert snapshot.state == {"failed_searches": [{"q": "black wireless earbuds", "page": 1}]}

    call, obs = find(
        "s1",
        "wireless earbuds",
        [
            {
                "product_id": "p1",
                "shop_id": "shop1",
                "title": "Black wireless earbuds",
                "price": 120,
                "service": ["freeShipping"],
            }
        ],
    )
    history.append(step([call], [obs]))
    snapshot = assert_state(history, "CANDIDATE_SELECT", "SEARCH->SELECT")
    assert len(snapshot.state["candidate_pool"]) == 1

    view_call, view_obs = view(
        "v1",
        "p1",
        [
            {
                "product_id": "p1",
                "sku_options": {"Color": ["Black"]},
                "attributes": {"Type": "Earbuds"},
                "service": ["freeShipping"],
            }
        ],
    )
    partial_history = history + [step([view_call], [view_obs])]
    assert_state(partial_history, "CANDIDATE_SELECT", "partial SELECT stays SELECT")

    py_call, py_obs = python_calc(
        "pcalc",
        {
            "product_ids": ["p1"],
            "shop_ids": ["shop1"],
            "total_before_voucher": 120,
            "meets_threshold": True,
            "eligible_scope": True,
            "voucher_used": True,
            "payable_total": 100,
            "budget": 110,
            "within_budget": True,
        },
    )
    history.append(step([view_call, py_call], [view_obs, py_obs]))
    snapshot = assert_state(history, "DECISION", "SELECT->DECISION")
    assert snapshot.state["budget_calculation"]["within_budget"] is True

    call, obs = find("r0", "cheaper black wireless earbuds", [])
    history.append(step([call], [obs]))
    snapshot = assert_state(history, "DECISION", "DECISION->DECISION")
    assert snapshot.state["failed_retry_searches"] == [
        {"q": "black wireless earbuds", "page": 1},
        {"q": "cheaper black wireless earbuds", "page": 1}
    ]

    call, obs = find(
        "r1",
        "black earbuds under 100",
        [
            {
                "product_id": "p2",
                "shop_id": "shop2",
                "title": "Budget black earbuds",
                "price": 90,
                "service": [],
            }
        ],
    )
    history.append(step([call], [obs]))
    snapshot = assert_state(history, "CANDIDATE_SELECT", "DECISION->SELECT")
    assert {item["product_id"] for item in snapshot.state["candidate_pool"]} == {"p1", "p2"}
    assert snapshot.state["previous_decision"]["selected_products"][0]["product_id"] == "p1"

    view_call, view_obs = view(
        "v2",
        "p2",
        [{"product_id": "p2", "sku_options": {"Color": ["Black"]}, "attributes": {"Type": "Earbuds"}, "service": []}],
    )
    py_call, py_obs = python_calc(
        "pcalc2",
        {
            "product_ids": ["p2"],
            "shop_ids": ["shop2"],
            "total_before_voucher": 90,
            "voucher_used": True,
            "payable_total": 90,
            "budget": 110,
            "within_budget": True,
        },
    )
    history.append(step([view_call, py_call], [view_obs, py_obs]))
    snapshot = assert_state(history, "DECISION", "retry failure preserved after new check")
    assert snapshot.state["failed_retry_searches"] == [
        {"q": "black wireless earbuds", "page": 1},
        {"q": "cheaper black wireless earbuds", "page": 1}
    ]

    multi = [user("Need a shirt and pants")]
    call, obs = find(
        "m1",
        "shirt",
        [{"product_id": "shirt1", "shop_id": "shopA", "title": "Blue shirt", "price": 30, "service": []}],
    )
    multi.append(step([call], [obs]))
    call, obs = find(
        "m2",
        "pants",
        [{"product_id": "pants1", "shop_id": "shopB", "title": "Black pants", "price": 40, "service": []}],
    )
    multi.append(step([call], [obs]))
    snapshot = assert_state(multi, "CANDIDATE_SELECT", "multiple non-empty search batches")
    assert {item["product_id"] for item in snapshot.state["candidate_pool"]} == {"shirt1", "pants1"}

    seq = [user("Need black earbuds")]
    call, obs = find(
        "sq",
        "black earbuds",
        [{"product_id": "p1", "shop_id": "shop1", "title": "Black earbuds", "price": 120, "service": []}],
    )
    seq.append(step([call], [obs]))
    view_call, view_obs = view(
        "sv",
        "p1",
        [{"product_id": "p1", "sku_options": {}, "attributes": {"Color": "Black"}, "service": []}],
    )
    seq.append(step([view_call], [view_obs]))
    py_call, py_obs = python_calc(
        "spy",
        {
            "product_ids": ["p1"],
            "shop_ids": ["shop1"],
            "total_before_voucher": 120,
            "voucher_used": True,
            "payable_total": 100,
            "budget": 110,
            "within_budget": True,
        },
    )
    seq.append(step([py_call], [py_obs]))
    assert_state(seq, "DECISION", "sequential SELECT view then python")

    mismatch = [user("Need black earbuds")]
    call, obs = find(
        "mq",
        "black earbuds",
        [
            {"product_id": "p1", "shop_id": "shop1", "title": "Black earbuds", "price": 120, "service": []},
            {"product_id": "p2", "shop_id": "shop2", "title": "White earbuds", "price": 80, "service": []},
        ],
    )
    mismatch.append(step([call], [obs]))
    view_call, view_obs = view(
        "mv",
        "p1",
        [{"product_id": "p1", "sku_options": {}, "attributes": {"Color": "Black"}, "service": []}],
    )
    py_call, py_obs = python_calc(
        "mpy",
        {"product_ids": ["p2"], "shop_ids": ["shop2"], "payable_total": 80, "budget": 110, "within_budget": True},
    )
    mismatch.append(step([view_call, py_call], [view_obs, py_obs]))
    assert_state(mismatch, "CANDIDATE_SELECT", "mismatched view and budget ids stay SELECT")

    missing_budget_ids = [user("Need black earbuds")]
    call, obs = find(
        "bq",
        "black earbuds",
        [{"product_id": "p1", "shop_id": "shop1", "title": "Black earbuds", "price": 120, "service": []}],
    )
    missing_budget_ids.append(step([call], [obs]))
    view_call, view_obs = view(
        "bv",
        "p1",
        [{"product_id": "p1", "sku_options": {}, "attributes": {"Color": "Black"}, "service": []}],
    )
    py_call, py_obs = python_calc(
        "bpy",
        {"payable_total": 100, "budget": 110, "within_budget": True},
    )
    missing_budget_ids.append(step([view_call, py_call], [view_obs, py_obs]))
    assert_state(missing_budget_ids, "CANDIDATE_SELECT", "budget ids required for DECISION")

    partial_view = [user("Need two earbuds")]
    call, obs = find(
        "pvq",
        "earbuds",
        [
            {"product_id": "p1", "shop_id": "shop1", "title": "Black earbuds", "price": 120, "service": []},
            {"product_id": "p2", "shop_id": "shop1", "title": "Backup earbuds", "price": 80, "service": []},
        ],
    )
    partial_view.append(step([call], [obs]))
    view_call, view_obs = view(
        "pvv",
        "p1,p2",
        [{"product_id": "p1", "sku_options": {}, "attributes": {"Color": "Black"}, "service": []}],
    )
    py_call, py_obs = python_calc(
        "pvpy",
        {
            "product_ids": ["p1", "p2"],
            "shop_ids": ["shop1", "shop1"],
            "total_before_voucher": 200,
            "voucher_used": True,
            "payable_total": 180,
            "budget": 210,
            "within_budget": True,
        },
    )
    partial_view.append(step([view_call, py_call], [view_obs, py_obs]))
    assert_state(partial_view, "CANDIDATE_SELECT", "partial view results stay SELECT")

    reordered_budget = [user("Need two earbuds")]
    call, obs = find(
        "roq",
        "earbuds",
        [
            {"product_id": "p1", "shop_id": "shop1", "title": "Black earbuds", "price": 120, "service": []},
            {"product_id": "p2", "shop_id": "shop1", "title": "Backup earbuds", "price": 80, "service": []},
        ],
    )
    reordered_budget.append(step([call], [obs]))
    view_call, view_obs = view(
        "rov",
        "p1,p2",
        [
            {"product_id": "p1", "sku_options": {}, "attributes": {"Color": "Black"}, "service": []},
            {"product_id": "p2", "sku_options": {}, "attributes": {"Color": "Black"}, "service": []},
        ],
    )
    py_call, py_obs = python_calc(
        "ropy",
        {
            "product_ids": ["p2", "p1"],
            "shop_ids": ["shop1", "shop1"],
            "total_before_voucher": 200,
            "voucher_used": True,
            "payable_total": 180,
            "budget": 210,
            "within_budget": True,
        },
    )
    reordered_budget.append(step([view_call, py_call], [view_obs, py_obs]))
    assert_state(reordered_budget, "DECISION", "budget ids can be reordered")

    split_view = [user("Need two earbuds")]
    call, obs = find(
        "svq2",
        "earbuds",
        [
            {"product_id": "p1", "shop_id": "shop1", "title": "Black earbuds", "price": 120, "service": []},
            {"product_id": "p2", "shop_id": "shop1", "title": "Backup earbuds", "price": 80, "service": []},
        ],
    )
    split_view.append(step([call], [obs]))
    view_call_1, view_obs_1 = view(
        "svv1",
        "p1",
        [{"product_id": "p1", "sku_options": {}, "attributes": {"Color": "Black"}, "service": []}],
    )
    view_call_2, view_obs_2 = view(
        "svv2",
        "p2",
        [{"product_id": "p2", "sku_options": {}, "attributes": {"Color": "Black"}, "service": []}],
    )
    py_call, py_obs = python_calc(
        "svpy2",
        {
            "product_ids": ["p1", "p2"],
            "shop_ids": ["shop1", "shop1"],
            "total_before_voucher": 200,
            "voucher_used": True,
            "payable_total": 180,
            "budget": 210,
            "within_budget": True,
        },
    )
    split_view.append(step([view_call_1, view_call_2, py_call], [view_obs_1, view_obs_2, py_obs]))
    assert_state(split_view, "DECISION", "split view calls are combined")

    changed_selection = [user("Need earbuds")]
    call, obs = find(
        "csq",
        "earbuds",
        [
            {"product_id": "p1", "shop_id": "shop1", "title": "Black earbuds", "price": 120, "service": []},
            {"product_id": "p2", "shop_id": "shop2", "title": "Cheaper earbuds", "price": 80, "service": []},
        ],
    )
    changed_selection.append(step([call], [obs]))
    view_call, view_obs = view(
        "csv1",
        "p1",
        [{"product_id": "p1", "sku_options": {}, "attributes": {"Color": "Black"}, "service": []}],
    )
    changed_selection.append(step([view_call], [view_obs]))
    view_call, view_obs = view(
        "csv2",
        "p2",
        [{"product_id": "p2", "sku_options": {}, "attributes": {"Color": "Black"}, "service": []}],
    )
    py_call, py_obs = python_calc(
        "cspy",
        {
            "product_ids": ["p2"],
            "shop_ids": ["shop2"],
            "total_before_voucher": 80,
            "voucher_used": True,
            "payable_total": 70,
            "budget": 100,
            "within_budget": True,
        },
    )
    changed_selection.append(step([view_call, py_call], [view_obs, py_obs]))
    snapshot = assert_state(changed_selection, "DECISION", "later view selection supersedes stale partial view")
    assert snapshot.state["view_requested_product_ids"] == ["p2"]

    malformed = [user("Need earbuds")]
    call = {"name": "find_product", "parameters": {"q": "earbuds", "page": 1}, "tool_call_id": "bad"}
    malformed.append(step([call], [{"tool_call_id": "bad", "results": {"error": "timeout"}}]))
    snapshot = assert_state(malformed, "CANDIDATE_SEARCH", "malformed find is not empty search")
    assert snapshot.state == {}

    print("harness FSM transition checks passed")


if __name__ == "__main__":
    main()
