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

from util.harness_fsm import (  # noqa: E402
    build_compact_harness_user_prompt,
    build_harness_snapshot,
    decision_ready_to_recommend,
    is_duplicate_find_product_in_turn,
    is_repeated_failed_search,
    is_repeated_search,
    previous_shop_voucher_selection_issue,
    shop_voucher_selection_issue,
)
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


def budget_check(call_id: str, payload: dict):
    return (
        {
            "name": "budget_check",
            "parameters": {"product_ids": payload.get("product_ids", []), "voucher": {"type": "none"}, "budget": payload.get("budget", 0)},
            "tool_call_id": call_id,
        },
        {"tool_call_id": call_id, "results": {**payload, "_tool_success": True, "_parse_ok": True}},
    )


def assert_state(history: list[str], expected: str, label: str):
    snapshot = build_harness_snapshot(history)
    assert snapshot.state_name == expected, (
        f"{label}: expected {expected}, got {snapshot.state_name}; state={snapshot.state}"
    )
    return snapshot


def main() -> None:
    parsed = Message.from_string(
        "",
        '<think>x</think><tool_call>[{"name":"find_product","parameters":{"q":"a","page":1}}]</tool_call>'
        '<tool_call>[{"name":"find_product","parameters":{"q":"b","page":1}}]</tool_call>',
    )
    assert parsed.tool_call == []
    assert parsed.format_error == "exactly_one_tool_call_block_required"

    history = [user("Need black wireless earbuds under my voucher budget")]
    assert_state(history, "CANDIDATE_SEARCH", "initial")

    call, obs = find("s0", "black wireless earbuds", [])
    history.append(step([call], [obs]))
    snapshot = assert_state(history, "CANDIDATE_SEARCH", "SEARCH->SEARCH")
    assert snapshot.state == {"failed_searches": [{"q": "black wireless earbuds", "page": 1}]}
    assert is_repeated_failed_search({"q": "black wireless earbuds", "page": 1}, snapshot)
    assert not is_repeated_failed_search({"q": "wireless earbuds", "page": 1}, snapshot)
    compact_prompt = build_compact_harness_user_prompt(snapshot, history)
    assert '"previous_searches"' in compact_prompt
    assert '"q":"black wireless earbuds"' in compact_prompt
    duplicate_call = {"name": "find_product", "parameters": {"q": "wireless earbuds", "page": 1}}
    assert is_duplicate_find_product_in_turn(duplicate_call, [duplicate_call, dict(duplicate_call)])

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
    snapshot = assert_state(history, "CANDIDATE_SEARCH", "first non-empty search stays SEARCH")
    assert len(snapshot.state["candidate_pool"]) == 1
    assert "find_product" in snapshot.include_tools
    compact_prompt = build_compact_harness_user_prompt(snapshot, history)
    assert '"candidate_pool"' in compact_prompt
    assert is_repeated_search({"q": "wireless earbuds", "page": 1}, snapshot)
    assert not is_repeated_search({"q": "wireless earbuds", "page": 2}, snapshot)

    call, obs = find(
        "s2",
        "black wireless earbuds",
        [
            {
                "product_id": "p1b",
                "shop_id": "shop1",
                "title": "Black wireless earbuds bundle",
                "price": 125,
                "service": ["freeShipping"],
            }
        ],
    )
    history.append(step([call], [obs]))
    snapshot = assert_state(history, "CANDIDATE_SELECT", "second non-empty search enters SELECT")
    assert len(snapshot.state["candidate_pool"]) == 2
    assert "budget_check" in snapshot.include_tools
    assert "find_product" not in snapshot.include_tools

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
    assert not decision_ready_to_recommend(snapshot)
    final_only_snapshot = build_harness_snapshot(history)
    final_only_snapshot.include_tools = {"recommend_product", "terminate"}
    compact_prompt = build_compact_harness_user_prompt(final_only_snapshot, history)
    assert '"allowed_tools":["recommend_product","terminate"]' in compact_prompt
    assert "Do not call find_product after a verified within-budget selection." in compact_prompt
    compact_prompt = build_compact_harness_user_prompt(snapshot, history)
    assert '"viewed_products"' in compact_prompt
    assert '"attrs":{"Type":"Earbuds"}' in compact_prompt
    assert '"sku_options":{"Color":["Black"]}' in compact_prompt
    assert '["p1","",120,"shop1",["freeShipping"]' in compact_prompt

    call, obs = find("r0", "cheaper black wireless earbuds", [])
    history.append(step([call], [obs]))
    snapshot = assert_state(history, "DECISION", "DECISION->DECISION")
    assert snapshot.state["failed_retry_searches"] == [
        {"q": "black wireless earbuds", "page": 1},
        {"q": "cheaper black wireless earbuds", "page": 1}
    ]
    assert is_repeated_failed_search({"q": "cheaper black wireless earbuds", "page": 1}, snapshot)

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
    assert decision_ready_to_recommend(snapshot)

    stale_candidate = [user("Need black wireless earbuds")]
    call, obs = find(
        "stq",
        "wireless earbuds",
        [
            {"product_id": "p1", "shop_id": "shop1", "title": "Black earbuds", "price": 120, "service": []},
            {"product_id": "pX", "shop_id": "shopX", "title": "Stale earbuds", "price": 70, "service": []},
        ],
    )
    stale_candidate.append(step([call], [obs]))
    view_call, view_obs = view(
        "stv1",
        "p1",
        [{"product_id": "p1", "sku_options": {}, "attributes": {"Color": "Black"}, "service": []}],
    )
    py_call, py_obs = python_calc(
        "stpy1",
        {
            "product_ids": ["p1"],
            "shop_ids": ["shop1"],
            "total_before_voucher": 120,
            "voucher_used": True,
            "payable_total": 110,
            "budget": 100,
            "within_budget": False,
        },
    )
    stale_candidate.append(step([view_call, py_call], [view_obs, py_obs]))
    call, obs = find(
        "strq",
        "cheaper black earbuds",
        [{"product_id": "p2", "shop_id": "shop2", "title": "Budget black earbuds", "price": 90, "service": []}],
    )
    stale_candidate.append(step([call], [obs]))
    snapshot = assert_state(stale_candidate, "CANDIDATE_SELECT", "retry candidate pool excludes stale unselected candidates")
    assert {item["product_id"] for item in snapshot.state["candidate_pool"]} == {"p1", "p2"}
    view_call, view_obs = view(
        "stvx",
        "pX",
        [{"product_id": "pX", "sku_options": {}, "attributes": {"Color": "Black"}, "service": []}],
    )
    py_call, py_obs = python_calc(
        "stpyx",
        {
            "product_ids": ["pX"],
            "shop_ids": ["shopX"],
            "total_before_voucher": 70,
            "voucher_used": True,
            "payable_total": 70,
            "budget": 100,
            "within_budget": True,
        },
    )
    stale_candidate.append(step([view_call, py_call], [view_obs, py_obs]))
    assert_state(stale_candidate, "CANDIDATE_SELECT", "stale candidate outside active pool is rejected")

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

    balanced = [user("Need shirt and pants")]
    shirt_results = [
        {"product_id": f"shirt{i}", "shop_id": "shopA", "title": f"Blue shirt {i}", "price": 30 + i, "service": []}
        for i in range(5)
    ]
    pants_results = [
        {"product_id": f"pants{i}", "shop_id": "shopB", "title": f"Black pants {i}", "price": 40 + i, "service": []}
        for i in range(5)
    ]
    c1, o1 = find("bal1", "shirt", shirt_results)
    c2, o2 = find("bal2", "pants", pants_results)
    balanced.append(step([c1, c2], [o1, o2]))
    snapshot = assert_state(balanced, "CANDIDATE_SELECT", "balanced candidate truncation setup")
    compact_prompt = build_compact_harness_user_prompt(snapshot, balanced, max_candidates=2)
    assert '"shirt0"' in compact_prompt
    assert '"pants0"' in compact_prompt

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

    budget_tool_history = [user("Need black earbuds")]
    call, obs = find(
        "btq",
        "black earbuds",
        [{"product_id": "p1", "shop_id": "shop1", "title": "Black earbuds", "price": 120, "service": []}],
    )
    budget_tool_history.append(step([call], [obs]))
    view_call, view_obs = view(
        "btv",
        "p1",
        [{"product_id": "p1", "sku_options": {}, "attributes": {"Color": "Black"}, "service": []}],
    )
    budget_call, budget_obs = budget_check(
        "btc",
        {
            "product_ids": ["p1"],
            "shop_ids": ["shop1"],
            "total_before_voucher": 120,
            "voucher_used": False,
            "payable_total": 120,
            "budget": 130,
            "within_budget": True,
        },
    )
    budget_tool_history.append(step([view_call, budget_call], [view_obs, budget_obs]))
    assert_state(budget_tool_history, "DECISION", "SELECT->DECISION with budget_check")

    cross_shop_voucher = [user("Need two products with a shop voucher")]
    call, obs = find(
        "csvq",
        "bundle",
        [
            {"product_id": "p1", "shop_id": "shop1", "title": "Item one", "price": 50, "service": []},
            {"product_id": "p2", "shop_id": "shop2", "title": "Item two", "price": 60, "service": []},
        ],
    )
    cross_shop_voucher.append(step([call], [obs]))
    view_call, view_obs = view(
        "csvv",
        "p1,p2",
        [
            {"product_id": "p1", "sku_options": {}, "attributes": {}, "service": []},
            {"product_id": "p2", "sku_options": {}, "attributes": {}, "service": []},
        ],
    )
    budget_call, budget_obs = budget_check(
        "csvb",
        {
            "product_ids": ["p1", "p2"],
            "shop_ids": ["shop1", "shop2"],
            "total_before_voucher": 110,
            "voucher_used": False,
            "payable_total": 110,
            "budget": 130,
            "within_budget": True,
            "agent_voucher": {"type": "shop_threshold_discount", "threshold": 100, "discount": 20, "scope_shop_id": "shop1"},
        },
    )
    cross_shop_voucher.append(step([view_call, budget_call], [view_obs, budget_obs]))
    snapshot = assert_state(cross_shop_voucher, "DECISION", "cross-shop shop voucher reaches decision with issue")
    issue = shop_voucher_selection_issue(snapshot)
    assert issue and issue["error"] == "shop_voucher_selection_has_multiple_shops"
    assert not decision_ready_to_recommend(snapshot)
    compact_prompt = build_compact_harness_user_prompt(snapshot, cross_shop_voucher)
    assert '"selection_issues"' in compact_prompt
    assert '"required_action":"find_product"' in compact_prompt
    retry_call, retry_obs = find(
        "csvr",
        "replacement",
        [
            {"product_id": "p3", "shop_id": "shop1", "title": "Replacement one", "price": 40, "service": ["COD"]},
            {"product_id": "p4", "shop_id": "shop1", "title": "Replacement two", "price": 45, "service": ["freeShipping"]},
            {"product_id": "p5", "shop_id": "shop2", "title": "Other shop", "price": 35, "service": []},
        ],
    )
    cross_shop_voucher.append(step([retry_call], [retry_obs]))
    snapshot = assert_state(cross_shop_voucher, "CANDIDATE_SELECT", "shop voucher retry returns to select")
    issue = previous_shop_voucher_selection_issue(snapshot)
    assert issue and issue["source"] == "previous_decision"
    snapshot.include_tools.add("find_product")
    compact_prompt = build_compact_harness_user_prompt(snapshot, cross_shop_voucher, max_candidates=2)
    assert '"selection_issues"' in compact_prompt
    assert '"candidate_shop_groups"' in compact_prompt
    assert '"previous_searches"' in compact_prompt

    legacy_disabled = [user("Need black earbuds")]
    call, obs = find(
        "ldq",
        "black earbuds",
        [{"product_id": "p1", "shop_id": "shop1", "title": "Black earbuds", "price": 120, "service": []}],
    )
    legacy_disabled.append(step([call], [obs]))
    view_call, view_obs = view(
        "ldv",
        "p1",
        [{"product_id": "p1", "sku_options": {}, "attributes": {"Color": "Black"}, "service": []}],
    )
    py_call, py_obs = python_calc(
        "ldpy",
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
    legacy_disabled.append(step([view_call, py_call], [view_obs, py_obs]))
    snapshot = build_harness_snapshot(legacy_disabled, allow_legacy_python_budget=False)
    assert snapshot.state_name == "CANDIDATE_SELECT", "legacy python budget must be ignored when disabled"

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

    wrong_budget_total = [user("Need black earbuds")]
    call, obs = find(
        "wbtq",
        "black earbuds",
        [{"product_id": "p1", "shop_id": "shop1", "title": "Black earbuds", "price": 120, "service": []}],
    )
    wrong_budget_total.append(step([call], [obs]))
    view_call, view_obs = view(
        "wbtv",
        "p1",
        [{"product_id": "p1", "sku_options": {}, "attributes": {"Color": "Black"}, "service": []}],
    )
    py_call, py_obs = python_calc(
        "wbtpy",
        {
            "product_ids": ["p1"],
            "shop_ids": ["shop1"],
            "total_before_voucher": 1,
            "voucher_used": True,
            "payable_total": 1,
            "budget": 100,
            "within_budget": True,
        },
    )
    wrong_budget_total.append(step([view_call, py_call], [view_obs, py_obs]))
    assert_state(wrong_budget_total, "CANDIDATE_SELECT", "budget total must match candidate evidence")

    duplicate_budget_ids = [user("Need black earbuds")]
    call, obs = find(
        "dbiq",
        "black earbuds",
        [{"product_id": "p1", "shop_id": "shop1", "title": "Black earbuds", "price": 120, "service": []}],
    )
    duplicate_budget_ids.append(step([call], [obs]))
    view_call, view_obs = view(
        "dbiv",
        "p1",
        [{"product_id": "p1", "sku_options": {}, "attributes": {"Color": "Black"}, "service": []}],
    )
    py_call, py_obs = python_calc(
        "dbipy",
        {
            "product_ids": ["p1", "p1"],
            "shop_ids": ["shop1", "shop1"],
            "total_before_voucher": 240,
            "voucher_used": True,
            "payable_total": 200,
            "budget": 220,
            "within_budget": True,
        },
    )
    duplicate_budget_ids.append(step([view_call, py_call], [view_obs, py_obs]))
    assert_state(duplicate_budget_ids, "CANDIDATE_SELECT", "duplicate budget product ids are rejected")

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

    split_view_across_turns = [user("Need two earbuds")]
    call, obs = find(
        "svatq",
        "earbuds",
        [
            {"product_id": "p1", "shop_id": "shop1", "title": "Black earbuds", "price": 120, "service": []},
            {"product_id": "p2", "shop_id": "shop1", "title": "Backup earbuds", "price": 80, "service": []},
        ],
    )
    split_view_across_turns.append(step([call], [obs]))
    view_call, view_obs = view(
        "svatv1",
        "p1",
        [{"product_id": "p1", "sku_options": {}, "attributes": {"Color": "Black"}, "service": []}],
    )
    split_view_across_turns.append(step([view_call], [view_obs]))
    view_call, view_obs = view(
        "svatv2",
        "p2",
        [{"product_id": "p2", "sku_options": {}, "attributes": {"Color": "Black"}, "service": []}],
    )
    py_call, py_obs = python_calc(
        "svatpy",
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
    split_view_across_turns.append(step([view_call, py_call], [view_obs, py_obs]))
    assert_state(split_view_across_turns, "DECISION", "split view calls across turns are combined when needed")

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

    arbitrary_id = [user("Need earbuds")]
    call, obs = find(
        "aiq",
        "earbuds",
        [{"product_id": "p1", "shop_id": "shop1", "title": "Black earbuds", "price": 120, "service": []}],
    )
    arbitrary_id.append(step([call], [obs]))
    view_call, view_obs = view(
        "aiv",
        "p999",
        [{"product_id": "p999", "sku_options": {}, "attributes": {}, "service": []}],
    )
    py_call, py_obs = python_calc(
        "aipy",
        {
            "product_ids": ["p999"],
            "shop_ids": ["shopX"],
            "total_before_voucher": 100,
            "voucher_used": True,
            "payable_total": 90,
            "budget": 110,
            "within_budget": True,
        },
    )
    arbitrary_id.append(step([view_call, py_call], [view_obs, py_obs]))
    assert_state(arbitrary_id, "CANDIDATE_SELECT", "selected ids must come from candidate_pool")

    mixed_search = [user("Need earbuds and charger")]
    c1, o1 = find("mix1", "earbuds", [{"product_id": "p1", "shop_id": "shop1", "title": "Earbuds", "price": 120, "service": []}])
    c2, o2 = find("mix2", "charger", [])
    mixed_search.append(step([c1, c2], [o1, o2]))
    view_call, view_obs = view(
        "mixv",
        "p1",
        [{"product_id": "p1", "sku_options": {}, "attributes": {}, "service": []}],
    )
    py_call, py_obs = python_calc(
        "mixpy",
        {
            "product_ids": ["p1"],
            "shop_ids": ["shop1"],
            "total_before_voucher": 120,
            "voucher_used": True,
            "payable_total": 100,
            "budget": 130,
            "within_budget": True,
        },
    )
    mixed_search.append(step([view_call, py_call], [view_obs, py_obs]))
    snapshot = assert_state(mixed_search, "DECISION", "mixed search empty is carried into decision")
    assert snapshot.state["failed_retry_searches"] == [{"q": "charger", "page": 1}]

    malformed = [user("Need earbuds")]
    call = {"name": "find_product", "parameters": {"q": "earbuds", "page": 1}, "tool_call_id": "bad"}
    malformed.append(step([call], [{"tool_call_id": "bad", "results": {"error": "timeout"}}]))
    snapshot = assert_state(malformed, "CANDIDATE_SEARCH", "malformed find is not empty search")
    assert snapshot.state == {}

    format_feedback = [user("Need earbuds")]
    format_feedback.append(
        Message(
            obs=[
                {
                    "tool_call_id": "format_error",
                    "results": {
                        "tool": "format",
                        "error": "tool_call_items_require_exactly_name_and_parameters",
                        "required_format": '<think>brief reasoning</think><tool_call>[{"name":"allowed_tool","parameters":{}}]</tool_call>',
                    },
                }
            ]
        ).to_string(["obs"])
    )
    snapshot = assert_state(format_feedback, "CANDIDATE_SEARCH", "format errors do not advance FSM")
    compact_prompt = build_compact_harness_user_prompt(snapshot, format_feedback)
    assert '"last_errors"' in compact_prompt
    assert "tool_call_items_require_exactly_name_and_parameters" in compact_prompt
    assert "required_format" in compact_prompt

    print("harness FSM transition checks passed")


if __name__ == "__main__":
    main()
