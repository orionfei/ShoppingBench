import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


MODULE_PATH = Path(__file__).with_name("reward_shoppingbench_asr_batch.py")
SPEC = importlib.util.spec_from_file_location("reward_shoppingbench_asr_batch_tested", MODULE_PATH)
reward = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(reward)


class FakeEmbedder:
    def __init__(self, similarities=None):
        self.similarities = similarities or {}
        self.prepare_calls = []

    def prepare(self, titles):
        self.prepare_calls.append(list(dict.fromkeys(titles)))

    def similarity(self, left, right):
        return self.similarities.get((left, right), 0.0)


PRODUCTS = {
    "1": {
        "product_id": "1",
        "title": "exact",
        "price": 60,
        "shop_id": "A",
        "service": ["free shipping"],
        "sku_options": {"sku1": {"color": "red", "size": "M"}},
        "attributes": {"material": ["cotton"]},
    },
    "2": {"product_id": "2", "title": "semantic shirt", "price": 50, "shop_id": "A", "service": ["free shipping"]},
    "3": {"product_id": "3", "title": "second", "price": 50, "shop_id": "B", "service": []},
}


def gt(items, voucher=None):
    return json.dumps(
        {
            "reward": items,
            "voucher": voucher
            or {
                "voucher_type": "platform",
                "threshold": 100,
                "discount_type": "fixed",
                "face_value": 20,
                "budget": 100,
            },
        }
    )


def message_calls(*calls):
    return {"messages": {"messages": [{"role": "assistant", "content": "", "tool_calls": list(calls)}]}}


def call(name, parameters):
    return {"function": {"name": name, "arguments": json.dumps(parameters)}}


def score(ground_truth, calls, embedder=None):
    return reward.compute_score_batched(
        ["shoppingbench_query"],
        ["ignored when messages exist"],
        [ground_truth],
        [message_calls(*calls)],
        product_cache=PRODUCTS,
        embedder=embedder or FakeEmbedder(),
    )[0]


def test_exact_id_fixed_voucher_and_binary_terminal_gate():
    ground_truth = gt([{"product_id": "1"}, {"product_id": "2"}])
    success = score(
        ground_truth,
        [call("recommend_product", {"product_ids": "1,2"}), call("terminate", {"status": "success"})],
    )
    assert {key: success[key] for key in ("score", "paper_asr", "terminate_success", "terminal_asr", "rule", "budget")} == {
        "score": 1.0, "paper_asr": 1.0, "terminate_success": 1.0,
        "terminal_asr": 1.0, "rule": 1.0, "budget": 1.0,
    }
    assert success["final_success"] == 1.0
    assert {"format", "tool_valid", "protocol", "workflow_valid", "steps", "dense_final_success"} <= success.keys()

    no_terminate = score(ground_truth, [call("recommend_product", {"product_ids": "1,2"})])
    assert no_terminate["paper_asr"] == 1.0
    assert no_terminate["terminal_asr"] == no_terminate["score"] == 0.0


def test_exact_id_fast_path_never_prepares_embeddings():
    embedder = FakeEmbedder()
    result = score(
        gt([{"product_id": "1"}]),
        [call("recommend_product", {"product_ids": "1"}), call("terminate", {"status": "success"})],
        embedder,
    )
    assert result["score"] == 1.0
    assert embedder.prepare_calls == []


def test_semantic_title_price_service_sku_and_attributes_match_official_rule():
    constraints = {
        "product_id": "gold",
        "title": ["red cotton shirt"],
        "price": [{"between": [55, 65]}],
        "service": ["free shipping"],
        "sku_options": [{"color": "red", "size": "M"}],
        "attributes": [{"material": ["cotton"]}],
    }
    embedder = FakeEmbedder({("exact", "red cotton shirt"): 0.75})
    result = score(
        gt([constraints], {"voucher_type": "platform", "threshold": 0, "discount_type": "fixed", "face_value": 0, "budget": 60}),
        [call("recommend_product", {"product_ids": "1"}), call("terminate", {"status": "success"})],
        embedder,
    )
    assert result["rule"] == 1.0
    assert result["paper_asr"] == result["score"] == 1.0
    assert embedder.prepare_calls == [["exact", "red cotton shirt"]]


def test_any_failed_constraint_makes_paper_asr_zero():
    constraints = {
        "product_id": "gold",
        "title": ["red cotton shirt"],
        "price": [{"less than": [0, 59]}],
        "service": ["free shipping"],
        "attributes": [{"material": ["cotton"]}],
    }
    embedder = FakeEmbedder({("exact", "red cotton shirt"): 0.8})
    result = score(
        gt([constraints]),
        [call("recommend_product", {"product_ids": "1"}), call("terminate", {"status": "success"})],
        embedder,
    )
    assert result["rule"] == pytest.approx(0.75)
    assert result["paper_asr"] == result["terminal_asr"] == result["score"] == 0.0


@pytest.mark.parametrize(
    ("voucher", "ids", "expected"),
    [
        ({"voucher_type": "platform", "threshold": 100, "discount_type": "fixed", "face_value": 20, "budget": 100}, "1,2", 1.0),
        ({"voucher_type": "platform", "threshold": 111, "discount_type": "fixed", "face_value": 20, "budget": 100}, "1,2", 0.0),
        ({"voucher_type": "platform", "threshold": 100, "discount_type": "percentage", "discount": 0.2, "cap": 15, "budget": 95}, "1,2", 1.0),
        ({"voucher_type": "shop", "threshold": 100, "discount_type": "fixed", "face_value": 20, "budget": 100}, "1,2", 1.0),
        ({"voucher_type": "shop", "threshold": 100, "discount_type": "fixed", "face_value": 20, "budget": 100}, "1,3", 0.0),
    ],
)
def test_budget_platform_shop_threshold_fixed_percentage_cap_and_cross_shop(voucher, ids, expected):
    result = score(
        gt([{"product_id": ids.split(",")[0]}, {"product_id": ids.split(",")[1]}], voucher),
        [call("recommend_product", {"product_ids": ids}), call("terminate", {"status": "success"})],
    )
    assert result["rule"] == 1.0
    assert result["budget"] == expected
    assert result["paper_asr"] == expected


def test_missing_product_malformed_call_and_failed_terminate_are_zero():
    ground_truth = gt([{"product_id": "1"}, {"product_id": "2"}])
    missing = score(ground_truth, [call("recommend_product", {"product_ids": "1,999"}), call("terminate", {"status": "success"})])
    malformed = reward.compute_score_batched(
        ["shoppingbench_query"],
        ["<tool_call>{bad json}</tool_call>"],
        [ground_truth],
        [{}],
        product_cache=PRODUCTS,
        embedder=FakeEmbedder(),
    )[0]
    failed = score(ground_truth, [call("recommend_product", {"product_ids": "1,2"}), call("terminate", {"status": "failure"})])
    assert missing["rule"] == 0.5 and missing["budget"] == 0.0 and missing["score"] == 0.0
    assert malformed["paper_asr"] == malformed["terminate_success"] == malformed["score"] == 0.0
    assert malformed["json_decode_failure"] is True
    assert failed["paper_asr"] == 1.0 and failed["terminate_success"] == failed["score"] == 0.0


def test_batch_unique_title_encoding_and_last_recommendation_semantics():
    embedder = FakeEmbedder({("semantic shirt", "shirt"): 0.6})
    ground_truth = gt([{"product_id": "gold", "title": ["shirt"]}], {"voucher_type": "platform", "threshold": 0, "discount_type": "fixed", "face_value": 0, "budget": 50})
    calls = [call("recommend_product", {"product_ids": "999"}), call("recommend_product", {"product_ids": "2"}), call("terminate", {"status": "success"})]
    results = reward.compute_score_batched(
        ["shoppingbench_query", "shoppingbench_query"],
        ["", ""],
        [ground_truth, ground_truth],
        [message_calls(*calls), message_calls(*calls)],
        product_cache=PRODUCTS,
        embedder=embedder,
    )
    assert [item["score"] for item in results] == [1.0, 1.0]
    assert embedder.prepare_calls == [["semantic shirt", "shirt"]]


def test_solution_xml_fallback_and_input_validation():
    solution = '<tool_call>[{"name":"recommend_product","parameters":{"product_ids":"1"}},{"name":"terminate","parameters":{"status":"success"}}]</tool_call>'
    result = reward.compute_score_batched(
        ["shoppingbench_query"], [solution], [gt([{"product_id": "1"}])], [{}], product_cache=PRODUCTS, embedder=FakeEmbedder()
    )[0]
    assert result["score"] == 1.0
    with pytest.raises(ValueError, match="equal lengths"):
        reward.compute_score_batched([], [solution], [], [], product_cache=PRODUCTS)


def test_binary_outcome_grpo_advantages_only_exist_for_mixed_groups():
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    response_mask = torch.ones((24, 2))
    outcomes = torch.tensor([0.0] * 8 + [1.0] * 8 + [0.0, 1.0] * 4)
    token_rewards = torch.zeros((24, 2))
    token_rewards[:, -1] = outcomes
    group_ids = np.asarray(["all_fail"] * 8 + ["all_success"] * 8 + ["mixed"] * 8)
    advantages, _ = compute_grpo_outcome_advantage(token_rewards, response_mask, group_ids)

    assert torch.count_nonzero(advantages[:16]) == 0
    assert torch.all(advantages[16::2] < 0)
    assert torch.all(advantages[17::2] > 0)
