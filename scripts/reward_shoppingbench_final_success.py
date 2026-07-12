"""Outcome-only ShoppingBench reward for VERL custom_reward_function.

This wrapper keeps the existing ShoppingBench reward parser and diagnostics,
but uses the paper-style final success / ASR signal as the scalar training
reward. It is intentionally separate from the default reward implementation so
the dense reward path remains available by removing the custom reward override.
"""

from verl.utils.reward_score import shoppingbench_query


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source != "shoppingbench_query":
        raise NotImplementedError(f"Unsupported data_source for outcome-only reward: {data_source!r}")

    result = shoppingbench_query.compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
    if not isinstance(result, dict):
        final_success = float(result)
        return {"score": final_success, "final_success": final_success, "outcome_reward": final_success}

    final_success = float(result.get("final_success", result.get("success", 0.0)) or 0.0)
    result = dict(result)
    result["score"] = final_success
    result["outcome_reward"] = final_success
    return result
