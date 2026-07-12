# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import hashlib
import heapq
import json
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig
from pydantic import BaseModel
from tensordict import TensorDict
from transformers import AutoTokenizer

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.workers.rollout.async_server import async_server_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _stable_sampling_config(config: DictConfig):
    agent_config = config.actor_rollout_ref.rollout.agent
    return agent_config.get("stable_sampling", {}) or {}


def _stable_sampling_enabled(config: DictConfig) -> bool:
    return _as_bool(_stable_sampling_config(config).get("enabled", False))


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _stable_fingerprint(value: Any) -> str:
    payload = json.dumps(_json_ready(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        stable_config = _stable_sampling_config(config)
        self.stable_sampling_enabled = _stable_sampling_enabled(config)
        self.stable_server_routing = self.stable_sampling_enabled and _as_bool(
            stable_config.get("stable_server_routing", True)
        )
        if not (
            self.stable_sampling_enabled
            and _as_bool(stable_config.get("stable_server_order", True))
        ):
            random.shuffle(self.server_handles)

        # Least requests load balancing
        tie_breaker = enumerate(self.server_handles) if self.stable_sampling_enabled else (
            (hash(server), server) for server in server_handles
        )
        self.weighted_serveres = [[0, (idx, server)] for idx, server in tie_breaker]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        if self.stable_server_routing:
            server_idx = int(hashlib.sha1(str(request_id).encode("utf-8")).hexdigest()[:8], 16) % len(
                self.server_handles
            )
            server = self.server_handles[server_idx]
        else:
            server = self.weighted_serveres[0][1][1]
            self.weighted_serveres[0][0] += 1
            heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> list[int]:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            List[int]: List of generated token ids.
        """
        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        return output


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    num_turns: int = 0
    metrics: AgentLoopMetrics
    messages: list[dict[str, Any]] | None = None


class AgentLoopBase(ABC):
    """An agent loop takes a input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    _class_initialized = False

    def __init__(self, config: DictConfig, server_manager: AsyncLLMServerManager, tokenizer: AutoTokenizer):
        """Initialize agent loop.

        Args:
            config (DictConfig): YAML config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        """
        self.config = config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.loop = asyncio.get_running_loop()
        self.init_class(config, tokenizer)

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        """Initialize class state shared across all instances."""
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any] | None = None,
    ) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            messages (List[Dict[str, Any]]): Input messages.
            sampling_params (Dict[str, Any]): LLM sampling params.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


@ray.remote
class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle]):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
        """
        self.config = config
        self.server_manager = AsyncLLMServerManager(config, server_handles)

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        trace_config = config.trainer.get("rollout_trace", {})

        RolloutTraceConfig.init(
            config.trainer.project_name,
            config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        tasks = []
        agent_names = batch.non_tensor_batch["agent_name"]
        raw_prompts = batch.non_tensor_batch["raw_prompt"]
        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(raw_prompts))

        trajectory_info = await get_trajectory_info(batch.meta_info.get("global_steps", -1), index, batch)

        for agent_name, messages, trajectory in zip(agent_names, raw_prompts, trajectory_info, strict=True):
            tasks.append(
                asyncio.create_task(self._run_agent_loop(agent_name, messages.tolist(), sampling_params, trajectory))
            )
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(outputs)
        return output

    async def _run_agent_loop(
        self,
        agent_name: str,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
    ) -> AgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"], sample_index=trajectory["sample_index"], rollout_n=trajectory["rollout_n"]
        ):
            agent_loop_class = self.get_agent_loop_class(agent_name)
            agent_loop = agent_loop_class(self.config, self.server_manager, self.tokenizer)
            trajectory_sampling_params = dict(sampling_params)
            stable_config = _stable_sampling_config(self.config)
            if (
                _stable_sampling_enabled(self.config)
                and _as_bool(stable_config.get("seed_requests", True))
                and self.config.actor_rollout_ref.rollout.name == "vllm"
                and trajectory.get("rollout_seed") is not None
            ):
                trajectory_sampling_params["seed"] = int(trajectory["rollout_seed"])
            output = await agent_loop.run(messages, trajectory_sampling_params, trajectory=trajectory)
            return output

    def get_agent_loop_class(self, agent_name: str) -> type[AgentLoopBase]:
        """Get the appropriate agent loop class based on agent name.

        Factory method that returns the correct agent loop class implementation
        for the specified agent type.

        Args:
            agent_name (str): Name of the agent type ('single_turn_agent' or 'tool_agent').

        Returns:
            Type[AgentLoopBase]: Agent loop class corresponding to the agent name.

        Raises:
            ValueError: If the agent_name is not recognized.
        """
        # TODO: add tool agent registrary
        from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
        from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

        if agent_name == "single_turn_agent":
            return SingleTurnAgentLoop
        elif agent_name == "tool_agent":
            return ToolAgentLoop
        raise ValueError(f"Unknown agent_name: {agent_name}")

    def _postprocess(self, inputs: list[AgentLoopOutput]) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts
        self.tokenizer.padding_side = "left"
        outputs = self.tokenizer.pad(
            [{"input_ids": input.prompt_ids} for input in inputs],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids, prompt_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # responses
        self.tokenizer.padding_side = "right"
        outputs = self.tokenizer.pad(
            [{"input_ids": input.response_ids} for input in inputs],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # response_mask
        outputs = self.tokenizer.pad(
            [{"input_ids": input.response_mask} for input in inputs],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        response_mask = outputs["input_ids"]
        assert response_ids.shape == response_mask.shape, (
            f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"
        )
        response_mask = response_mask * response_attention_mask

        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )

        num_turns = np.array([input.num_turns for input in inputs], dtype=np.int32)
        metrics = [input.metrics.model_dump() for input in inputs]
        non_tensor_batch = {"__num_turns__": num_turns}
        if any(input.messages is not None for input in inputs):
            messages = np.empty(len(inputs), dtype=object)
            for idx, input in enumerate(inputs):
                messages[idx] = input.messages or []
            non_tensor_batch["messages"] = messages
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"metrics": metrics})


async def get_trajectory_info(step, index, batch: DataProto | None = None):
    """Get the trajectory info (step, sample_index, rollout_n) asynchrously"""
    if batch is not None and "__trajectory_rollout_n__" in batch.non_tensor_batch:
        sample_index = batch.non_tensor_batch.get("__trajectory_sample_index__", index)
        rollout_n = batch.non_tensor_batch["__trajectory_rollout_n__"]
        request_id = batch.non_tensor_batch.get("__trajectory_request_id__")
        rollout_seed = batch.non_tensor_batch.get("__trajectory_seed__")
        trajectory_info = []
        for i in range(len(rollout_n)):
            item = {"step": step, "sample_index": sample_index[i], "rollout_n": int(rollout_n[i])}
            if request_id is not None:
                item["request_id"] = str(request_id[i])
            if rollout_seed is not None:
                item["rollout_seed"] = int(rollout_seed[i])
            trajectory_info.append(item)
        return trajectory_info

    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group.
        """
        self.config = config
        self.worker_group = worker_group

        self._initialize_llm_servers()
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        self.sleep()

    def _initialize_llm_servers(self):
        self.rollout_tp_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        register_center = ray.get_actor(f"{self.worker_group.name_prefix}_register_center")
        workers_info = ray.get(register_center.get_worker_info.remote())
        assert len(workers_info) == self.worker_group.world_size

        self.async_llm_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size

        if self.config.actor_rollout_ref.rollout.agent.custom_async_server:
            server_class = async_server_class(
                rollout_backend=self.config.actor_rollout_ref.rollout.name,
                rollout_backend_module=self.config.actor_rollout_ref.rollout.agent.custom_async_server.path,
                rollout_backend_class=self.config.actor_rollout_ref.rollout.agent.custom_async_server.name,
            )
        else:
            server_class = async_server_class(rollout_backend=self.config.actor_rollout_ref.rollout.name)

        # Start all server instances, restart if address already in use.
        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: server_class.options(
                    # make sure AsyncvLLMServer colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(self.config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
                for rollout_dp_rank in unready_dp_ranks
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    address = ray.get(server.get_server_address.remote())
                    self.server_addresses[rollout_dp_rank] = address
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # All server instances are ready, init AsyncLLM engine.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        for i in range(self.config.actor_rollout_ref.rollout.agent.num_workers):
            self.agent_loop_workers.append(
                AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                ).remote(self.config, self.async_llm_servers)
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        if _stable_sampling_enabled(self.config):
            self._annotate_stable_sampling(prompts)
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        # calculate performance metrics
        metrics = [output.meta_info["metrics"] for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing}
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def wake_up(self):
        """Wake up all rollout server instances."""
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all rollout server instances."""
        ray.get([server.sleep.remote() for server in self.async_llm_servers])

    def _annotate_stable_sampling(self, prompts: DataProto):
        stable_config = _stable_sampling_config(self.config)
        seed_base = int(stable_config.get("seed_base", 0))
        rollout_offset_base = int(stable_config.get("rollout_offset_base", 0))
        step = int(prompts.meta_info.get("global_steps", -1))

        source_values = None
        for key in ("raw_prompt_ids", "raw_prompt", "full_prompts", "index"):
            if key in prompts.non_tensor_batch:
                source_values = prompts.non_tensor_batch[key]
                break
        if source_values is None:
            source_values = np.arange(len(prompts), dtype=object)

        seen: dict[str, int] = {}
        counters: dict[str, int] = {}
        sample_indices = []
        rollout_ns = []
        rollout_seeds = []
        request_ids = []

        for value in source_values:
            fingerprint = _stable_fingerprint(value)
            sample_index = seen.setdefault(fingerprint, len(seen))
            rollout_n_local = counters.get(fingerprint, 0)
            counters[fingerprint] = rollout_n_local + 1
            rollout_n = rollout_offset_base + rollout_n_local

            fp_int = int(fingerprint[:16], 16)
            seed = (seed_base + fp_int + max(step, 0) * 1009 + rollout_n * 1000003) % (2**32)

            sample_indices.append(sample_index)
            rollout_ns.append(rollout_n)
            rollout_seeds.append(seed)
            if _as_bool(stable_config.get("deterministic_request_id", True)):
                request_ids.append(f"stable-{step}-{fingerprint[:16]}-{rollout_n}")
            else:
                request_ids.append("")

        prompts.non_tensor_batch["__trajectory_sample_index__"] = np.array(sample_indices, dtype=object)
        prompts.non_tensor_batch["__trajectory_rollout_n__"] = np.array(rollout_ns, dtype=np.int64)
        prompts.non_tensor_batch["__trajectory_seed__"] = np.array(rollout_seeds, dtype=np.int64)
        prompts.non_tensor_batch["__trajectory_request_id__"] = np.array(request_ids, dtype=object)
