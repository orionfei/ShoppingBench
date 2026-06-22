export PROJECT_NAME="${PROJECT_NAME:-shoppingbench-rl}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_qwen3-1.7b_outcome_n4}"
export PYTHONUNBUFFERED=1

MODEL_PATH=$1
if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <sft_or_probe_selected_model_path>"
    exit 1
fi

QUERY_LEVEL_RL="${QUERY_LEVEL_RL:-0}"
if [ "$QUERY_LEVEL_RL" = "1" ]; then
    TRAIN_FILES="${TRAIN_FILES:-dataset/shoppingbench_query/train.parquet}"
    VAL_FILES="${VAL_FILES:-dataset/shoppingbench_query/test.parquet}"
    SHOPPINGBENCH_PRODUCT_CACHE="${SHOPPINGBENCH_PRODUCT_CACHE:-dataset/shoppingbench_query/product_cache.json}"
    ROLLOUT_MODE="${ROLLOUT_MODE:-async}"
    ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
    MULTI_TURN_ENABLE="${MULTI_TURN_ENABLE:-True}"
    MULTI_TURN_FORMAT="${MULTI_TURN_FORMAT:-shoppingbench_xml}"
    MULTI_TURN_TOOL_CONFIG_PATH="${MULTI_TURN_TOOL_CONFIG_PATH:-config/rl/shoppingbench_tools.yaml}"
    RETURN_RAW_CHAT="${RETURN_RAW_CHAT:-True}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
else
    TRAIN_FILES="${TRAIN_FILES:-dataset/shoppingbench/train.parquet}"
    VAL_FILES="${VAL_FILES:-dataset/shoppingbench/test.parquet}"
    ROLLOUT_MODE="${ROLLOUT_MODE:-sync}"
    ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
    MULTI_TURN_ENABLE="${MULTI_TURN_ENABLE:-False}"
    MULTI_TURN_FORMAT="${MULTI_TURN_FORMAT:-hermes}"
    MULTI_TURN_TOOL_CONFIG_PATH="${MULTI_TURN_TOOL_CONFIG_PATH:-null}"
    RETURN_RAW_CHAT="${RETURN_RAW_CHAT:-False}"
fi
SHOPPINGBENCH_PROTOCOL_WEIGHT_START="${SHOPPINGBENCH_PROTOCOL_WEIGHT_START:-0.2}"
SHOPPINGBENCH_PROTOCOL_ANNEAL_FRACTION="${SHOPPINGBENCH_PROTOCOL_ANNEAL_FRACTION:-0.1}"
SHOPPINGBENCH_PROTOCOL_ANNEAL_STEPS="${SHOPPINGBENCH_PROTOCOL_ANNEAL_STEPS:-0}"
SHOPPINGBENCH_STEP_PENALTY="${SHOPPINGBENCH_STEP_PENALTY:-0.02}"
export SHOPPINGBENCH_PRODUCT_CACHE
export SHOPPINGBENCH_PROTOCOL_WEIGHT_START
export SHOPPINGBENCH_PROTOCOL_ANNEAL_FRACTION
export SHOPPINGBENCH_PROTOCOL_ANNEAL_STEPS
export SHOPPINGBENCH_STEP_PENALTY
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-24576}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.9}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"
NNODES="${NNODES:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
SAVE_FREQ="${SAVE_FREQ:-0.2}"
TEST_FREQ="${TEST_FREQ:-0.05}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-rollouts/${EXPERIMENT_NAME}}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.return_raw_chat=$RETURN_RAW_CHAT \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
    actor_rollout_ref.rollout.mode=$ROLLOUT_MODE \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$ROLLOUT_TOP_P \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.multi_turn.enable=$MULTI_TURN_ENABLE \
    actor_rollout_ref.rollout.multi_turn.format=$MULTI_TURN_FORMAT \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$MULTI_TURN_TOOL_CONFIG_PATH \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.rollout_data_dir=$ROLLOUT_DATA_DIR
