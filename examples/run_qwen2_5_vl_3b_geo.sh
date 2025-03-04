set -x

# 指定 GPU 设备
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS

# 配置 WandB (如果不使用 WandB，可以直接 export WANDB_DISABLED=true)
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_PROJECT=verl_test
export WANDB_API_KEY=19e2bb17296ca54e3b6de27ef184eac2eb7efd5f  # 确保 API Key 正确
export WANDB_RUN_NAME=Qwen-VL2_5-3B-GRPO-$(date +%Y-%m-%d-%H-%M-%S)

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct

# 正确的 Python 运行命令
python3 -m verl.trainer.main \
    config=examples/grpo_example.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=4 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_3b_geo \
    trainer.n_gpus_per_node=4 \
    worker.rollout.gpu_memory_utilization=0.5 \
    worker.rollout.enforce_eager=True
