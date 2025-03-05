set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_PROJECT=verl_test
export WANDB_API_KEY=19e2bb17296ca54e3b6de27ef184eac2eb7efd5f  # 确保 API Key 正确
export WANDB_RUN_NAME=Qwen-VL2_5-7B-GRPO-$(date +%Y-%m-%d-%H-%M-%S)

python3 -m verl.trainer.main \
    config=examples/grpo_example.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_7b_geo \
    trainer.n_gpus_per_node=8
