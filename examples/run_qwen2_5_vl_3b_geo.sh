set -x

# 指定使用所有 4 张 A100
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS

# 如果不使用 wandb，则禁用
export WANDB_DISABLED=true

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/grpo_example.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=4 \  # 开启 4 卡并行
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_3b_geo \
    trainer.n_gpus_per_node=4 \  # 使用 4 张 GPU
    worker.rollout.gpu_memory_utilization=0.5 \  # 降低显存占用，减少 OOM
    worker.rollout.enforce_eager=True  # 避免 CUDA Graph 相关问题
