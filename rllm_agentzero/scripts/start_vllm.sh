#!/bin/bash
# vLLM 服务器启动脚本
# 用于为 rllm_agentzero 提供模型推理服务

# 模型配置
MODEL_PATH="/root/autodl-tmp/models/qwen/Qwen2___5-3B-Instruct"  # ModelScope 下载路径
# 如果模型不存在，先下载
if [ ! -d "$MODEL_PATH" ]; then
    echo "模型不存在，正在从 ModelScope 下载..."
    python -c "
from modelscope.hub.snapshot_download import snapshot_download
model_path = snapshot_download('qwen/Qwen2.5-3B-Instruct', cache_dir='/root/autodl-tmp/models')
print(f'Downloaded to: {model_path}')
"
    MODEL_PATH=$(find /root/autodl-tmp/models -type d -name "*Qwen2*3B*" | head -1)
fi

echo "使用模型: $MODEL_PATH"

# 启动 vLLM 服务器
vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --trust-remote-code \
    --api-key "token-abc123"

# 服务启动后，API 地址为: http://localhost:8000/v1
# 可以通过以下方式测试:
# curl http://localhost:8000/v1/models
