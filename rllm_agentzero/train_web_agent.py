"""
Training Script for rllm_agentzero

Entry point for training web exploration agents using rLLM's verl backend.
"""
# 必须在所有其他 import 之前设置这些环境变量！
# 禁用 uvloop 以允许 Playwright Sync API 与 asyncio 一起工作
import os
os.environ["UVLOOP_ENABLE"] = "0"  # 禁用 uvloop
os.environ["RAY_DISABLE_UVLOOP"] = "1"  # 告诉 Ray 不要使用 uvloop

import logging
from typing import Optional

import hydra
from omegaconf import DictConfig

from rllm.data import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer

from rllm_agentzero.agents.explorer_agent import ExplorerAgent
# 使用 rLLM 的 BrowserGymEnv（已修复兼容性）
from rllm.environments.browsergym.browsergym import BrowserGymEnv
from rllm_agentzero.workflows.exploration_workflow import ExplorationWorkflow
from rllm_agentzero.data.prepare_data import prepare_webarena_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 默认配置
DEFAULT_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_CACHE_DIR = "/root/autodl-tmp/models"


def download_model_from_modelscope(model_id: str, cache_dir: str) -> str:
    """
    从 ModelScope 下载模型
    
    Args:
        model_id: ModelScope 模型ID
        cache_dir: 缓存目录
        
    Returns:
        本地模型路径
    """
    from modelscope.hub.snapshot_download import snapshot_download
    logger.info(f"Downloading model from ModelScope: {model_id}")
    local_path = snapshot_download(model_id, cache_dir=cache_dir)
    logger.info(f"Model downloaded to: {local_path}")
    return local_path


def create_encoder_fn(
    model_name: str = DEFAULT_ENCODER_MODEL,
    model_cache_dir: str = DEFAULT_MODEL_CACHE_DIR,
    use_modelscope: bool = True
):
    """
    Create semantic encoder function for Graph world model.
    
    使用延迟加载模式，避免 Ray 序列化 GPU tensor 问题。
    
    Args:
        model_name: Model name or ModelScope ID
        model_cache_dir: Directory to cache models
        use_modelscope: Whether to use ModelScope for download
        
    Returns:
        Encoder function: obs -> embedding
    """
    logger.info(f"Loading embedding model ({model_name})...")
    
    # 确定模型路径
    local_model_path = None
    
    # 1. 首先检查是否已有本地模型
    if os.path.exists(model_name):
        local_model_path = model_name
        logger.info(f"Using local model at: {local_model_path}")
    elif model_cache_dir:
        # 检查 cache 目录下是否存在
        cached_path = os.path.join(model_cache_dir, model_name.replace("/", "_"))
        if os.path.exists(cached_path):
            local_model_path = cached_path
            logger.info(f"Using cached model at: {local_model_path}")
    
    # 2. 如果本地没有，尝试下载
    if local_model_path is None:
        if use_modelscope:
            try:
                local_model_path = download_model_from_modelscope(model_name, model_cache_dir)
            except Exception as e:
                logger.warning(f"ModelScope download failed: {e}")
                local_model_path = model_name
        else:
            local_model_path = model_name
    
    # 保存路径供延迟加载使用
    _model_path = local_model_path
    logger.info(f"Embedding model path resolved: {_model_path}")
    
    # 创建延迟加载的 encoder 类
    class LazyEncoder:
        """延迟加载的 encoder，避免 Ray 序列化问题"""
        def __init__(self, model_path):
            self.model_path = model_path
            self._model = None
        
        def _load_model(self):
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                import torch
                # 在 worker 中根据 CUDA 可用性选择设备
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = SentenceTransformer(self.model_path, device=device)
                logger.info(f"Lazy loaded model on {device}")
            return self._model
        
        def __call__(self, obs: dict) -> list[float]:
            model = self._load_model()
            url = obs.get("url", "")
            axtree_txt = obs.get("axtree_txt", "")
            
            # Smart truncation for long pages
            max_chars = 8000
            if len(axtree_txt) > max_chars:
                keep_start = int(max_chars * 0.6)
                keep_end = max_chars - keep_start
                axtree_txt = (
                    axtree_txt[:keep_start] + 
                    "\n...[truncated]...\n" + 
                    axtree_txt[-keep_end:]
                )
            
            text = f"URL: {url}\n\nPage Content:\n{axtree_txt}"
            embedding = model.encode(text, convert_to_tensor=False, show_progress_bar=False)
            return embedding.tolist()
    
    encoder = LazyEncoder(_model_path)
    logger.info("Embedding model configured (lazy loading enabled)!")
    
    return encoder


@hydra.main(config_path="configs", config_name="train_config", version_base=None)
def main(config: DictConfig):
    """
    Main training entry point.
    
    Uses Hydra for configuration management.
    """
    logger.info("="*60)
    logger.info("rllm_agentzero Training Script")
    logger.info("="*60)
    
    # Prepare data
    logger.info("Preparing datasets...")
    train_urls = list(config.get("train_urls", ["http://3.148.75.200:7770/"]))
    test_urls = list(config.get("test_urls", train_urls[:1]))
    
    train_dataset, val_dataset = prepare_webarena_data(
        train_urls=train_urls,
        test_urls=test_urls
    )
    
    # Create encoder function
    encoder_fn = create_encoder_fn(config.get("encoder_model", "all-MiniLM-L6-v2"))
    
    # Workflow arguments
    workflow_args = {
        "encoder_fn": encoder_fn,
        "encoder_name": config.get("encoder_model", "all-MiniLM-L6-v2"),
        "similarity_threshold": config.get("similarity_threshold", 0.95),
        "max_steps": config.get("max_steps", 20),
        "max_nodes": config.get("max_nodes", 50),
        "agent_cls": ExplorerAgent,
        "agent_args": {
            "model_id": config.get("model_id", "base-model"),
            "temperature": config.get("temperature", 1.0),
            "max_repeats": config.get("max_repeats", 3),
        },
        # 使用 rLLM 的 BrowserGymEnv（已修复兼容性问题）
        "env_cls": BrowserGymEnv,
        "env_args": {
            "headless": config.get("headless", True),
        },
        "gamma": config.get("gamma", 0.99),
    }
    
    # Create trainer
    logger.info("Initializing AgentTrainer with verl backend...")
    trainer = AgentTrainer(
        workflow_class=ExplorationWorkflow,
        workflow_args=workflow_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        backend="verl",
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed!")


def train_simple(
    base_url: str = "http://3.148.75.200:7770/",
    model_id: str = "base-model",
    max_steps: int = 20,
    headless: bool = True
):
    """
    Simple training function without Hydra configuration.
    
    Useful for quick testing and debugging.
    
    Args:
        base_url: URL to explore
        model_id: Model ID for vLLM server
        max_steps: Maximum steps per episode
        headless: Run browser in headless mode
    """
    logger.info("Simple training mode")
    
    # Prepare data
    train_dataset, val_dataset = prepare_webarena_data(
        train_urls=[base_url],
        test_urls=[base_url]
    )
    
    # Create encoder
    encoder_fn = create_encoder_fn()
    
    # Workflow args
    workflow_args = {
        "encoder_fn": encoder_fn,
        "max_steps": max_steps,
        "agent_args": {"model_id": model_id},
        "env_args": {"headless": headless},
    }
    
    # For simple mode, just test environment setup
    from rllm_agentzero.environments.browsergym_env import WebAgentZeroEnv
    from rllm_agentzero.agents.explorer_agent import ExplorerAgent
    
    # Test environment
    env = WebAgentZeroEnv(task={"url": base_url}, headless=headless)
    obs, info = env.reset()
    logger.info(f"Environment reset successful. URL: {obs.get('url', 'N/A')}")
    
    # Test agent
    agent = ExplorerAgent(model_id=model_id)
    agent.reset()
    agent.update_from_env(obs, reward=0.0, done=False, info={})
    logger.info(f"Agent initialized. Messages: {len(agent.chat_completions)}")
    
    env.close()
    logger.info("Simple test completed successfully!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        # Simple mode for testing
        train_simple()
    else:
        # Full Hydra training
        main()
