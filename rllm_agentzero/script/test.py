import logging
import os
import sys
import json
from typing import Optional
from dataclasses import dataclass, field
import gymnasium as gym
from browsergym.core.env import BrowserEnv
from browsergym.utils.obs import flatten_axtree_to_str

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sentence_transformers import SentenceTransformer
from rllm_agentzero.agents.explorer_agent import ExplorerAgent
from rllm_agentzero.core.graph import Graph
from rllm_agentzero.core.node import Node
from rllm_agentzero.core.element_utils import extract_interactive_elements
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ExplorerTestConfig:
    """Configuration for Explorer testing"""
    target_url: str
    max_steps: int = 20
    max_nodes: int = 50
    exp_dir: str = "./test_explorer_output"
    model_id: str = "gpt-4o-mini"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    max_repeats: int = 3
    similarity_threshold: float = 0.95
    temperature: float = 1.0
    dry_run: bool = True  # If True, print prompts without sending to LLM
    print_llm_response: bool = True  # If True, print full LLM response
    char_limit: int = 100000
    headless: bool = True


class ExplorerTester:
    """
    Universal Explorer Agent Tester
    
    Features:
    1. Initialize BrowserGym environment
    2. Create World Model (Graph) with encoder
    3. Initialize Explorer Agent
    4. Run exploration loop
    5. Support dry-run mode (print prompts without LLM calls)
    """
    
    def __init__(self, config: ExplorerTestConfig):
        self.config = config
        self.env: Optional[BrowserEnv] = None
        self.graph: Optional[Graph] = None
        self.agent: Optional[ExplorerAgent] = None
        self.current_node: Optional[Node] = None
        
        os.makedirs(config.exp_dir, exist_ok=True)
        
        logger.info(f"Explorer Tester initialized")
        logger.info(f"Target URL: {config.target_url}")
        logger.info(f"Max Steps: {config.max_steps}")
        logger.info(f"Max Nodes: {config.max_nodes}")
        logger.info(f"Dry Run: {config.dry_run}")
        
    def _create_encoder_fn(self):
        """
        Create semantic encoder using small embedding model
        Uses sentence-transformers for semantic similarity
        """
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        def semantic_encoder(obs: dict) -> list[float]:
            """
            Semantic encoder using complete axtree with smart truncation
            
            Features:
            - Uses full axtree (not just first 500 chars)
            - Semantic similarity for similar pages
            - Robust to minor text changes
            """
            url = obs.get("url", "")
            axtree_txt = obs.get("axtree_txt", "")
            
            # 智能截断：保留开头和结尾，避免超过模型限制
            max_chars = 8000  # ~2000 tokens, 适配模型的 512 token 限制
            if len(axtree_txt) > max_chars:
                keep_start = int(max_chars * 0.6)  # 保留前 60%
                keep_end = max_chars - keep_start   # 保留后 40%
                axtree_txt = (
                    axtree_txt[:keep_start] + 
                    "\n...[middle content truncated for encoding]...\n" + 
                    axtree_txt[-keep_end:]
                )
            
            # 组合 URL 和完整内容
            text = f"URL: {url}\n\nPage Content:\n{axtree_txt}"
            
            # 编码为 384 维 embedding
            embedding = model.encode(text, convert_to_tensor=False, show_progress_bar=False)
            
            return embedding.tolist()
        
        return semantic_encoder
    
    def setup_environment(self):
        """Initialize BrowserGym environment"""
        logger.info("Setting up BrowserGym environment...")
        
        task_kwargs = {
            "start_url": self.config.target_url,
            "goal": "Explore this website to discover new pages and states"
        }
        
        self.env = gym.make(
            "browsergym/openended",
            task_kwargs=task_kwargs,
            headless=self.config.headless,
        )
        
        logger.info("Environment created successfully")

    def preprocess_obs(self, obs: dict) -> dict:
        """Preprocess observation for agent"""
        axtree_txt = flatten_axtree_to_str(
            obs["axtree_object"],
            filter_visible_only=False,
            extra_properties=obs.get("extra_element_properties", {})
        )
        
        # 提取可交互元素（替代 extra_element_properties）
        interactive_elements = extract_interactive_elements(
            axtree_txt,
            obs.get("extra_element_properties", {})
        )
        
        return {
            "goal_object": [{"text": "Explore the website to discover new pages and states"}],
            "axtree_txt": axtree_txt,
            "last_action_error": obs.get("last_action_error", ""),
            "open_pages_urls": obs.get("open_pages_urls", []),
            "interactive_elements": [item for item in interactive_elements if item.get('visible') and item.get('clickable')],  # 新增：直接包含可交互元素
            "url": obs.get("url", ""),
        }
    
    def setup_graph(self):
        """Initialize World Model (Graph)"""
        encoder_fn = self._create_encoder_fn()
        self.graph = Graph(
            root_url=self.config.target_url,
            exp_dir=self.config.exp_dir,
            encoder_fn=encoder_fn,
            similarity_threshold=self.config.similarity_threshold,
            resume=False
        )


    def run_exploration(self):
        obs, info = self.env.reset() 
        processed_obs = self.preprocess_obs(obs)
        current_node = self.graph.current_node
        visited_actions = []
        target_bid = "202"  
        action = f"click('{target_bid}')"
        obs, reward, terminated, truncated, info = self.env.step(action)

if __name__ == "__main__":
    config = ExplorerTestConfig(
            target_url="http://3.148.75.200:7770/",
            max_steps=10,
            max_nodes=20,
            exp_dir="./test_explorer_output",
            model_id="base-model",  # Must match --served-model-name from vLLM server
            api_base="http://127.0.0.1:6006/v1",  # vLLM server endpoint
            max_repeats=3,
            similarity_threshold=0.95,
            temperature=1.0,
            dry_run=False,  # Set to False to call LLM and get response
            print_llm_response=True,  # Print full LLM response
            char_limit=100000,
            headless=False  # Set to False to see browser
        )
    tester = ExplorerTester(config)
    tester.setup_environment()
    tester.setup_graph()
    tester.run_exploration()
    input()
    tester.env.close()