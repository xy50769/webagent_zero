import logging
import os
import sys
import json
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
import gymnasium as gym
from browsergym.core.env import BrowserEnv
from browsergym.utils.obs import flatten_axtree_to_str
from sentence_transformers import SentenceTransformer
from ..core.element_utils import extract_interactive_elements
from ..core.graph import Graph
from ..core.node import Node
from ..agents.explorer_agent import ExplorerAgent
from ..core.element_utils import extract_interactive_elements

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@dataclass
class WebExploreAgentConfig:
    model_id: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    api_key: Optional[str] = ""
    temperature: float = 1.0
    max_retries: int = 3
    char_limit: int = 100000
    consistency_weight: float = 1.0
    efficiency_penalty: float = 0.05
    demo_mode: str = "off"
    debug_mode: str = "off"


@dataclass
class WebExploreEnvConfig:
    target_url: str
    encoder_name: str = "all-MiniLM-L6-v2"
    max_steps: int = 20
    max_nodes: int = 50
    resume: bool = False
    exp_dir: str = "./test_output"
    similarity_threshold: float = 0.95
    debug_mode: str = 'off'
    headless: bool = True




@dataclass
class WebExplorer:
    """Explorer"""
    env_config: WebExploreEnvConfig
    agent_config: WebExploreAgentConfig
    env: Optional[BrowserEnv] = field(init=False, default=None)
    agent: Optional[ExplorerAgent] = field(init=False, default=None)
    graph: Optional[Graph] = field(init=False, default=None)
    current_node: Optional[Node] = field(init=False, default=None)

    def __post_init__(self):
        os.makedirs(self.env_config.exp_dir, exist_ok=True)
        self._setup_logging()
        logger.info(f"Explorer initialized")
        logger.info(f"Target URL: {self.env_config.target_url}")
        logger.info(f"Max Steps: {self.env_config.max_steps}")
        logger.info(f"Max Nodes: {self.env_config.max_nodes}")

    def _setup_logging(self):
        graph_dir = os.path.join(self.env_config.exp_dir, "graph")
        os.makedirs(graph_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(graph_dir, f"exploration_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        logger.info(f"Log file created: {log_file}")
    
    def _create_encoder_fn(self):
        """Create semantic encoder using small embedding model"""
        logger.info("Loading embedding model (all-MiniLM-L6-v2, 22MB)...")
        model = SentenceTransformer(self.env_config.encoder_name)
        def semantic_encoder(obs: dict) -> list[float]:
            url = obs.get("url", "")
            axtree_txt = obs.get("axtree_txt", "")
            text = f"URL: {url}\n\nPage Content:\n{axtree_txt}"
            embedding = model.encode(text, convert_to_tensor=False, show_progress_bar=False)
            return embedding.tolist()
        return semantic_encoder

    def setup_environment(self):
        """Initialize BrowserGym environment"""
        logger.info("Setting up BrowserGym environment...")
        task_kwargs = {
            "start_url": self.env_config.target_url,
            "goal": "Explore this website to discover new pages and states"
        }
        self.env = gym.make(
            "browsergym/openended",
            task_kwargs=task_kwargs,
            headless=self.env_config.headless,
        )
        logger.info("Environment created successfully")

    def setup_graph(self):
        """Initialize World Model (Graph)"""
        logger.info("Setting up World Model (Graph)...")
        encoder_fn = self._create_encoder_fn()
        self.graph = Graph(
            root_url=self.env_config.target_url,
            exp_dir=self.env_config.exp_dir,
            encoder_fn=encoder_fn,
            encoder_name=self.env_config.encoder_name,
            similarity_threshold=self.env_config.similarity_threshold,
            resume=self.env_config.resume
        )
        logger.info("Graph initialized successfully")

    def setup_agent(self):
        """Initialize Explorer Agent"""
        logger.info("Setting up Explorer Agent...")
        self.agent = ExplorerAgent(
            model_id=self.agent_config.model_id,
            temperature=self.agent_config.temperature,
            base_url=self.agent_config.base_url,
            api_key=self.agent_config.api_key,
            max_retries=self.agent_config.max_retries,
            char_limit=self.agent_config.char_limit,
            consistency_weight=self.agent_config.consistency_weight,
            efficiency_penalty=self.agent_config.efficiency_penalty,
            demo_mode=self.agent_config.demo_mode,
            debug_mode=self.agent_config.debug_mode,
        )
        logger.info("Agent initialized successfully")

    def run(self):
        """Run the exploration loop"""
        logger.info("Starting exploration...")
        
        if self.env_config.resume and len(self.graph.nodes) > 0:
            logger.info(f"\n{'='*100}")
            logger.info("RESUMING PREVIOUS EXPLORATION")
            logger.info(f"{'='*100}")
            logger.info(f"Loaded graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
            logger.info(f"Total steps completed so far: {self.graph.total_steps}")
            logger.info(f"Will continue until reaching {self.env_config.max_nodes} nodes")
            logger.info(f"{'='*100}\n")
            
            if len(self.graph.nodes) >= self.env_config.max_nodes:
                logger.info(f"Graph already has {len(self.graph.nodes)} nodes (target: {self.env_config.max_nodes})")
                logger.info("No additional exploration needed!")
                return
        
        total_episodes = 0
        
        while len(self.graph.nodes) < self.env_config.max_nodes:
            total_episodes += 1
            logger.info(f"\n{'='*100}")
            logger.info(f"EPISODE {total_episodes} - Current Nodes: {len(self.graph.nodes)}/{self.env_config.max_nodes}")
            logger.info(f"{'='*100}\n")
            
            obs, info = self.env.reset()
            self.agent.reset()
            processed_obs = self.agent.obs_preprocessor(obs)
            interactive_elements = extract_interactive_elements(processed_obs["axtree_txt"],obs.get("extra_element_properties", {}))
            processed_obs["interactive_elements"] = interactive_elements
            processed_obs["url"] = obs.get("url", "")
            logger.info(f"Reset to Root URL: {obs.get('url', 'N/A')}")
            
            for step in range(self.env_config.max_steps):
                self.current_node = self.graph.current_node
                if self.current_node:
                    logger.info(f"Current Node: {self.current_node.node_id}")
                    logger.info(f"Current URL: {self.current_node.url}")
                    logger.info(f"Action History: {self.current_node.action_history}")
                else:
                    logger.info("Current Node: None (cold start)")
                logger.info(f"Total Nodes: {len(self.graph.nodes)}")
                logger.info(f"Total Edges: {len(self.graph.edges)}")

                unvisited_elements = []
                if self.current_node and hasattr(self.current_node, 'interactive_elements'):
                    unvisited_elements = self.current_node.interactive_elements
                    logger.info(f"Unvisited elements: {len(unvisited_elements)}, "
                            f"Visited: {len(self.current_node.interactive_elements_visited)}")
                else:
                    unvisited_elements = processed_obs.get("interactive_elements", [])
                    logger.info(f"Using all elements from obs: {len(unvisited_elements)}")
                action, action_extras = self.agent.get_action(processed_obs, node=self.current_node, graph=self.graph)
                thought = action_extras.get("thought", "")
                raw_action = action_extras.get("raw_action", "")
                logger.info(f"Thought: {thought[:100]}..." if len(thought) > 100 else f"Thought: {thought}")
                logger.info(f"Raw Action: {raw_action}")
                if not action:
                    logger.warning("No action generated, stopping")
                    break

                source_node_for_element_tracking = self.current_node
                obs, reward, terminated, truncated, info = self.env.step(action)
                processed_obs_next = self.agent.obs_preprocessor(obs)
                interactive_elements_next = extract_interactive_elements(
                    processed_obs_next["axtree_txt"],
                    obs.get("extra_element_properties", {})
                )
                processed_obs_next["interactive_elements"] = interactive_elements_next
                processed_obs_next["url"] = obs.get("url", "")
                
                logger.info(f"New URL: {obs.get('url', 'N/A')}")
                logger.info(f"Reward: {reward}")
                action_success = not obs.get('last_action_error', '')
                new_node = self.graph.process_transition(
                    obs_t=processed_obs,
                    action=action_extras.get('raw_action', action),
                    obs_t1=processed_obs_next,
                    explorer_execute_success=action_success,
                    thought=action_extras.get('thought', '')
                )
                logger.info(f"Transitioned to Node: {new_node.node_id}")
                explore_reward = self.agent.calculate_reward(
                    source_node=source_node_for_element_tracking,
                    target_node=new_node,
                    action_str=action_extras.get('raw_action', action),
                    graph=self.graph
                )
                logger.info(f"Exploration Reward: {explore_reward:.3f}")

                processed_obs = processed_obs_next

                if terminated or truncated:
                    logger.info("Episode terminated by environment")
                    break

                if len(self.graph.nodes) >= self.env_config.max_nodes:
                    logger.info(f"Reached max nodes limit: {self.env_config.max_nodes}")
                    break
            
            logger.info(f"\nEpisode {total_episodes} completed!")
            logger.info(f"Steps in this episode: {step + 1}")
            logger.info(f"Total nodes so far: {len(self.graph.nodes)}/{self.env_config.max_nodes}")
            
            if len(self.graph.nodes) >= self.env_config.max_nodes:
                logger.info(f"Reached max nodes limit: {self.env_config.max_nodes}")
                break
        
        logger.info("\n" + "="*100)
        logger.info("EXPLORATION COMPLETED!")
        logger.info("="*100)
        logger.info(f"Total episodes: {total_episodes}")
        logger.info(f"Total nodes discovered: {len(self.graph.nodes)}")
        logger.info(f"Total edges discovered: {len(self.graph.edges)}")
        logger.info(f"Total steps: {self.graph.total_steps}")

    def cleanup(self):
        """Cleanup resources"""
        if self.env:
            self.env.close()
            logger.info("Environment closed")

if __name__ == "__main__":
    env_config = WebExploreEnvConfig(
        target_url="http://3.148.75.200:7770/",
        encoder_name="all-MiniLM-L6-v2",
        max_steps=20,
        max_nodes=100,
        exp_dir="./one_stop_market",
        resume=False,
        similarity_threshold=0.96,
        debug_mode="off",
        headless=False,
    )
    agent_config = WebExploreAgentConfig(
        model_id="base-model", 
        base_url="http://127.0.0.1:6006/v1",
        api_key="",
        temperature=1.0,
        max_retries=3,
        char_limit=100000,
        consistency_weight=1.0,
        efficiency_penalty=0.05,
        demo_mode="off",
        debug_mode="on",
    )
    tester = WebExplorer(env_config=env_config, agent_config=agent_config)
    tester.setup_environment()
    tester.setup_graph()
    tester.setup_agent()
    tester.run()
    tester.cleanup()