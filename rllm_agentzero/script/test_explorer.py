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

from rllm_agentzero.agents.explorer_agent import ExplorerAgent
from rllm_agentzero.core.graph import Graph
from rllm_agentzero.core.node import Node

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
        Create a simple encoder function
        In production, this should use a proper embedding model
        """
        def simple_encoder(obs: dict) -> list[float]:
            """
            Simple hash-based encoder for testing
            In production, replace with proper embedding model
            """
            url = obs.get("url", "")
            axtree_txt = obs.get("axtree_txt", "")
            
            # Simple embedding: use hash values
            import hashlib
            text = f"{url}_{axtree_txt[:500]}"
            hash_val = hashlib.md5(text.encode()).hexdigest()
            
            # Convert to pseudo-embedding (32-dim)
            embedding = []
            for i in range(0, 32):
                byte_idx = i % len(hash_val)
                embedding.append(int(hash_val[byte_idx], 16) / 15.0)
            
            return embedding
        
        return simple_encoder
    
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
        
    def setup_graph(self):
        """Initialize World Model (Graph)"""
        logger.info("Setting up World Model (Graph)...")
        
        encoder_fn = self._create_encoder_fn()
        
        self.graph = Graph(
            root_url=self.config.target_url,
            exp_dir=self.config.exp_dir,
            encoder_fn=encoder_fn,
            similarity_threshold=self.config.similarity_threshold,
            resume=False
        )
        
        logger.info("Graph initialized successfully")
        
    def setup_agent(self):
        """Initialize Explorer Agent"""
        logger.info("Setting up Explorer Agent...")
        
        agent_kwargs = {
            "model_id": self.config.model_id,
            "max_repeats": self.config.max_repeats,
            "temperature": self.config.temperature,
            "char_limit": self.config.char_limit,
        }
        
        if self.config.api_base:
            agent_kwargs["api_base"] = self.config.api_base
        if self.config.api_key:
            agent_kwargs["api_key"] = self.config.api_key
            
        self.agent = ExplorerAgent(**agent_kwargs)
        
        logger.info(f"Explorer Agent initialized with model: {self.config.model_id}")
        
    def preprocess_obs(self, obs: dict) -> dict:
        """Preprocess observation for agent"""
        return {
            "goal_object": [{"text": "Explore the website to discover new pages and states"}],
            "axtree_txt": flatten_axtree_to_str(
                obs["axtree_object"],
                filter_visible_only=False,
                extra_properties=obs.get("extra_element_properties", {})
            ),
            "last_action_error": obs.get("last_action_error", ""),
            "open_pages_urls": obs.get("open_pages_urls", []),
            "extra_element_properties": obs.get("extra_element_properties", {}),
            "url": obs.get("url", ""),
        }
        
    def count_tokens(self, messages: list[dict], model: str = "qwen2.5") -> int:
        """Count tokens in messages using Qwen tokenizer"""
        from transformers import AutoTokenizer
        
        # Initialize tokenizer (cached after first call)
        if not hasattr(self, '_tokenizer'):
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct",
                trust_remote_code=True
            )
        
        # Extract text content from messages
        text_content = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            # Add role prefix for more accurate count
            text_content.append(f"<|im_start|>{role}")
            
            if isinstance(content, str):
                text_content.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content.append(item.get("text", ""))
            
            text_content.append("<|im_end|>")
        
        full_text = "\n".join(text_content)
        tokens = self._tokenizer.encode(full_text)
        
        return len(tokens)
    
    def print_prompt(self, messages: list[dict], step: int):
        """Print prompt messages in a readable format"""
        # Calculate tokens
        token_count = self.count_tokens(messages, self.config.model_id)
        
        output = []
        output.append("=" * 100)
        output.append(f"PROMPT FOR STEP {step} | Tokens: {token_count:,}")
        output.append("=" * 100)
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            output.append(f"\n[{role.upper()}]")
            
            if isinstance(content, str):
                output.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            output.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            output.append("[Image URL provided]")
        
        output.append("=" * 100)
        output.append(f"Total Tokens: {token_count:,}")
        output.append("=" * 100)
        
        print("\n".join(output))
        
    def run_exploration(self):
        """Run the exploration loop"""
        logger.info("Starting exploration...")
        
        # Reset environment
        obs, info = self.env.reset()
        processed_obs = self.preprocess_obs(obs)
        
        # Initialize Graph with starting observation
        logger.info(f"Initial URL: {obs.get('url', 'N/A')}")
        
        # Agent reset
        self.agent.reset()
        
        for step in range(self.config.max_steps):
            logger.info(f"\n{'='*80}")
            logger.info(f"EXPLORATION STEP {step + 1}/{self.config.max_steps}")
            logger.info(f"{'='*80}")
            
            # Get current node from graph
            self.current_node = self.graph.current_node
            
            # Log current state
            if self.current_node:
                logger.info(f"Current Node: {self.current_node.node_id}")
                logger.info(f"Current URL: {self.current_node.url}")
                logger.info(f"Action History: {dict(self.current_node.action_history)}")
            else:
                logger.info("Current Node: None (cold start)")
            
            logger.info(f"Total Nodes: {len(self.graph.nodes)}")
            logger.info(f"Total Edges: {len(self.graph.edges)}")
            
            # Build prompt messages
            # We need to intercept before LLM call
            visited_actions = []
            if self.current_node and hasattr(self.current_node, "action_history"):
                visited_actions = [
                    act for act, count in self.current_node.action_history.items()
                    if count >= self.agent.max_repeats
                ]
            
            # Get frontier info
            frontier_info = None
            if self.graph:
                frontier_node = self.graph.get_next_node()
                if frontier_node:
                    frontier_info = {
                        "node_id": frontier_node.node_id,
                        "url": getattr(frontier_node, 'url', ''),
                    }
            
            # Build messages using prompt builder
            messages = self.agent.prompt_builder.construct_explorer_prompt_messages(
                goal=self.agent._goal,
                obs=processed_obs,
                history=self.agent.history,
                visited_actions=visited_actions,
                frontier_info=frontier_info
            )
            
            # Print the prompt
            self.print_prompt(messages, step + 1)
            
            if self.config.dry_run:
                logger.info("DRY RUN MODE: Stopping before LLM call")
                logger.info(f"Visited Actions: {visited_actions}")
                if frontier_info:
                    logger.info(f"Frontier Info: {frontier_info}")
                break
            
            # Get action from agent
            action, action_extras = self.agent.get_action(
                obs=processed_obs,
                node=self.current_node,
                graph=self.graph
            )
            
            logger.info(f"Action: {action}")
            logger.info(f"Thought: {action_extras.get('thought', 'N/A')}")
            
            if not action:
                logger.warning("No action generated, stopping")
                break
            
            # Execute action in environment
            try:
                obs, reward, terminated, truncated, info = self.env.step(action)
                processed_obs_next = self.preprocess_obs(obs)
                
                logger.info(f"New URL: {obs.get('url', 'N/A')}")
                logger.info(f"Reward: {reward}")
                
                # Update graph with transition
                new_node = self.graph.process_transition(
                    obs_t=processed_obs,
                    action=action_extras.get('raw_action', action),
                    obs_t1=processed_obs_next,
                    success=(reward > 0)
                )
                
                logger.info(f"Transitioned to Node: {new_node.node_id}")
                
                # Calculate exploration reward
                if self.current_node:
                    explore_reward = self.agent.calculate_reward(
                        source_node=self.current_node,
                        target_node=new_node,
                        action_str=action_extras.get('raw_action', action),
                        graph=self.graph
                    )
                    logger.info(f"Exploration Reward: {explore_reward:.3f}")
                
                # Update for next iteration
                processed_obs = processed_obs_next
                
                # Check termination conditions
                if terminated or truncated:
                    logger.info("Episode terminated by environment")
                    break
                
                if len(self.graph.nodes) >= self.config.max_nodes:
                    logger.info(f"Reached max nodes limit: {self.config.max_nodes}")
                    break
                    
            except Exception as e:
                logger.error(f"Error during step execution: {e}", exc_info=True)
                break
        
        logger.info("\nExploration completed!")
        logger.info(f"Total nodes discovered: {len(self.graph.nodes)}")
        logger.info(f"Total edges discovered: {len(self.graph.edges)}")
        
    def cleanup(self):
        """Cleanup resources"""
        if self.env:
            self.env.close()
            logger.info("Environment closed")
            
    def run(self):
        """Main entry point"""
        try:
            self.setup_environment()
            self.setup_graph()
            self.setup_agent()
            self.run_exploration()
        except Exception as e:
            logger.error(f"Error during exploration: {e}", exc_info=True)
        finally:
            self.cleanup()


def main():
    """Test Explorer on One Stop Market website"""
    
    config = ExplorerTestConfig(
        target_url="http://3.148.75.200:7770/",
        max_steps=10,
        max_nodes=20,
        exp_dir="./test_explorer_output",
        model_id="qwen2.5-7b",  # Using Qwen 2.5
        max_repeats=3,
        similarity_threshold=0.95,
        temperature=1.0,
        dry_run=True,  # Set to True to only print prompts
        char_limit=100000,
        headless=False  # Set to False to see browser
    )
    
    tester = ExplorerTester(config)
    tester.run()


if __name__ == "__main__":
    main()
