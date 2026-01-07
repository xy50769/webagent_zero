import sys
import os
import logging
import gymnasium as gym
from collections import Counter
from unittest.mock import Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rllm_agentzero.agents_old.explorer_agent import ExplorerAgent
from rllm_agentzero.agents_old.server.llm_engine import LLMEngine
from browsergym.core.task import AbstractBrowserTask
from browsergym.core.env import BrowserEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RemoteLLMEngine:
    """Wrapper for remote LLM service via SSH tunnel."""
    
    def __init__(self, api_base: str = None):
        try:
            import requests
        except ImportError:
            raise ImportError("requests package is required for remote LLM. Install with: pip install requests")
        
        if api_base is None:
            api_base = os.getenv("LLM_API_BASE", "http://127.0.0.1:6006")
        
        self.api_base = api_base
        self.requests = requests
        
        logger.info(f"Initialized RemoteLLMEngine with API base: {api_base}")
        logger.info(f"Server-side model will be used (configured in server/llm_engine.py)")
    
    def construct_prompt(self, system_msg: str, user_msg: str) -> str:
        """Construct prompt for OpenAI-compatible API."""
        return system_msg + "\n\n" + user_msg
    
    def generate(self, system_msg: str, user_msg: str, mode: str = "base", temperature: float = 1.0) -> str:
        """Generate response using remote API."""
        try:
            # Server uses /generate endpoint with custom format
            # Remove /v1 suffix if present
            base_url = self.api_base.rstrip('/v1').rstrip('/')
            url = f"{base_url}/generate"
            
            payload = {
                "system_msg": system_msg,
                "user_msg": user_msg,
                "mode": mode,
                "temperature": temperature
            }
            
            logger.info(f"Calling remote API with mode: {mode}, url: {url}")
            
            response = self.requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300
            )
            response.raise_for_status()
            
            result = response.json()
            # Server returns {"text": response}
            content = result.get("text", "").strip()
            if not content:
                # Fallback: try to get from response directly
                content = str(result).strip()
            
            logger.info(f"Remote API call successful, response length: {len(content)}")
            return content
        except self.requests.exceptions.RequestException as e:
            logger.error(f"Remote API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Remote API call failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def create_remote_llm_engine():
    """Create a remote LLMEngine for online testing via SSH tunnel."""
    # Default to /generate endpoint (not /v1)
    api_base = os.getenv("LLM_API_BASE", "http://127.0.0.1:6006")
    
    logger.info(f"Creating RemoteLLMEngine")
    logger.info(f"API Base (via SSH tunnel): {api_base}")
    logger.info(f"Server will use model configured in server/llm_engine.py")
    
    return RemoteLLMEngine(api_base=api_base)


class ExplorerTestTask(AbstractBrowserTask):
    """Test task for ExplorerAgent online test."""
    def setup(self, page):
        test_url = os.getenv("TEST_URL", "http://3.148.75.200:7770/")
        page.goto(test_url)
        return "Explore the website and discover new pages or state changes", {}
    
    def validate(self, page, chat_messages):
        return 0.0, False, "", {}
    
    def teardown(self):
        pass


def register_test_env():
    """Register the test environment."""
    if "browsergym/explorer_online_test" not in gym.envs.registry:
        gym.register(
            id="browsergym/explorer_online_test",
            entry_point="browsergym.core.env:BrowserEnv",
            kwargs={"task_entrypoint": ExplorerTestTask}
        )


def test_explorer_agent_online_single_step():
    """Test ExplorerAgent with remote LLM (via SSH tunnel) and local browser - single step."""
    register_test_env()
    
    llm_engine = create_remote_llm_engine()
    agent = ExplorerAgent(llm_engine=llm_engine)
    
    env = None
    try:
        headless = os.getenv("HEADLESS", "False").lower() == "true"
        env = gym.make("browsergym/explorer_online_test", headless=headless)
        obs, info = env.reset()
        
        logger.info(f"Browser opened, current URL: {obs.get('url', 'unknown')}")
        logger.info(f"Observation keys: {list(obs.keys())}")
        
        processed_obs = agent.obs_preprocessor(obs)
        logger.info(f"Processed observation has axtree_txt: {'axtree_txt' in processed_obs}")
        logger.info(f"Axtree length: {len(processed_obs.get('axtree_txt', ''))}")
        
        logger.info("Calling agent.get_action() with remote LLM via SSH tunnel...")
        response, extras = agent.get_action(processed_obs)
        
        logger.info(f"Agent response length: {len(response)}")
        logger.info(f"Agent response preview: {response[:200]}...")
        logger.info(f"Parsed action preview: {extras.get('parsed_action', '')[:200]}...")
        logger.info(f"Thought: {extras.get('thought', '')}")
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "parsed_action" in extras
        assert len(agent.history) == 1
        
        logger.info("Successfully got action from ExplorerAgent with remote LLM and local browser")
        
    finally:
        if env:
            env.close()
            logger.info("Browser closed")


def test_explorer_agent_online_multiple_steps():
    """Test ExplorerAgent with remote LLM (via SSH tunnel) and local browser - multiple steps."""
    register_test_env()
    
    llm_engine = create_remote_llm_engine()
    agent = ExplorerAgent(llm_engine=llm_engine)
    
    max_steps = int(os.getenv("MAX_STEPS", "5"))
    
    env = None
    try:
        headless = os.getenv("HEADLESS", "False").lower() == "true"
        env = gym.make("browsergym/explorer_online_test", headless=headless)
        obs, info = env.reset()
        
        logger.info(f"Starting multi-step test with remote LLM (via SSH tunnel) and local browser")
        logger.info(f"Initial URL: {obs.get('url', 'unknown')}")
        logger.info(f"Max steps: {max_steps}")
        
        for step in range(max_steps):
            logger.info(f"\n{'='*60}")
            logger.info(f"Step {step + 1}/{max_steps}")
            logger.info(f"{'='*60}")
            
            processed_obs = agent.obs_preprocessor(obs)
            logger.info(f"Getting action from agent...")
            
            response, extras = agent.get_action(processed_obs)
            
            parsed_action = extras.get('parsed_action', '')
            thought = extras.get('thought', '')
            
            logger.info(f"Step {step + 1} - Thought: {thought}")
            raw_action = extras.get('raw_action', '')
            logger.info(f"Step {step + 1} - Raw Action: {raw_action}")
            logger.info(f"History length: {len(agent.history)}")
            
            if step < max_steps - 1:
                logger.info(f"Executing action...")
                try:
                    # Set action_mapping for BrowserEnv
                    browser_env = env.unwrapped if hasattr(env, "unwrapped") else env
                    if hasattr(browser_env, "action_mapping"):
                        browser_env.action_mapping = agent.action_processor
                    
                    # Pass raw_action (action string) to env.step() - BrowserGym will convert it using action_mapping
                    obs, reward, terminated, truncated, info = env.step(raw_action)
                    logger.info(f"Step {step + 1} - New URL: {obs.get('url', 'unknown')}")
                    logger.info(f"Step {step + 1} - Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
                    
                    if terminated or truncated:
                        logger.info(f"Episode ended at step {step + 1}")
                        break
                except Exception as e:
                    logger.error(f"Error executing action at step {step + 1}: {e}")
                    break
        
        logger.info(f"\nCompleted {len(agent.history)} steps successfully")
        logger.info(f"Final history length: {len(agent.history)}")
        
    finally:
        if env:
            env.close()
            logger.info("Browser closed")


def test_explorer_agent_online_with_node():
    """Test ExplorerAgent with remote LLM (via SSH tunnel), local browser, and node with action history."""
    import tempfile
    import numpy as np
    from rllm_agentzero.core.node import Node
    from rllm_agentzero.core.graph import Graph
    
    def create_mock_encoder(embedding_dim: int = 128):
        """Create a deterministic mock encoder for testing."""
        def encoder_fn(obs: dict) -> list[float]:
            url = obs.get("url", "")
            axtree = obs.get("axtree_txt", "")
            content = f"{url}|{axtree[:200]}"
            
            np.random.seed(hash(content) % (2**32))
            embedding = np.random.randn(embedding_dim).tolist()
            embedding = np.array(embedding)
            return embedding.tolist()
        
        return encoder_fn
    
    register_test_env()
    
    llm_engine = create_remote_llm_engine()
    agent = ExplorerAgent(llm_engine=llm_engine)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_url = os.getenv("TEST_URL", "http://3.148.75.200:7770/")
        encoder_fn = create_mock_encoder()
        graph = Graph(root_url=test_url, exp_dir=temp_dir, encoder_fn=encoder_fn)
        obs_mock = {"url": test_url, "axtree_txt": "One Stop Market"}
        # 使用 process_transition 创建初始节点
        node = graph.process_transition(
            obs_t=obs_mock,
            action="noop",
            obs_t1=obs_mock,
            success=True,
            hint="Start Here"
        )
        
        node.record_action("click('10')")
        node.record_action("click('11')")
        node.record_action("click('10')")
        
        logger.info(f"Node action history: {dict(node.action_history)}")
        
        env = None
        try:
            headless = os.getenv("HEADLESS", "False").lower() == "true"
            env = gym.make("browsergym/explorer_online_test", headless=headless)
            obs, info = env.reset()
            
            logger.info(f"Testing with node containing action history")
            logger.info(f"Current URL: {obs.get('url', 'unknown')}")
            
            processed_obs = agent.obs_preprocessor(obs)
            response, extras = agent.get_action(processed_obs, node=node)
            
            logger.info(f"Agent response: {response[:200]}...")
            logger.info(f"Thought: {extras.get('thought', '')}")
            logger.info(f"Parsed action: {extras.get('parsed_action', '')[:150]}...")
            
            assert isinstance(response, str)
            assert "parsed_action" in extras
            assert len(agent.history) == 1
            
            logger.info("Successfully tested ExplorerAgent with node action history")
            
        finally:
            if env:
                env.close()
                logger.info("Browser closed")


def test_explorer_agent_online_save_graph():
    """Test ExplorerAgent exploration and save graph to local directory for proposer testing."""
    import numpy as np
    from rllm_agentzero.core.graph import Graph
    from datetime import datetime
    
    def create_mock_encoder(embedding_dim: int = 128):
        """Create a deterministic mock encoder for testing."""
        def encoder_fn(obs: dict) -> list[float]:
            url = obs.get("url", "")
            axtree = obs.get("axtree_txt", "")
            content = f"{url}|{axtree[:200]}"  # Use first 200 chars for stability
            
            np.random.seed(hash(content) % (2**32))
            embedding = np.random.randn(embedding_dim).tolist()
            embedding = np.array(embedding)
            return embedding.tolist()
        
        return encoder_fn
    
    register_test_env()
    
    llm_engine = create_remote_llm_engine()
    agent = ExplorerAgent(llm_engine=llm_engine)
    
    # Get save directory from environment variable or use default
    base_save_dir = os.getenv("GRAPH_SAVE_DIR", "./explored_graphs")
    test_url = os.getenv("TEST_URL", "http://3.148.75.200:7770/")
    
    # Create timestamped directory for this exploration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Make URL safe for Windows file system (remove invalid characters)
    url_safe = test_url.replace("https://", "").replace("http://", "").replace("/", "_").replace(".", "_").replace(":", "_")
    save_dir = os.path.join(base_save_dir, f"{url_safe}_{timestamp}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Graph Exploration and Save Test")
    logger.info(f"{'='*60}")
    logger.info(f"Test URL: {test_url}")
    logger.info(f"Graph will be saved to: {save_dir}")
    
    # Create encoder and graph
    encoder_fn = create_mock_encoder()
    graph = Graph(
        root_url=test_url,
        exp_dir=save_dir,
        encoder_fn=encoder_fn,
        similarity_threshold=0.95
    )
    
    env = None
    try:
        headless = os.getenv("HEADLESS", "False").lower() == "true"
        env = gym.make("browsergym/explorer_online_test", headless=headless)
        obs, info = env.reset()
        
        logger.info(f"Initial URL: {obs.get('url', 'unknown')}")
        
        # Create initial node
        processed_obs = agent.obs_preprocessor(obs)
        initial_node = graph.process_transition(
            obs_t=processed_obs,
            action="noop",
            obs_t1=processed_obs,
            success=True,
            hint="Start Here"
        )
        graph.current_node = initial_node
        logger.info(f"Created initial node: {initial_node.node_id}")
        
        # Explore multiple steps
        max_steps = int(os.getenv("GRAPH_EXPLORE_STEPS", "10"))
        logger.info(f"Exploring {max_steps} steps...")
        
        for step in range(max_steps):
            logger.info(f"\n--- Step {step + 1}/{max_steps} ---")
            
            processed_obs = agent.obs_preprocessor(obs)
            response, extras = agent.get_action(processed_obs, node=graph.current_node)
            
            parsed_action = extras.get('parsed_action', '')
            raw_action = extras.get('raw_action', '')  # Simple action like click('261')
            thought = extras.get('thought', '')
            
            logger.info(f"Thought: {thought[:100]}...")
            logger.info(f"Raw Action: {raw_action}")
            logger.info(f"Parsed Action (code): {parsed_action[:80]}...")
            
            # Record raw_action (simple action string) in current node, not parsed_action (full code)
            if raw_action and raw_action.strip():
                graph.current_node.record_action(raw_action)
            
            # Execute action and process transition
            try:
                # BrowserEnv.step() expects action string (raw_action), not full Python code
                # BrowserGym will use action_mapping (agent.action_processor) to convert it to code
                browser_env = env.unwrapped if hasattr(env, "unwrapped") else env
                if hasattr(browser_env, "action_mapping"):
                    browser_env.action_mapping = agent.action_processor
                
                logger.info(f"Executing raw_action: {raw_action}")
                logger.info(f"Parsed action (code) length: {len(parsed_action)} chars")
                
                # Pass raw_action (action string) to env.step() - BrowserGym will convert it using action_mapping
                obs_new, reward, terminated, truncated, info = env.step(raw_action)
                
                # Log execution results
                logger.info(f"Step {step + 1} - Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
                logger.info(f"Step {step + 1} - URL before: {processed_obs.get('url', 'unknown')}")
                logger.info(f"Step {step + 1} - URL after: {obs_new.get('url', 'unknown')}")
                
                # Check for action errors
                last_action_error = obs_new.get('last_action_error', '')
                if last_action_error:
                    logger.warning(f"Step {step + 1} - Action error: {last_action_error}")
                
                # Process transition in graph (use raw_action for action_history, not parsed_action)
                new_node = graph.process_transition(
                    obs_t=processed_obs,
                    action=raw_action,  # Use raw_action (simple string) instead of parsed_action (full code)
                    obs_t1=agent.obs_preprocessor(obs_new),
                    success=(reward >= 0),
                    hint=f"Step {step + 1}",
                    thought=thought
                )
                
                logger.info(f"Current node: {graph.current_node.node_id}")
                logger.info(f"New node: {new_node.node_id}")
                
                obs = obs_new
                
                if terminated or truncated:
                    logger.info("Episode ended")
                    break
                    
            except Exception as e:
                logger.error(f"Error executing action: {e}")
                import traceback
                logger.error(traceback.format_exc())
                break
        
        # Ensure all nodes are saved (including prefixes)
        for node in graph.nodes.values():
            node.update_save(save_prefix=True, save_info=True)
        
        # Print graph structure summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Graph Structure Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Total Nodes: {len(graph.nodes)}")
        logger.info(f"Explored Nodes: {len(graph.explored_nodes)}")
        logger.info(f"Unexplored Nodes: {len(graph.unexplored_nodes)}")
        logger.info(f"Total Edges: {len(graph.edges)}")
        
        # Print save location
        graph_save_path = graph.exp_dir
        logger.info(f"\n{'='*60}")
        logger.info(f"Graph Saved Successfully")
        logger.info(f"{'='*60}")
        logger.info(f"Graph directory: {graph_save_path}")
        logger.info(f"\nTo load this graph later, use:")
        logger.info(f"  from rllm_agentzero.core.graph import Graph")
        logger.info(f"  graph = Graph.load('{graph_save_path}', encoder_fn=your_encoder_fn)")
        logger.info(f"{'='*60}\n")
        
        # Verify files exist
        graph_info_path = os.path.join(graph_save_path, "graph_info.json")
        edges_path = os.path.join(graph_save_path, "edges.json")
        
        if os.path.exists(graph_info_path):
            logger.info(f"✓ graph_info.json exists")
        else:
            logger.warning(f"✗ graph_info.json not found")
        
        if os.path.exists(edges_path):
            logger.info(f"✓ edges.json exists")
        else:
            logger.warning(f"✗ edges.json not found")
        
        logger.info(f"✓ {len(graph.nodes)} node directories created")
        
    finally:
        if env:
            env.close()
            logger.info("Browser closed")


def test_explorer_agent_online_batch_success_rate():
    """Test ExplorerAgent on multiple real websites, measure success rate, and save graphs."""
    import numpy as np
    from rllm_agentzero.core.graph import Graph
    from datetime import datetime
    
    def create_mock_encoder(embedding_dim: int = 128):
        """Create a deterministic mock encoder for testing."""
        def encoder_fn(obs: dict) -> list[float]:
            url = obs.get("url", "")
            axtree = obs.get("axtree_txt", "")
            content = f"{url}|{axtree[:200]}"
            
            np.random.seed(hash(content) % (2**32))
            embedding = np.random.randn(embedding_dim).tolist()
            embedding = np.array(embedding)
            return embedding.tolist()
        
        return encoder_fn
    
    # Define test websites - diverse set of real websites
    test_websites = [
        "https://www.wikipedia.org",
        "https://www.github.com",
        "https://news.ycombinator.com",
        "https://www.stackoverflow.com",
        "https://www.reddit.com",
    ]
    
    # Allow override via environment variable
    test_urls_str = os.getenv("TEST_WEBSITES", None)
    if test_urls_str:
        test_websites = [url.strip() for url in test_urls_str.split(",")]
    
    num_tests = len(test_websites)
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Batch Success Rate Test with Graph Saving")
    logger.info(f"{'='*60}")
    logger.info(f"Total websites to test: {num_tests}")
    logger.info(f"Websites: {test_websites}")
    
    # Get save directory
    base_save_dir = os.getenv("GRAPH_SAVE_DIR", "./explored_graphs")
    save_graphs = os.getenv("SAVE_GRAPHS", "true").lower() == "true"
    
    if save_graphs:
        logger.info(f"Graphs will be saved to: {base_save_dir}")
    
    results = []
    
    for i, test_url in enumerate(test_websites):
        logger.info(f"\n{'='*60}")
        logger.info(f"Test {i+1}/{num_tests}: {test_url}")
        logger.info(f"{'='*60}")
        
        # Create a task class for this specific URL
        class DynamicTestTask(AbstractBrowserTask):
            def setup(self, page):
                page.goto(test_url)
                return "Explore the website and discover new pages or state changes", {}
            
            def validate(self, page, chat_messages):
                return 0.0, False, "", {}
            
            def teardown(self):
                pass
        
        # Register environment for this URL
        env_id = f"browsergym/explorer_batch_test_{i}"
        if env_id in gym.envs.registry:
            gym.envs.registry.pop(env_id)
        gym.register(
            id=env_id,
            entry_point="browsergym.core.env:BrowserEnv",
            kwargs={"task_entrypoint": DynamicTestTask}
        )
        
        llm_engine = create_remote_llm_engine()
        agent = ExplorerAgent(llm_engine=llm_engine)
        
        # Create graph for this website if saving is enabled
        graph = None
        graph_save_path = None
        if save_graphs:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            url_safe = test_url.replace("https://", "").replace("http://", "").replace("/", "_").replace(".", "_")
            save_dir = os.path.join(base_save_dir, f"{url_safe}_{timestamp}")
            encoder_fn = create_mock_encoder()
            graph = Graph(
                root_url=test_url,
                exp_dir=save_dir,
                encoder_fn=encoder_fn,
                similarity_threshold=0.95
            )
            graph_save_path = graph.exp_dir
        
        env = None
        success = False
        error_msg = None
        action_preview = None
        thought_preview = None
        
        try:
            headless = os.getenv("HEADLESS", "False").lower() == "true"
            env = gym.make(env_id, headless=headless)
            obs, info = env.reset()
            
            logger.info(f"Test {i+1} - Browser opened, URL: {obs.get('url', 'unknown')}")
            
            processed_obs = agent.obs_preprocessor(obs)
            axtree_len = len(processed_obs.get('axtree_txt', ''))
            logger.info(f"Test {i+1} - Axtree length: {axtree_len}")
            
            # Create initial node if graph exists
            if graph:
                initial_node = graph.process_transition(
                    obs_t=processed_obs,
                    action="noop",
                    obs_t1=processed_obs,
                    success=True,
                    hint="Start Here"
                )
                graph.current_node = initial_node
                logger.info(f"Test {i+1} - Created initial graph node: {initial_node.node_id}")
            
            response, extras = agent.get_action(processed_obs, node=graph.current_node if graph else None)
            
            if response and len(response) > 0 and "parsed_action" in extras:
                parsed_action = extras.get('parsed_action', '')
                raw_action = extras.get('raw_action', '')  # Simple action like click('261')
                thought = extras.get('thought', '')
                
                if parsed_action and parsed_action.strip():
                    success = True
                    action_preview = raw_action if raw_action else parsed_action[:150]
                    thought_preview = thought[:100] if len(thought) > 100 else thought
                    logger.info(f"Test {i+1} - SUCCESS: Got valid action")
                    logger.info(f"Test {i+1} - Thought: {thought_preview}...")
                    logger.info(f"Test {i+1} - Raw Action: {raw_action}")
                    
                    # Record raw_action (simple action string) and process transition in graph if exists
                    if graph and graph.current_node:
                        if raw_action and raw_action.strip():
                            graph.current_node.record_action(raw_action)
                        try:
                            # Set action_mapping for BrowserEnv
                            browser_env = env.unwrapped if hasattr(env, "unwrapped") else env
                            if hasattr(browser_env, "action_mapping"):
                                browser_env.action_mapping = agent.action_processor
                            
                            # Pass raw_action (action string) to env.step() - BrowserGym will convert it using action_mapping
                            obs_new, reward, terminated, truncated, info = env.step(raw_action)
                            new_node = graph.process_transition(
                                obs_t=processed_obs,
                                action=raw_action,  # Use raw_action (simple string) instead of parsed_action (full code)
                                obs_t1=agent.obs_preprocessor(obs_new),
                                success=(reward >= 0),
                                hint=f"Step 1",
                                thought=thought
                            )
                            logger.info(f"Test {i+1} - Graph updated: {graph.current_node.node_id} -> {new_node.node_id}")
                        except Exception as e:
                            logger.warning(f"Test {i+1} - Failed to process transition in graph: {e}")
                else:
                    error_msg = "Empty parsed action"
                    logger.warning(f"Test {i+1} - FAILED: {error_msg}")
            else:
                error_msg = "Invalid response or missing parsed_action"
                logger.warning(f"Test {i+1} - FAILED: {error_msg}")
                
        except ValueError as e:
            if "empty action" in str(e).lower():
                error_msg = "Empty action error"
            else:
                error_msg = f"ValueError: {str(e)}"
            logger.error(f"Test {i+1} - FAILED: {error_msg}")
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            logger.error(f"Test {i+1} - FAILED: {error_msg}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            if env:
                env.close()
            
            # Save graph if it exists
            if graph:
                try:
                    for node in graph.nodes.values():
                        node.update_save(save_prefix=True, save_info=True)
                    logger.info(f"Test {i+1} - Graph saved to: {graph_save_path}")
                    logger.info(f"Test {i+1} - Graph stats: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
                    # Log prefix information
                    for node_id, node in graph.nodes.items():
                        logger.info(f"Test {i+1} - Node {node_id}: {len(node.prefixes)} prefixes")
                except Exception as e:
                    logger.error(f"Test {i+1} - Failed to save graph: {e}")
            
        results.append({
            "test_num": i+1,
            "url": test_url,
            "success": success,
            "error": error_msg,
            "action": action_preview,
            "thought": thought_preview,
            "graph_saved": graph_save_path if graph else None
        })
        
        # Small delay between tests
        import time
        time.sleep(2)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Batch Test Summary")
    logger.info(f"{'='*60}")
    
    successful = sum(1 for r in results if r["success"])
    failed = num_tests - successful
    success_rate = (successful / num_tests) * 100 if num_tests > 0 else 0
    
    logger.info(f"Total websites tested: {num_tests}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {success_rate:.2f}%")
    
    logger.info(f"\nDetailed Results:")
    for r in results:
        status = "✓ PASS" if r["success"] else "✗ FAIL"
        logger.info(f"  Test {r['test_num']}: {status} - {r['url']}")
        if r['error']:
            logger.info(f"    Error: {r['error']}")
        if r['thought']:
            logger.info(f"    Thought: {r['thought']}...")
        if r['action']:
            logger.info(f"    Action: {r['action'][:80]}...")
        if r.get('graph_saved'):
            logger.info(f"    Graph saved: {r['graph_saved']}")
    
    logger.info(f"{'='*60}\n")
    
    # Assert that we have at least some success
    assert successful > 0, f"No successful tests out of {num_tests} attempts"
    
    return success_rate


if __name__ == "__main__":
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    tests = [
        ("test_explorer_agent_online_single_step", test_explorer_agent_online_single_step),
        ("test_explorer_agent_online_multiple_steps", test_explorer_agent_online_multiple_steps),
        ("test_explorer_agent_online_with_node", test_explorer_agent_online_with_node),
        ("test_explorer_agent_online_save_graph", test_explorer_agent_online_save_graph),
        ("test_explorer_agent_online_batch_success_rate", test_explorer_agent_online_batch_success_rate),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        output = io.StringIO()
        try:
            with redirect_stdout(output), redirect_stderr(output):
                test_func()
            test_output = output.getvalue()
            if test_output.strip():
                print(test_output)
            print(f"RESULT: PASSED - {test_name}")
            passed += 1
        except AssertionError as e:
            test_output = output.getvalue()
            if test_output.strip():
                print(test_output)
            print(f"RESULT: FAILED - {test_name}")
            print(f"Error: {str(e)}")
            failed += 1
        except Exception as e:
            test_output = output.getvalue()
            if test_output.strip():
                print(test_output)
            print(f"RESULT: ERROR - {test_name}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Test Summary")
    print(f"{'='*60}")
    print(f"Total: {len(tests)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("All explorer agent online tests passed!")
    else:
        print(f"Some tests failed. Total failures: {failed}")
        sys.exit(1)

