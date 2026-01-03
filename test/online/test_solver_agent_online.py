import os
import sys
import tempfile
import logging
import gymnasium as gym
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rllm_agentzero.agents.solver_agent import SolverAgent
from browsergym.core.task import AbstractBrowserTask
from browsergym.core.env import BrowserEnv

logging.basicConfig(level=logging.INFO)
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
    
    def generate(self, system_msg: str, user_msg: str, mode: str = "base", temperature: float = 0.01) -> str:
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


class SolverTestTask(AbstractBrowserTask):
    """Test task for SolverAgent online test."""
    def setup(self, page):
        test_url = os.getenv("TEST_URL", "http://3.148.75.200:7770/")
        page.goto(test_url)
        # Solver 接收的是具体的任务指令，而不是 Explorer 的开放式探索
        return "Click on the 'Beauty & Personal Care' category to explore beauty products.", {}
    
    def validate(self, page, chat_messages):
        return 0.0, False, "", {}
    
    def teardown(self):
        pass


def test_solver_agent_online_single_step():
    """Test SolverAgent with remote LLM and local browser - single step."""
    logger.info("\n" + "="*60)
    logger.info("Starting SolverAgent Online Single Step Test")
    logger.info("="*60)
    
    # Create remote LLM engine
    llm_engine = create_remote_llm_engine()
    
    # Create SolverAgent
    agent = SolverAgent(llm_engine=llm_engine, temperature=0.01)
    
    # Register test environment
    env_id = "browsergym/solver_test"
    if env_id in gym.envs.registry:
        gym.envs.registry.pop(env_id)
    gym.register(
        id=env_id,
        entry_point="browsergym.core.env:BrowserEnv",
        kwargs={"task_entrypoint": SolverTestTask}
    )
    
    env = None
    try:
        # Create environment
        headless = os.getenv("HEADLESS", "False").lower() == "true"
        env = gym.make(env_id, headless=headless)
        obs, info = env.reset()
        
        logger.info(f"Browser opened, current URL: {obs.get('url', 'unknown')}")
        logger.info(f"Observation keys: {list(obs.keys())}")
        
        # Preprocess observation
        processed_obs = agent.obs_preprocessor(obs)
        logger.info(f"Processed observation has axtree_txt: {'axtree_txt' in processed_obs}")
        logger.info(f"Axtree length: {len(processed_obs.get('axtree_txt', ''))}")
        
        # Get action from agent
        logger.info(f"Calling agent.get_action() with remote LLM via SSH tunnel...")
        response, extras = agent.get_action(processed_obs)
        
        raw_action = extras.get('raw_action', '')
        parsed_action = extras.get('parsed_action', '')
        thought = extras.get('thought', '')
        
        logger.info(f"Agent response length: {len(response)}")
        logger.info(f"Agent response preview: {response[:200]}...")
        logger.info(f"Raw action: {raw_action}")
        logger.info(f"Parsed action preview: {parsed_action[:150]}...")
        logger.info(f"Thought: {thought}")
        
        # Verify we got valid outputs
        assert response is not None and len(response) > 0, "Empty response from agent"
        assert "raw_action" in extras, "Missing raw_action in extras"
        assert "parsed_action" in extras, "Missing parsed_action in extras"
        assert "thought" in extras, "Missing thought in extras"
        
        # Verify parsing correctness
        assert raw_action is not None and len(raw_action) > 0, f"Empty raw_action: {raw_action}"
        assert thought is not None and len(thought) > 0, f"Empty thought: {thought}"
        
        # Verify action format (should be like click('123') or scroll(0, 200))
        assert "(" in raw_action and ")" in raw_action, f"Invalid action format: {raw_action}"
        
        logger.info(f"✅ Parsing verification passed!")
        logger.info(f"✅ Successfully got action from SolverAgent with remote LLM and local browser")
        
    finally:
        if env:
            env.close()
            logger.info("Browser closed")


def test_solver_agent_online_multiple_steps():
    """Test SolverAgent with remote LLM - multiple steps."""
    logger.info("\n" + "="*60)
    logger.info("Starting SolverAgent Online Multiple Steps Test")
    logger.info("="*60)
    
    llm_engine = create_remote_llm_engine()
    agent = SolverAgent(llm_engine=llm_engine, temperature=0.01)
    
    env_id = "browsergym/solver_multi_step_test"
    if env_id in gym.envs.registry:
        gym.envs.registry.pop(env_id)
    gym.register(
        id=env_id,
        entry_point="browsergym.core.env:BrowserEnv",
        kwargs={"task_entrypoint": SolverTestTask}
    )
    
    env = None
    try:
        headless = os.getenv("HEADLESS", "False").lower() == "true"
        env = gym.make(env_id, headless=headless)
        obs, info = env.reset()
        
        logger.info(f"Initial URL: {obs.get('url', 'unknown')}")
        
        max_steps = 3
        logger.info(f"Running {max_steps} steps...")
        
        for step in range(max_steps):
            logger.info(f"\n{'='*60}")
            logger.info(f"Step {step + 1}/{max_steps}")
            logger.info(f"{'='*60}")
            
            processed_obs = agent.obs_preprocessor(obs)
            logger.info(f"Getting action from agent...")
            
            response, extras = agent.get_action(processed_obs)
            
            raw_action = extras.get('raw_action', '')
            parsed_action = extras.get('parsed_action', '')
            thought = extras.get('thought', '')
            
            logger.info(f"Step {step + 1} - Thought: {thought}")
            logger.info(f"Step {step + 1} - Raw Action: {raw_action}")
            logger.info(f"History length: {len(agent.history)}")
            
            if step < max_steps - 1:
                logger.info(f"Executing action...")
                try:
                    # Set action_mapping for BrowserEnv
                    browser_env = env.unwrapped if hasattr(env, "unwrapped") else env
                    if hasattr(browser_env, "action_mapping"):
                        browser_env.action_mapping = agent.action_processor
                    
                    # Pass raw_action (action string) to env.step()
                    obs, reward, terminated, truncated, info = env.step(raw_action)
                    logger.info(f"Step {step + 1} - New URL: {obs.get('url', 'unknown')}")
                    logger.info(f"Step {step + 1} - Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
                    
                    if terminated or truncated:
                        logger.info(f"Episode ended at step {step + 1}")
                        break
                except Exception as e:
                    logger.error(f"Error executing action: {e}")
                    import traceback
                    traceback.print_exc()
                    break
        
        logger.info(f"\nCompleted {len(agent.history)} steps")
        
    finally:
        if env:
            env.close()
            logger.info("Browser closed")


def test_solver_agent_online_with_task():
    """Test SolverAgent with a specific task."""
    logger.info("\n" + "="*60)
    logger.info("Starting SolverAgent Online Task Test")
    logger.info("="*60)
    
    # Create a more specific task
    class SpecificTask(AbstractBrowserTask):
        def setup(self, page):
            test_url = os.getenv("TEST_URL", "http://3.148.75.200:7770/")
            page.goto(test_url)
            # 给 Solver 一个更具体的任务
            return "Find and click on the product 'V8 +Energy' in the product list.", {}
        
        def validate(self, page, chat_messages):
            # 简单验证：检查 URL 是否变化
            current_url = page.url
            if current_url != "http://3.148.75.200:7770/":
                return 1.0, True, "Successfully navigated", {}
            return 0.0, False, "", {}
        
        def teardown(self):
            pass
    
    llm_engine = create_remote_llm_engine()
    agent = SolverAgent(llm_engine=llm_engine, temperature=0.01)
    
    env_id = "browsergym/solver_specific_task"
    if env_id in gym.envs.registry:
        gym.envs.registry.pop(env_id)
    gym.register(
        id=env_id,
        entry_point="browsergym.core.env:BrowserEnv",
        kwargs={"task_entrypoint": SpecificTask}
    )
    
    env = None
    try:
        headless = os.getenv("HEADLESS", "False").lower() == "true"
        env = gym.make(env_id, headless=headless)
        obs, info = env.reset()
        
        logger.info(f"Task goal: {obs.get('goal', 'N/A')}")
        logger.info(f"Initial URL: {obs.get('url', 'unknown')}")
        
        # Execute one action
        processed_obs = agent.obs_preprocessor(obs)
        response, extras = agent.get_action(processed_obs)
        
        raw_action = extras.get('raw_action', '')
        thought = extras.get('thought', '')
        
        logger.info(f"Thought: {thought}")
        logger.info(f"Raw Action: {raw_action}")
        
        # Execute action
        browser_env = env.unwrapped if hasattr(env, "unwrapped") else env
        if hasattr(browser_env, "action_mapping"):
            browser_env.action_mapping = agent.action_processor
        
        obs_new, reward, terminated, truncated, info = env.step(raw_action)
        
        logger.info(f"Reward: {reward}")
        logger.info(f"New URL: {obs_new.get('url', 'unknown')}")
        logger.info(f"Terminated: {terminated}, Truncated: {truncated}")
        
        # Validate
        if reward > 0 or obs_new.get('url') != obs.get('url'):
            logger.info("Task execution successful - page changed!")
        else:
            logger.info("Task execution completed but page didn't change")
        
    finally:
        if env:
            env.close()
            logger.info("Browser closed")


if __name__ == "__main__":
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    tests = [
        ("test_solver_agent_online_single_step", test_solver_agent_online_single_step),
        ("test_solver_agent_online_multiple_steps", test_solver_agent_online_multiple_steps),
        ("test_solver_agent_online_with_task", test_solver_agent_online_with_task),
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
        print("All solver agent online tests passed!")
    else:
        print(f"Some tests failed. Total failures: {failed}")
        sys.exit(1)

