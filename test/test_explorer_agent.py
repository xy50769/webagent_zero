import sys
import os
from unittest.mock import Mock, MagicMock
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rllm_agentzero.agents_old.explorer_agent import ExplorerAgent
from rllm_agentzero.agents_old.server.llm_engine import LLMEngine
from rllm_agentzero.agents_old.trajectory_data import BrowserGymAgentStepData
from rllm_agentzero.core.node import Node


def create_mock_llm_engine():
    """Create a mock LLMEngine for offline testing."""
    mock_llm = Mock(spec=LLMEngine)
    mock_llm.generate.return_value = '{"thought": "I will explore the page", "action": "click(\'12\')"}'
    return mock_llm


def create_mock_node():
    """Create a mock Node for testing."""
    mock_node = Mock(spec=Node)
    mock_node.action_history = Counter({
        "click('10')": 2,
        "click('11')": 1,
        "type('search', 'test')": 1
    })
    return mock_node


def test_explorer_agent_initialization():
    """Test ExplorerAgent initialization."""
    mock_llm = create_mock_llm_engine()
    agent = ExplorerAgent(llm_engine=mock_llm)
    
    assert agent.llm == mock_llm
    assert agent.action_set is not None
    assert agent.prompt_builder is not None
    assert len(agent.history) == 0
    assert "Explore the website" in agent._goal


def test_explorer_agent_reset():
    """Test ExplorerAgent reset method."""
    mock_llm = create_mock_llm_engine()
    agent = ExplorerAgent(llm_engine=mock_llm)
    
    agent.history.append(BrowserGymAgentStepData(
        action="click('12')",
        thought="Test thought",
        axtree="test axtree"
    ))
    assert len(agent.history) == 1
    
    agent.reset()
    assert len(agent.history) == 0


def test_obs_preprocessor():
    """Test obs_preprocessor method."""
    from unittest.mock import patch
    
    mock_llm = create_mock_llm_engine()
    agent = ExplorerAgent(llm_engine=mock_llm)
    
    obs = {
        "axtree_object": Mock(),
        "last_action_error": "Test error",
        "url": "https://example.com",
        "extra_element_properties": {}
    }
    
    with patch('rllm_agentzero.agents.explorer_agent.flatten_axtree_to_str', return_value="Mock axtree text"):
        result = agent.obs_preprocessor(obs)
    
    assert "axtree_txt" in result
    assert result["axtree_txt"] == "Mock axtree text"
    assert result["last_action_error"] == "Test error"
    assert result["url"] == "https://example.com"
    assert "goal_object" in result
    print(f"Processed obs keys: {list(result.keys())}")


def test_action_processor():
    """Test action_processor method."""
    mock_llm = create_mock_llm_engine()
    agent = ExplorerAgent(llm_engine=mock_llm)
    
    action = "click('12')"
    result = agent.action_processor(action)
    
    assert isinstance(result, str)
    print(f"Processed action: {result}")


def test_parse_output_json_format():
    """Test _parse_output with JSON format."""
    mock_llm = create_mock_llm_engine()
    agent = ExplorerAgent(llm_engine=mock_llm)
    
    response = '{"thought": "I need to click the button", "action": "click(\'12\')"}'
    thought, action = agent._parse_output(response)
    
    assert thought == "I need to click the button"
    assert action == "click('12')"
    print(f"Parsed thought: {thought}, action: {action}")


def test_parse_output_thought_action_format():
    """Test _parse_output with Thought: ... Action: ... format."""
    mock_llm = create_mock_llm_engine()
    agent = ExplorerAgent(llm_engine=mock_llm)
    
    response = "Thought: I should explore this page\nAction: click('13')"
    thought, action = agent._parse_output(response)
    
    assert "explore" in thought.lower()
    assert "click('13')" in action
    print(f"Parsed thought: {thought}, action: {action}")


def test_parse_output_fallback():
    """Test _parse_output fallback behavior."""
    mock_llm = create_mock_llm_engine()
    agent = ExplorerAgent(llm_engine=mock_llm)
    
    response = "Just some random text without structure"
    thought, action = agent._parse_output(response)
    
    assert thought == "Exploration"
    assert action == response
    print(f"Fallback thought: {thought}, action: {action}")


def test_get_action_basic():
    """Test get_action with basic inputs."""
    mock_llm = create_mock_llm_engine()
    agent = ExplorerAgent(llm_engine=mock_llm)
    
    obs = {
        "axtree_txt": "Button: Submit, Link: Home",
        "last_action_error": None
    }
    
    response, extras = agent.get_action(obs)
    
    assert isinstance(response, str)
    assert "parsed_action" in extras
    assert "thought" in extras
    assert len(agent.history) == 1
    print(f"Response: {response}")
    print(f"Extras: {extras}")


def test_get_action_with_node():
    """Test get_action with node containing action history."""
    mock_llm = create_mock_llm_engine()
    agent = ExplorerAgent(llm_engine=mock_llm)
    
    mock_node = create_mock_node()
    obs = {
        "axtree_txt": "Button: Submit",
        "last_action_error": None
    }
    
    response, extras = agent.get_action(obs, node=mock_node)
    
    assert isinstance(response, str)
    assert "parsed_action" in extras
    assert len(agent.history) == 1
    print(f"Response with node: {response}")
    print(f"Extras: {extras}")


def test_get_action_history_accumulation():
    """Test that get_action accumulates history."""
    mock_llm = create_mock_llm_engine()
    agent = ExplorerAgent(llm_engine=mock_llm)
    
    obs = {
        "axtree_txt": "Button: Submit",
        "last_action_error": None
    }
    
    assert len(agent.history) == 0
    
    agent.get_action(obs)
    assert len(agent.history) == 1
    
    agent.get_action(obs)
    assert len(agent.history) == 2
    
    print(f"History length after 2 actions: {len(agent.history)}")
    print(f"First step thought: {agent.history[0].thought}")
    print(f"Second step thought: {agent.history[1].thought}")


def test_get_action_with_error():
    """Test get_action with last action error."""
    mock_llm = create_mock_llm_engine()
    agent = ExplorerAgent(llm_engine=mock_llm)
    
    obs = {
        "axtree_txt": "Button: Submit",
        "last_action_error": "Element not found"
    }
    
    response, extras = agent.get_action(obs)
    
    assert isinstance(response, str)
    assert len(agent.history) == 1
    assert agent.history[0].last_action_error == "Element not found"
    print(f"Response with error: {response}")


def test_get_action_step_data_structure():
    """Test that get_action creates proper step data structure."""
    mock_llm = create_mock_llm_engine()
    agent = ExplorerAgent(llm_engine=mock_llm)
    
    obs = {
        "axtree_txt": "Button: Submit, Link: Home",
        "last_action_error": None
    }
    
    agent.get_action(obs)
    
    step_data = agent.history[0]
    assert isinstance(step_data, BrowserGymAgentStepData)
    assert step_data.action is not None
    assert step_data.thought is not None
    assert step_data.axtree == "Button: Submit, Link: Home"
    assert step_data.misc is not None
    assert "parsed_action" in step_data.misc
    assert "raw_output" in step_data.misc
    print(f"Step data action: {step_data.action}")
    print(f"Step data thought: {step_data.thought}")
    print(f"Step data misc: {step_data.misc}")


def test_get_action_llm_called_correctly():
    """Test that LLM generate is called with correct parameters."""
    mock_llm = create_mock_llm_engine()
    agent = ExplorerAgent(llm_engine=mock_llm)
    
    obs = {
        "axtree_txt": "Button: Submit",
        "last_action_error": None
    }
    
    agent.get_action(obs)
    
    assert mock_llm.generate.called
    call_args = mock_llm.generate.call_args
    assert call_args[1]["mode"] == "base"
    assert call_args[1]["temperature"] == 1.0
    assert "system_msg" in call_args[1]
    assert "user_msg" in call_args[1]
    print(f"LLM generate called with mode: {call_args[1]['mode']}")
    print(f"LLM generate called with temperature: {call_args[1]['temperature']}")


def test_get_action_with_oracle_action():
    """Test get_action with oracle_action parameter."""
    mock_llm = create_mock_llm_engine()
    agent = ExplorerAgent(llm_engine=mock_llm)
    
    obs = {
        "axtree_txt": "Button: Submit",
        "last_action_error": None
    }
    
    response, extras = agent.get_action(obs, oracle_action=("click('99')", "Oracle thought"))
    
    assert isinstance(response, str)
    print(f"Response with oracle: {response}")


def test_explorer_agent_config():
    """Test that ExplorerAgent has proper config."""
    mock_llm = create_mock_llm_engine()
    agent = ExplorerAgent(llm_engine=mock_llm)
    
    config = agent.get_config()
    assert "name" in config
    assert config["name"] == "ExplorerAgent"
    print(f"Agent config: {config}")


def test_explorer_agent_goal():
    """Test that ExplorerAgent has correct goal."""
    mock_llm = create_mock_llm_engine()
    agent = ExplorerAgent(llm_engine=mock_llm)
    
    assert agent._goal is not None
    assert "Explore" in agent._goal
    print(f"Agent goal: {agent._goal}")


def test_explorer_agent_with_real_browser():
    """Test ExplorerAgent with real browser environment but mock LLM."""
    import gymnasium as gym
    import tempfile
    from browsergym.core.task import AbstractBrowserTask
    from browsergym.core.env import BrowserEnv
    
    class ExplorerTestTask(AbstractBrowserTask):
        """Test task for ExplorerAgent integration test."""
        def setup(self, page):
            page.goto("https://www.example.com")
            return "Explore the example.com website", {}
        
        def validate(self, page, chat_messages):
            return 0.0, False, "", {}
        
        def teardown(self):
            pass
    
    if "browsergym/explorer_test" not in gym.envs.registry:
        gym.register(
            id="browsergym/explorer_test",
            entry_point="browsergym.core.env:BrowserEnv",
            kwargs={"task_entrypoint": ExplorerTestTask}
        )
    
    mock_llm = create_mock_llm_engine()
    agent = ExplorerAgent(llm_engine=mock_llm)
    
    env = None
    try:
        env = gym.make("browsergym/explorer_test", headless=False)
        obs, info = env.reset()
        
        print(f"Browser opened, current URL: {obs.get('url', 'unknown')}")
        print(f"Observation keys: {list(obs.keys())}")
        
        processed_obs = agent.obs_preprocessor(obs)
        print(f"Processed observation has axtree_txt: {'axtree_txt' in processed_obs}")
        print(f"Axtree length: {len(processed_obs.get('axtree_txt', ''))}")
        
        response, extras = agent.get_action(processed_obs)
        
        print(f"Agent response: {response[:100]}...")
        print(f"Parsed action: {extras.get('parsed_action', '')[:100]}...")
        print(f"Thought: {extras.get('thought', '')}")
        
        assert isinstance(response, str)
        assert "parsed_action" in extras
        assert len(agent.history) == 1
        
        print("Successfully got action from ExplorerAgent with real browser")
        
    finally:
        if env:
            env.close()
            print("Browser closed")


def test_explorer_agent_multiple_steps_with_real_browser():
    """Test ExplorerAgent with multiple steps using real browser."""
    import gymnasium as gym
    from browsergym.core.task import AbstractBrowserTask
    from browsergym.core.env import BrowserEnv
    
    class ExplorerTestTask(AbstractBrowserTask):
        """Test task for ExplorerAgent multi-step test."""
        def setup(self, page):
            page.goto("https://www.example.com")
            return "Explore the example.com website", {}
        
        def validate(self, page, chat_messages):
            return 0.0, False, "", {}
        
        def teardown(self):
            pass
    
    if "browsergym/explorer_test" not in gym.envs.registry:
        gym.register(
            id="browsergym/explorer_test",
            entry_point="browsergym.core.env:BrowserEnv",
            kwargs={"task_entrypoint": ExplorerTestTask}
        )
    
    mock_llm = create_mock_llm_engine()
    agent = ExplorerAgent(llm_engine=mock_llm)
    
    env = None
    try:
        env = gym.make("browsergym/explorer_test", headless=False)
        obs, info = env.reset()
        
        print(f"Starting multi-step test with real browser")
        print(f"Initial URL: {obs.get('url', 'unknown')}")
        
        for step in range(3):
            processed_obs = agent.obs_preprocessor(obs)
            response, extras = agent.get_action(processed_obs)
            
            print(f"Step {step + 1}: Got action - {extras.get('parsed_action', '')[:50]}...")
            print(f"History length: {len(agent.history)}")
            
            if step < 2:
                obs, reward, terminated, truncated, info = env.step(extras.get('parsed_action', 'noop()'))
                print(f"Step {step + 1}: New URL: {obs.get('url', 'unknown')}")
        
        assert len(agent.history) == 3
        print("Successfully completed 3 steps with real browser")
        
    finally:
        if env:
            env.close()
            print("Browser closed")


if __name__ == "__main__":
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    tests = [
        ("test_explorer_agent_initialization", test_explorer_agent_initialization),
        ("test_explorer_agent_reset", test_explorer_agent_reset),
        ("test_obs_preprocessor", test_obs_preprocessor),
        ("test_action_processor", test_action_processor),
        ("test_parse_output_json_format", test_parse_output_json_format),
        ("test_parse_output_thought_action_format", test_parse_output_thought_action_format),
        ("test_parse_output_fallback", test_parse_output_fallback),
        ("test_get_action_basic", test_get_action_basic),
        ("test_get_action_with_node", test_get_action_with_node),
        ("test_get_action_history_accumulation", test_get_action_history_accumulation),
        ("test_get_action_with_error", test_get_action_with_error),
        ("test_get_action_step_data_structure", test_get_action_step_data_structure),
        ("test_get_action_llm_called_correctly", test_get_action_llm_called_correctly),
        ("test_get_action_with_oracle_action", test_get_action_with_oracle_action),
        ("test_explorer_agent_config", test_explorer_agent_config),
        ("test_explorer_agent_goal", test_explorer_agent_goal),
        ("test_explorer_agent_with_real_browser", test_explorer_agent_with_real_browser),
        ("test_explorer_agent_multiple_steps_with_real_browser", test_explorer_agent_multiple_steps_with_real_browser),
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
        print("All explorer agent tests passed!")
    else:
        print(f"Some tests failed. Total failures: {failed}")
        sys.exit(1)

