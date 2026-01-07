import sys
import os
from unittest.mock import Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rllm_agentzero.agents_old.prompt_builders.explorer_prompt_builder import RLLMExplorerPromptBuilder
from rllm_agentzero.agents_old.trajectory_data import BrowserGymAgentStepData


def create_mock_action_set():
    """Create a mock AbstractActionSet for testing."""
    mock_action_set = Mock()
    mock_action_set.describe.return_value = "Mock action set description with examples."
    return mock_action_set


def test_explorer_prompt_builder_initialization():
    """Test RLLMExplorerPromptBuilder initialization."""
    mock_action_set = create_mock_action_set()
    builder = RLLMExplorerPromptBuilder(mock_action_set)
    
    assert builder.action_set == mock_action_set
    assert builder.action_set_description == "Mock action set description with examples."
    mock_action_set.describe.assert_called_once_with(with_long_description=True, with_examples=True)


def test_construct_explorer_prompt_basic():
    """Test construct_explorer_prompt with basic inputs."""
    mock_action_set = create_mock_action_set()
    builder = RLLMExplorerPromptBuilder(mock_action_set)
    
    goal = "Test goal"
    obs = {
        "axtree_txt": "Button: Submit, Link: Home",
        "last_action_error": None
    }
    history = []
    visited_actions = None
    
    result = builder.construct_explorer_prompt(goal, obs, history, visited_actions)
    
    assert isinstance(result, str)
    assert goal in result
    assert "Button: Submit" in result
    assert "Action Space" in result


def test_construct_explorer_prompt_with_visited_actions():
    """Test construct_explorer_prompt with visited actions."""
    mock_action_set = create_mock_action_set()
    builder = RLLMExplorerPromptBuilder(mock_action_set)
    
    goal = "Test goal"
    obs = {
        "axtree_txt": "Button: Submit",
        "last_action_error": None
    }
    history = []
    visited_actions = ["click('12')", "click('13')", "click('14')"]
    
    result = builder.construct_explorer_prompt(goal, obs, history, visited_actions)
    
    assert isinstance(result, str)
    assert "IMPORTANT" in result
    assert "DO NOT repeat them" in result
    assert "click('12')" in result
    assert "click('13')" in result
    assert "click('14')" in result


def test_construct_explorer_prompt_visited_actions_limit():
    """Test that construct_explorer_prompt limits visited actions to last 5."""
    mock_action_set = create_mock_action_set()
    builder = RLLMExplorerPromptBuilder(mock_action_set)
    
    goal = "Test goal"
    obs = {
        "axtree_txt": "Button: Submit",
        "last_action_error": None
    }
    history = []
    visited_actions = [f"click('{i}')" for i in range(10)]
    
    result = builder.construct_explorer_prompt(goal, obs, history, visited_actions)
    
    assert "click('5')" in result
    assert "click('6')" in result
    assert "click('7')" in result
    assert "click('8')" in result
    assert "click('9')" in result
    assert "click('0')" not in result
    assert "click('4')" not in result


def test_construct_explorer_prompt_with_history():
    """Test construct_explorer_prompt with history."""
    mock_action_set = create_mock_action_set()
    builder = RLLMExplorerPromptBuilder(mock_action_set)
    
    goal = "Test goal"
    obs = {
        "axtree_txt": "Button: Submit",
        "last_action_error": None
    }
    history = [
        BrowserGymAgentStepData(
            action="action_1",
            thought="thought_1",
            axtree="axtree_1",
            misc={'parsed_action': 'action_1'}
        ),
        BrowserGymAgentStepData(
            action="action_2",
            thought="thought_2",
            axtree="axtree_2",
            misc={'parsed_action': 'action_2'}
        )
    ]
    visited_actions = None
    
    result = builder.construct_explorer_prompt(goal, obs, history, visited_actions)
    
    assert "History of past actions" in result
    assert "thought_1" in result or "action_1" in result
    assert "thought_2" in result or "action_2" in result


def test_construct_explorer_prompt_with_error():
    """Test construct_explorer_prompt with last action error."""
    mock_action_set = create_mock_action_set()
    builder = RLLMExplorerPromptBuilder(mock_action_set)
    
    goal = "Test goal"
    obs = {
        "axtree_txt": "Button: Submit",
        "last_action_error": "Element not found"
    }
    history = []
    visited_actions = None
    
    result = builder.construct_explorer_prompt(goal, obs, history, visited_actions)
    
    assert "Error message from last action" in result
    assert "Element not found" in result


def test_construct_explorer_prompt_empty_obs():
    """Test construct_explorer_prompt with empty observation."""
    mock_action_set = create_mock_action_set()
    builder = RLLMExplorerPromptBuilder(mock_action_set)
    
    goal = "Test goal"
    obs = {}
    history = []
    visited_actions = None
    
    result = builder.construct_explorer_prompt(goal, obs, history, visited_actions)
    
    assert isinstance(result, str)
    assert goal in result


def test_construct_explorer_prompt_empty_visited_actions():
    """Test construct_explorer_prompt with empty visited actions list."""
    mock_action_set = create_mock_action_set()
    builder = RLLMExplorerPromptBuilder(mock_action_set)
    
    goal = "Test goal"
    obs = {
        "axtree_txt": "Button: Submit",
        "last_action_error": None
    }
    history = []
    visited_actions = []
    
    result = builder.construct_explorer_prompt(goal, obs, history, visited_actions)
    
    assert isinstance(result, str)
    assert "IMPORTANT" not in result


def test_construct_explorer_prompt_inheritance():
    """Test that RLLMExplorerPromptBuilder inherits methods from SolverPromptBuilder."""
    mock_action_set = create_mock_action_set()
    builder = RLLMExplorerPromptBuilder(mock_action_set)
    
    assert hasattr(builder, 'format_thought_and_action')
    assert hasattr(builder, 'system_message')
    assert hasattr(builder, 'goal_message')
    assert hasattr(builder, 'axtree_message')
    
    result = builder.format_thought_and_action("test thought", "test action")
    assert "test thought" in result
    assert "test action" in result


def test_construct_explorer_prompt_user_content_structure():
    """Test that construct_explorer_prompt returns properly structured user content."""
    mock_action_set = create_mock_action_set()
    builder = RLLMExplorerPromptBuilder(mock_action_set)
    
    goal = "Test goal"
    obs = {
        "axtree_txt": "Button: Submit",
        "last_action_error": None
    }
    history = []
    visited_actions = ["click('12')"]
    
    result = builder.construct_explorer_prompt(goal, obs, history, visited_actions)
    
    print(f"Generated prompt length: {len(result)} characters")
    print(f"Prompt contains {len(result.split('\\n\\n'))} sections")
    
    assert "\n\n" in result
    parts = result.split("\n\n")
    assert len(parts) > 1
    assert any("Goal" in part for part in parts)
    assert any("Action Space" in part for part in parts)
    assert any("IMPORTANT" in part for part in parts)


if __name__ == "__main__":
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    tests = [
        ("test_explorer_prompt_builder_initialization", test_explorer_prompt_builder_initialization),
        ("test_construct_explorer_prompt_basic", test_construct_explorer_prompt_basic),
        ("test_construct_explorer_prompt_with_visited_actions", test_construct_explorer_prompt_with_visited_actions),
        ("test_construct_explorer_prompt_visited_actions_limit", test_construct_explorer_prompt_visited_actions_limit),
        ("test_construct_explorer_prompt_with_history", test_construct_explorer_prompt_with_history),
        ("test_construct_explorer_prompt_with_error", test_construct_explorer_prompt_with_error),
        ("test_construct_explorer_prompt_empty_obs", test_construct_explorer_prompt_empty_obs),
        ("test_construct_explorer_prompt_empty_visited_actions", test_construct_explorer_prompt_empty_visited_actions),
        ("test_construct_explorer_prompt_inheritance", test_construct_explorer_prompt_inheritance),
        ("test_construct_explorer_prompt_user_content_structure", test_construct_explorer_prompt_user_content_structure),
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
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Test Summary")
    print(f"{'='*60}")
    print(f"Total: {len(tests)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("All explorer prompt builder tests passed!")
    else:
        print(f"Some tests failed. Total failures: {failed}")
        sys.exit(1)

