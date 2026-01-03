import sys
import os
import json
from unittest.mock import Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rllm_agentzero.agents.prompt_builders.solver_prompt_builder import SolverPromptBuilder
from rllm_agentzero.agents.trajectory_data import BrowserGymAgentStepData, BrowserGymAgentTrajectoryData


def create_mock_action_set():
    """Create a mock AbstractActionSet for testing."""
    mock_action_set = Mock()
    mock_action_set.describe.return_value = "Mock action set description with examples."
    return mock_action_set


def test_solver_prompt_builder_initialization():
    """Test SolverPromptBuilder initialization with action set."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    assert builder.action_set == mock_action_set
    assert builder.action_set_description == "Mock action set description with examples."
    mock_action_set.describe.assert_called_once_with(with_long_description=True, with_examples=True)


def test_format_thought_and_action():
    """Test format_thought_and_action method."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    result = builder.format_thought_and_action("Test thought", "click('12')")
    parsed = json.loads(result)
    assert parsed['thought'] == "Test thought"
    assert parsed['action'] == "click('12')"
    
    result_no_thought = builder.format_thought_and_action(None, "click('12')")
    parsed_no_thought = json.loads(result_no_thought)
    assert 'thought' not in parsed_no_thought
    assert parsed_no_thought['action'] == "click('12')"
    
    result_no_action = builder.format_thought_and_action("Test thought", None)
    parsed_no_action = json.loads(result_no_action)
    assert parsed_no_action['thought'] == "Test thought"
    assert 'action' not in parsed_no_action


def test_trim_axtree():
    """Test trim_axtree method."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    axtree = "A" * 100
    num_chars_overflow = 20
    result = builder.trim_axtree(axtree, num_chars_overflow)
    
    trim_str = "...trimmed due to context size limit"
    expected_length = len(axtree) - num_chars_overflow
    assert len(result) == expected_length
    assert result.endswith(trim_str)


def test_count_message_chars():
    """Test count_message_chars method."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "System message"}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "User message"}]
        }
    ]
    
    char_count = builder.count_message_chars(messages)
    expected_count = len("System message") + len("User message")
    assert char_count == expected_count


def test_trim_past_thoughts_and_actions():
    """Test trim_past_thoughts_and_actions method."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    past_thoughts = [f"thought_{i}" for i in range(10)]
    past_actions = [f"action_{i}" for i in range(10)]
    
    trimmed_thoughts, trimmed_actions = builder.trim_past_thoughts_and_actions(
        past_thoughts, past_actions, max_allowed=3
    )
    
    assert len(trimmed_thoughts) == 3
    assert len(trimmed_actions) == 3
    assert trimmed_thoughts == past_thoughts[-3:]
    assert trimmed_actions == past_actions[-3:]
    
    short_thoughts = ["thought_1", "thought_2"]
    short_actions = ["action_1", "action_2"]
    trimmed_short_thoughts, trimmed_short_actions = builder.trim_past_thoughts_and_actions(
        short_thoughts, short_actions, max_allowed=3
    )
    
    assert len(trimmed_short_thoughts) == 2
    assert len(trimmed_short_actions) == 2


def test_system_message():
    """Test system_message method."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    message = builder.system_message()
    assert message['type'] == 'text'
    assert 'UI Assistant' in message['text']
    assert 'web browser' in message['text']


def test_goal_message():
    """Test goal_message method."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    goal = "Test goal"
    message = builder.goal_message(goal)
    assert message['type'] == 'text'
    assert '# Goal' in message['text']
    assert goal in message['text']


def test_axtree_message():
    """Test axtree_message method."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    axtree = "Button: Submit, Link: Home"
    message = builder.axtree_message(axtree)
    assert message['type'] == 'text'
    assert '# Current page Accessibility Tree' in message['text']
    assert axtree in message['text']


def test_last_action_error_message():
    """Test last_action_error_message method."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    error = "Action failed: element not found"
    message = builder.last_action_error_message(error)
    assert message['type'] == 'text'
    assert '# Error message from last action' in message['text']
    assert error in message['text']


def test_action_history_messages():
    """Test action_history_messages method."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    thoughts = ["thought_1", "thought_2"]
    actions = ["action_1", "action_2"]
    message = builder.action_history_messages(thoughts, actions)
    
    assert message['type'] == 'text'
    assert '# History of past actions' in message['text']
    
    for thought, action in zip(thoughts, actions):
        formatted = builder.format_thought_and_action(thought, action)
        assert formatted in message['text']


def test_next_action_request_message():
    """Test next_action_request_message method."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    message = builder.next_action_request_message()
    assert message['type'] == 'text'
    assert '# Next action' in message['text']
    assert 'json' in message['text'].lower()
    assert 'thought' in message['text'].lower()
    assert 'action' in message['text'].lower()


def test_completion_message():
    """Test completion_message method."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    thought = "Test thought"
    action = "click('12')"
    message = builder.completion_message(thought, action)
    
    assert message['type'] == 'text'
    formatted = builder.format_thought_and_action(thought, action)
    assert message['text'] == formatted


def test_cot_examples():
    """Test cot_examples method."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    examples = builder.cot_examples()
    assert len(examples) == 3
    assert all('thought' in ex and 'action' in ex for ex in examples)


def test_build_messages_basic():
    """Test build_messages method with basic inputs."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    goal = "Test goal"
    current_step = BrowserGymAgentStepData(
        action=None,
        thought=None,
        axtree="Button: Submit",
        last_action_error=None,
        misc=None
    )
    history = []
    
    result = builder.build_messages(goal, current_step, history)
    
    assert 'prompt' in result
    assert len(result['prompt']) == 2
    assert result['prompt'][0]['role'] == 'system'
    assert result['prompt'][1]['role'] == 'user'
    assert 'completion' not in result


def test_build_messages_with_completion():
    """Test build_messages method with completion."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    goal = "Test goal"
    current_step = BrowserGymAgentStepData(
        action="click('12')",
        thought="Test thought",
        axtree="Button: Submit",
        last_action_error=None,
        misc={'parsed_action': "click('12')"}
    )
    history = []
    
    result = builder.build_messages(goal, current_step, history)
    
    assert 'prompt' in result
    assert 'completion' in result
    assert len(result['completion']) == 1
    assert result['completion'][0]['role'] == 'assistant'


def test_build_messages_with_history():
    """Test build_messages method with history."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    goal = "Test goal"
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
    current_step = BrowserGymAgentStepData(
        action="click('12')",
        thought="Test thought",
        axtree="Button: Submit",
        last_action_error=None,
        misc=None
    )
    
    result = builder.build_messages(goal, current_step, history)
    
    assert 'prompt' in result
    user_content = result['prompt'][1]['content']
    assert any('History of past actions' in part for part in user_content.split('\n'))


def test_build_messages_with_error():
    """Test build_messages method with last action error."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    goal = "Test goal"
    current_step = BrowserGymAgentStepData(
        action="click('12')",
        thought="Test thought",
        axtree="Button: Submit",
        last_action_error="Element not found",
        misc=None
    )
    history = []
    
    result = builder.build_messages(goal, current_step, history)
    
    assert 'prompt' in result
    user_content = result['prompt'][1]['content']
    assert 'Error message from last action' in user_content
    assert 'Element not found' in user_content


def test_build_messages_with_char_limit():
    """Test build_messages method with character limit."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    goal = "Test goal"
    large_axtree = "A" * 10000
    history = [
        BrowserGymAgentStepData(
            action=f"action_{i}",
            thought=f"thought_{i}",
            axtree=f"axtree_{i}",
            misc={'parsed_action': f'action_{i}'}
        ) for i in range(20)
    ]
    current_step = BrowserGymAgentStepData(
        action="click('12')",
        thought="Test thought",
        axtree=large_axtree,
        last_action_error=None,
        misc=None
    )
    
    result_no_limit = builder.build_messages(goal, current_step, history, char_limit=-1)
    result_with_limit = builder.build_messages(goal, current_step, history, char_limit=1000)
    
    assert 'prompt' in result_with_limit
    total_chars_no_limit = sum(len(msg['content']) for msg in result_no_limit['prompt'])
    total_chars_with_limit = sum(len(msg['content']) for msg in result_with_limit['prompt'])
    
    assert total_chars_with_limit < total_chars_no_limit
    assert "...trimmed due to context size limit" in result_with_limit['prompt'][1]['content']


def test_build_trajectory_messages():
    """Test build_trajectory_messages method."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    steps = [
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
    trajectory_data = BrowserGymAgentTrajectoryData(
        steps=steps,
        goal="Test goal",
        reward=1.0
    )
    
    messages = builder.build_trajectory_messages(trajectory_data)
    
    assert len(messages) == 2
    assert all('prompt' in msg for msg in messages)


def test_build_messages_parsed_action_priority():
    """Test that parsed_action takes priority over action in misc."""
    mock_action_set = create_mock_action_set()
    builder = SolverPromptBuilder(mock_action_set)
    
    goal = "Test goal"
    history = [
        BrowserGymAgentStepData(
            action="raw_action",
            thought="thought_1",
            axtree="axtree_1",
            misc={'parsed_action': 'parsed_action_1'}
        )
    ]
    current_step = BrowserGymAgentStepData(
        action="raw_action",
        thought="Test thought",
        axtree="Button: Submit",
        last_action_error=None,
        misc={'parsed_action': 'parsed_action_current'}
    )
    
    result = builder.build_messages(goal, current_step, history)
    
    assert 'prompt' in result
    user_content = result['prompt'][1]['content']
    assert 'parsed_action_1' in user_content or any('parsed_action_1' in part for part in str(user_content).split('\n'))


if __name__ == "__main__":
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    tests = [
        ("test_solver_prompt_builder_initialization", test_solver_prompt_builder_initialization),
        ("test_format_thought_and_action", test_format_thought_and_action),
        ("test_trim_axtree", test_trim_axtree),
        ("test_count_message_chars", test_count_message_chars),
        ("test_trim_past_thoughts_and_actions", test_trim_past_thoughts_and_actions),
        ("test_system_message", test_system_message),
        ("test_goal_message", test_goal_message),
        ("test_axtree_message", test_axtree_message),
        ("test_last_action_error_message", test_last_action_error_message),
        ("test_action_history_messages", test_action_history_messages),
        ("test_next_action_request_message", test_next_action_request_message),
        ("test_completion_message", test_completion_message),
        ("test_cot_examples", test_cot_examples),
        ("test_build_messages_basic", test_build_messages_basic),
        ("test_build_messages_with_completion", test_build_messages_with_completion),
        ("test_build_messages_with_history", test_build_messages_with_history),
        ("test_build_messages_with_error", test_build_messages_with_error),
        ("test_build_messages_with_char_limit", test_build_messages_with_char_limit),
        ("test_build_trajectory_messages", test_build_trajectory_messages),
        ("test_build_messages_parsed_action_priority", test_build_messages_parsed_action_priority),
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
        print("All solver prompt builder tests passed!")
    else:
        print(f"Some tests failed. Total failures: {failed}")
        sys.exit(1)

