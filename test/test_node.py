import os
import sys
import tempfile
import logging
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rllm_agentzero.core.node import Node
from rllm_agentzero.core.trace import Trace
from rllm_agentzero.core.trajectory import TrajectoryStep

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def create_dummy_trace() -> Trace:
    dummy_step = TrajectoryStep(
        action="click('test')",
        parsed_action="click('test')",
        thought="testing",
        observation={"url": "http://mock", "axtree_txt": "mock"},
        misc={}
    )
    return Trace.from_trajectory_steps(
        steps=[dummy_step],
        start_url="http://start",
        end_url="http://end"
    )


def create_test_embedding(dim: int = 128) -> list[float]:
    np.random.seed(42)
    embedding = np.random.randn(dim).tolist()
    embedding = np.array(embedding)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()


def test_node_creation():
    """Test Node creation with embedding."""
    with tempfile.TemporaryDirectory() as temp_dir:
        node_exp_dir = os.path.join(temp_dir, "node_test_123")
        embedding = create_test_embedding()
        
        node = Node(
            node_id="node_0",
            url="https://www.example.com",
            embedding=embedding,
            hint="Test Node",
            tasks={},
            exploration_tasks={},
            children=[],
            prefixes=[],
            visited=False,
            exp_dir=node_exp_dir
        )
        
        assert node.node_id == "node_0"
        assert node.url == "https://www.example.com"
        assert node.embedding == embedding
        assert node.hint == "Test Node"
        assert len(node.tasks) == 0
        assert len(node.children) == 0
        assert node.visited == False


def test_node_persistence():
    """Test Node save and load."""
    with tempfile.TemporaryDirectory() as temp_dir:
        node_exp_dir = os.path.join(temp_dir, "node_test_456")
        embedding = create_test_embedding()
        dummy_trace = create_dummy_trace()
        
        node = Node(
            node_id="node_1",
            url="https://www.example.com",
            embedding=embedding,
            hint="Persistent Node",
            tasks={},
            exploration_tasks={},
            children=["child_1", "child_2"],
            prefixes=[dummy_trace],
            visited=True,
            exp_dir=node_exp_dir
        )
        
        loaded_node = Node.load(
            load_dir=node_exp_dir,
            load_steps=False,
            load_prefix=True,
            load_images=False
        )
        
        assert loaded_node.node_id == "node_1"
        assert loaded_node.url == "https://www.example.com"
        assert loaded_node.hint == "Persistent Node"
        assert loaded_node.visited == True
        assert len(loaded_node.children) == 2
        assert "child_1" in loaded_node.children
        assert "child_2" in loaded_node.children


def test_node_add_task():
    """Test adding tasks to a node."""
    with tempfile.TemporaryDirectory() as temp_dir:
        node_exp_dir = os.path.join(temp_dir, "node_test_789")
        embedding = create_test_embedding()
        
        node = Node(
            node_id="node_2",
            url="https://www.example.com",
            embedding=embedding,
            hint="Task Node",
            tasks={},
            exploration_tasks={},
            children=[],
            prefixes=[],
            visited=False,
            exp_dir=node_exp_dir
        )
        
        task = node.add_task(
            goal="Check Price",
            instruction="Navigate to price section",
            target_edge={},
            task_misc={"priority": "high"}
        )
        
        assert "Check Price" in node.tasks
        assert node.tasks["Check Price"] == task
        assert task.goal == "Check Price"


def test_node_add_prefix():
    """Test adding prefix traces to a node."""
    with tempfile.TemporaryDirectory() as temp_dir:
        node_exp_dir = os.path.join(temp_dir, "node_test_prefix")
        embedding = create_test_embedding()
        
        node = Node(
            node_id="node_3",
            url="https://www.example.com",
            embedding=embedding,
            hint="Prefix Node",
            tasks={},
            exploration_tasks={},
            children=[],
            prefixes=[],
            visited=False,
            exp_dir=node_exp_dir
        )
        
        trace1 = create_dummy_trace()
        trace2 = create_dummy_trace()
        
        node.add_prefix(trace1)
        node.add_prefix(trace2)
        
        assert len(node.prefixes) == 2
        
        loaded_node = Node.load(
            load_dir=node_exp_dir,
            load_steps=False,
            load_prefix=True,
            load_images=False
        )
        
        assert len(loaded_node.prefixes) == 2


def test_node_action_history():
    """Test action history recording."""
    with tempfile.TemporaryDirectory() as temp_dir:
        node_exp_dir = os.path.join(temp_dir, "node_test_history")
        embedding = create_test_embedding()
        
        node = Node(
            node_id="node_4",
            url="https://www.example.com",
            embedding=embedding,
            hint="History Node",
            tasks={},
            exploration_tasks={},
            children=[],
            prefixes=[],
            visited=False,
            exp_dir=node_exp_dir
        )
        
        node.record_action("click(button_1)")
        node.record_action("click(button_2)")
        node.record_action("click(button_1)")
        
        assert node.action_history["click(button_1)"] == 2
        assert node.action_history["click(button_2)"] == 1
        
        loaded_node = Node.load(
            load_dir=node_exp_dir,
            load_steps=False,
            load_prefix=False,
            load_images=False
        )
        
        assert loaded_node.action_history["click(button_1)"] == 2
        assert loaded_node.action_history["click(button_2)"] == 1


def test_node_children_management():
    """Test children node management."""
    with tempfile.TemporaryDirectory() as temp_dir:
        node_exp_dir = os.path.join(temp_dir, "node_test_children")
        embedding = create_test_embedding()
        
        node = Node(
            node_id="node_5",
            url="https://www.example.com",
            embedding=embedding,
            hint="Parent Node",
            tasks={},
            exploration_tasks={},
            children=[],
            prefixes=[],
            visited=False,
            exp_dir=node_exp_dir
        )
        
        node.children.append("child_node_1")
        node.children.append("child_node_2")
        node.update_save(save_info=True, save_prefix=False)
        
        loaded_node = Node.load(
            load_dir=node_exp_dir,
            load_steps=False,
            load_prefix=False,
            load_images=False
        )
        
        assert len(loaded_node.children) == 2
        assert "child_node_1" in loaded_node.children
        assert "child_node_2" in loaded_node.children


def test_node_embedding_persistence():
    """Test that embedding is correctly saved and loaded."""
    with tempfile.TemporaryDirectory() as temp_dir:
        node_exp_dir = os.path.join(temp_dir, "node_test_embedding")
        embedding = create_test_embedding(dim=64)
        
        node = Node(
            node_id="node_6",
            url="https://www.example.com",
            embedding=embedding,
            hint="Embedding Node",
            tasks={},
            exploration_tasks={},
            children=[],
            prefixes=[],
            visited=False,
            exp_dir=node_exp_dir
        )
        
        loaded_node = Node.load(
            load_dir=node_exp_dir,
            load_steps=False,
            load_prefix=False,
            load_images=False
        )
        
        assert loaded_node.embedding is not None
        assert len(loaded_node.embedding) == 64
        assert abs(np.linalg.norm(loaded_node.embedding) - 1.0) < 0.001


def test_node_exploration_tasks():
    """Test exploration task management."""
    with tempfile.TemporaryDirectory() as temp_dir:
        node_exp_dir = os.path.join(temp_dir, "node_test_exploration")
        embedding = create_test_embedding()
        
        node = Node(
            node_id="node_7",
            url="https://www.example.com",
            embedding=embedding,
            hint="Exploration Node",
            tasks={},
            exploration_tasks={},
            children=[],
            prefixes=[],
            visited=False,
            exp_dir=node_exp_dir
        )
        
        exp_task = node.add_exploration_task(
            goal="Explore new page",
            instruction="Navigate to unknown area",
            target_edge={}
        )
        
        assert "Explore new page" in node.exploration_tasks
        assert node.exploration_tasks["Explore new page"] == exp_task


def test_node_visited_state():
    """Test visited state persistence."""
    with tempfile.TemporaryDirectory() as temp_dir:
        node_exp_dir = os.path.join(temp_dir, "node_test_visited")
        embedding = create_test_embedding()
        
        node = Node(
            node_id="node_8",
            url="https://www.example.com",
            embedding=embedding,
            hint="Visited Node",
            tasks={},
            exploration_tasks={},
            children=[],
            prefixes=[],
            visited=False,
            exp_dir=node_exp_dir
        )
        
        node.visited = True
        node.update_save(save_info=True, save_prefix=False)
        
        loaded_node = Node.load(
            load_dir=node_exp_dir,
            load_steps=False,
            load_prefix=False,
            load_images=False
        )
        
        assert loaded_node.visited == True


if __name__ == "__main__":
    test_node_creation()
    test_node_persistence()
    test_node_add_task()
    test_node_add_prefix()
    test_node_action_history()
    test_node_children_management()
    test_node_embedding_persistence()
    test_node_exploration_tasks()
    test_node_visited_state()
    print("All node tests passed!")
