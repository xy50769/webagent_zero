import sys
import os
import tempfile
import logging
import numpy as np
from typing import Callable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rllm_agentzero.core.graph import Graph
from rllm_agentzero.core.node import Node

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def create_mock_encoder(embedding_dim: int = 128) -> Callable[[dict], list[float]]:
    """
    Create a deterministic mock encoder for testing.
    Generates embeddings based on URL and axtree_txt content.
    """
    def encoder_fn(obs: dict) -> list[float]:
        url = obs.get("url", "")
        axtree = obs.get("axtree_txt", "")
        content = f"{url}|{axtree}"
        
        np.random.seed(hash(content) % (2**32))
        embedding = np.random.randn(embedding_dim).tolist()
        embedding = np.array(embedding)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    return encoder_fn


def test_graph_initialization():
    """Test Graph initialization with encoder function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        encoder_fn = create_mock_encoder()
        graph = Graph(
            root_url="http://start",
            exp_dir=temp_dir,
            encoder_fn=encoder_fn,
            similarity_threshold=0.95
        )
        
        assert graph.encoder_fn is not None
        assert graph.similarity_threshold == 0.95
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert graph.current_node is None


def test_process_transition_new_nodes():
    """Test process_transition creates new nodes for distinct states."""
    with tempfile.TemporaryDirectory() as temp_dir:
        encoder_fn = create_mock_encoder()
        graph = Graph(
            root_url="http://start",
            exp_dir=temp_dir,
            encoder_fn=encoder_fn,
            similarity_threshold=0.95
        )
        
        obs_t = {
            "url": "http://amazon.com/home",
            "axtree_txt": "Button: Search, Link: Home"
        }
        obs_t1 = {
            "url": "http://amazon.com/search",
            "axtree_txt": "Text: Results, Button: Filter"
        }
        
        target_node = graph.process_transition(
            obs_t=obs_t,
            action="click(search_button)",
            obs_t1=obs_t1,
            success=True,
            hint="Search page"
        )
        
        assert len(graph.nodes) == 2
        assert target_node.node_id in graph.nodes
        assert graph.current_node == target_node
        assert target_node.node_id in graph.nodes[list(graph.nodes.keys())[0]].children


def test_process_transition_matching_existing():
    """Test process_transition matches existing nodes when similar."""
    with tempfile.TemporaryDirectory() as temp_dir:
        encoder_fn = create_mock_encoder(embedding_dim=64)
        graph = Graph(
            root_url="http://start",
            exp_dir=temp_dir,
            encoder_fn=encoder_fn,
            similarity_threshold=0.99
        )
        
        obs_1 = {
            "url": "http://amazon.com/cart",
            "axtree_txt": "Button: Empty Cart"
        }
        
        obs_2 = {
            "url": "http://amazon.com/cart",
            "axtree_txt": "Button: Empty Cart"
        }
        
        node1 = graph.process_transition(
            obs_t=obs_1,
            action="noop",
            obs_t1=obs_1,
            success=True
        )
        
        node2 = graph.process_transition(
            obs_t=obs_1,
            action="noop",
            obs_t1=obs_2,
            success=True
        )
        
        assert node1.node_id == node2.node_id
        assert len(graph.nodes) == 1


def test_edge_statistics():
    """Test edge statistics calculation with Bayesian smoothing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        encoder_fn = create_mock_encoder()
        graph = Graph(
            root_url="http://start",
            exp_dir=temp_dir,
            encoder_fn=encoder_fn,
            similarity_threshold=0.95
        )
        
        obs_1 = {
            "url": "http://amazon.com/home",
            "axtree_txt": "Button: Add to Cart"
        }
        obs_2 = {
            "url": "http://amazon.com/cart",
            "axtree_txt": "Text: Item in Cart"
        }
        
        source_node = graph.process_transition(
            obs_t=obs_1,
            action="noop",
            obs_t1=obs_1,
            success=True
        )
        
        graph.current_node = source_node
        target_node = graph.process_transition(
            obs_t=obs_1,
            action="click(add_button)",
            obs_t1=obs_2,
            success=True
        )
        
        graph.current_node = source_node
        graph.process_transition(
            obs_t=obs_1,
            action="click(add_button)",
            obs_t1=obs_2,
            success=False
        )
        
        graph.current_node = source_node
        graph.process_transition(
            obs_t=obs_1,
            action="click(add_button)",
            obs_t1=obs_2,
            success=True
        )
        
        out_edges = graph.get_out_edges(source_node.node_id)
        target_edges = [e for e in out_edges if e["target_node_id"] == target_node.node_id]
        assert len(target_edges) == 1
        
        edge = target_edges[0]
        assert edge["target_node_id"] == target_node.node_id
        assert edge["total_attempts"] == 3
        
        expected_succ_rate = (2 + 1) / (3 + 2)
        assert abs(edge["succ_rate"] - expected_succ_rate) < 0.001


def test_similarity_threshold():
    """Test that similarity threshold correctly distinguishes states."""
    with tempfile.TemporaryDirectory() as temp_dir:
        encoder_fn = create_mock_encoder(embedding_dim=32)
        graph = Graph(
            root_url="http://start",
            exp_dir=temp_dir,
            encoder_fn=encoder_fn,
            similarity_threshold=0.95
        )
        
        obs_1 = {
            "url": "http://amazon.com/page1",
            "axtree_txt": "Button: Submit"
        }
        obs_2 = {
            "url": "http://amazon.com/page2",
            "axtree_txt": "Button: Cancel"
        }
        
        node1 = graph.process_transition(
            obs_t=obs_1,
            action="noop",
            obs_t1=obs_1,
            success=True
        )
        
        node2 = graph.process_transition(
            obs_t=obs_1,
            action="noop",
            obs_t1=obs_2,
            success=True
        )
        
        assert node1.node_id != node2.node_id
        assert len(graph.nodes) == 2


def test_explored_unexplored_nodes():
    """Test explored and unexplored node management."""
    with tempfile.TemporaryDirectory() as temp_dir:
        encoder_fn = create_mock_encoder()
        graph = Graph(
            root_url="http://start",
            exp_dir=temp_dir,
            encoder_fn=encoder_fn,
            similarity_threshold=0.95
        )
        
        obs_1 = {
            "url": "http://amazon.com/home",
            "axtree_txt": "Button: Search"
        }
        obs_2 = {
            "url": "http://amazon.com/search",
            "axtree_txt": "Text: Results"
        }
        
        node1 = graph.process_transition(
            obs_t=obs_1,
            action="click(search)",
            obs_t1=obs_1,
            success=True
        )
        
        node2 = graph.process_transition(
            obs_t=obs_1,
            action="click(search)",
            obs_t1=obs_2,
            success=True
        )
        
        assert node1 in graph.explored_nodes
        assert node2 not in graph.explored_nodes
        assert node2 in graph.unexplored_nodes
        
        next_node = graph.get_next_node()
        assert next_node == node2


def test_graph_persistence():
    """Test graph save and load functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        encoder_fn = create_mock_encoder()
        graph = Graph(
            root_url="http://start",
            exp_dir=temp_dir,
            encoder_fn=encoder_fn,
            similarity_threshold=0.95
        )
        
        obs_1 = {
            "url": "http://amazon.com/home",
            "axtree_txt": "Button: Search"
        }
        obs_2 = {
            "url": "http://amazon.com/search",
            "axtree_txt": "Text: Results"
        }
        
        node1 = graph.process_transition(
            obs_t=obs_1,
            action="noop",
            obs_t1=obs_1,
            success=True
        )
        
        graph.current_node = node1
        node2 = graph.process_transition(
            obs_t=obs_1,
            action="click(search)",
            obs_t1=obs_2,
            success=True
        )
        
        graph.current_node = node1
        graph.process_transition(
            obs_t=obs_1,
            action="click(search)",
            obs_t1=obs_2,
            success=False
        )
        
        loaded_graph = Graph.load(
            path=temp_dir,
            encoder_fn=encoder_fn,
            load_steps=False,
            load_prefixes=False,
            load_images=False
        )
        
        assert len(loaded_graph.nodes) == 2
        assert node1.node_id in loaded_graph.nodes
        assert node2.node_id in loaded_graph.nodes
        
        loaded_edges = loaded_graph.get_out_edges(node1.node_id)
        target_edges = [e for e in loaded_edges if e["target_node_id"] == node2.node_id]
        assert len(target_edges) == 1
        assert target_edges[0]["total_attempts"] == 2


def test_multiple_transitions():
    """Test multiple sequential transitions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        encoder_fn = create_mock_encoder()
        graph = Graph(
            root_url="http://start",
            exp_dir=temp_dir,
            encoder_fn=encoder_fn,
            similarity_threshold=0.95
        )
        
        obs_home = {"url": "http://amazon.com/home", "axtree_txt": "Home"}
        obs_search = {"url": "http://amazon.com/search", "axtree_txt": "Search"}
        obs_product = {"url": "http://amazon.com/product", "axtree_txt": "Product"}
        
        node_home = graph.process_transition(
            obs_t=obs_home,
            action="noop",
            obs_t1=obs_home,
            success=True
        )
        
        node_search = graph.process_transition(
            obs_t=obs_home,
            action="click(search)",
            obs_t1=obs_search,
            success=True
        )
        
        node_product = graph.process_transition(
            obs_t=obs_search,
            action="click(product)",
            obs_t1=obs_product,
            success=True
        )
        
        assert len(graph.nodes) == 3
        assert node_search.node_id in graph.nodes[node_home.node_id].children
        assert node_product.node_id in graph.nodes[node_search.node_id].children


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        encoder_fn = create_mock_encoder()
        graph = Graph(
            root_url="http://start",
            exp_dir=temp_dir,
            encoder_fn=encoder_fn,
            similarity_threshold=0.95
        )
        
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [1.0, 0.0, 0.0]
        sim = graph._cosine_similarity(vec_a, vec_b)
        assert abs(sim - 1.0) < 0.001
        
        vec_c = [0.0, 1.0, 0.0]
        sim_orthogonal = graph._cosine_similarity(vec_a, vec_c)
        assert abs(sim_orthogonal) < 0.001


if __name__ == "__main__":
    test_graph_initialization()
    test_process_transition_new_nodes()
    test_process_transition_matching_existing()
    test_edge_statistics()
    test_similarity_threshold()
    test_explored_unexplored_nodes()
    test_graph_persistence()
    test_multiple_transitions()
    test_cosine_similarity()
    print("All graph tests passed!")
