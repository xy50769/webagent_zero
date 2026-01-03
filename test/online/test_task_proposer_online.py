import sys
import os
import logging
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rllm_agentzero.agents.task_proposer import TaskProposer
from rllm_agentzero.agents.server.llm_engine import LLMEngine
from rllm_agentzero.core.graph import Graph
from rllm_agentzero.core.node import Node

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
            content = result.get("text", "").strip()
            if not content:
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
    api_base = os.getenv("LLM_API_BASE", "http://127.0.0.1:6006")
    
    logger.info(f"Creating RemoteLLMEngine")
    logger.info(f"API Base (via SSH tunnel): {api_base}")
    logger.info(f"Server will use model configured in server/llm_engine.py")
    
    return RemoteLLMEngine(api_base=api_base)


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


def load_graph_from_directory(graph_dir: str) -> Optional[Graph]:
    """
    Load a previously saved Graph from disk.
    
    Args:
        graph_dir: Path to the directory containing the graph (should contain a 'graph' subdirectory)
    
    Returns:
        Graph object if loading succeeds, None otherwise
    """
    try:
        encoder_fn = create_mock_encoder()
        
        if not os.path.exists(os.path.join(graph_dir, "graph")):
            logger.error(f"Graph directory not found: {graph_dir}/graph")
            return None
        
        logger.info(f"Loading graph from: {graph_dir}")
        graph = Graph.load(
            path=os.path.join(graph_dir, "graph"),
            encoder_fn=encoder_fn,
            load_steps=True,
            load_prefixes=True,
            load_images=False
        )
        
        logger.info(f"Successfully loaded graph:")
        logger.info(f"  - Nodes: {len(graph.nodes)}")
        logger.info(f"  - Edges: {len(graph.edges)}")
        logger.info(f"  - Explored: {len(graph.explored_nodes)}")
        logger.info(f"  - Unexplored: {len(graph.unexplored_nodes)}")
        
        return graph
    except Exception as e:
        logger.error(f"Failed to load graph from {graph_dir}: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_mock_axtree(node: Node) -> str:
    """
    Generate a mock AxTree for testing.
    In a real scenario, this would come from the browser environment.
    """
    mock_axtree = f"""
[Document] url={node.url}
  [Main]
    [Navigation Menu]
      [MenuItem] BID=100 "Home"
      [MenuItem] BID=200 "Products"
      [MenuItem] BID=300 "About"
    [Content Area]
      [Heading] "Welcome to our store"
      [Button] BID=400 "Learn More"
      [Link] BID=500 "View Products"
    [Footer]
      [Link] BID=600 "Contact Us"
"""
    return mock_axtree


def test_proposer_curriculum_selection():
    """Test Proposer's curriculum selection capability using loaded Graph."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Test: Curriculum Selection")
    logger.info(f"{'='*60}")
    
    graph_dir = os.getenv("GRAPH_DIR", "explored_graphs/3_148_75_200_7770__20260103_130754")
    
    if not os.path.exists(graph_dir):
        logger.warning(f"Graph directory not found: {graph_dir}")
        logger.info("Skipping test - please set GRAPH_DIR environment variable")
        return
    
    graph = load_graph_from_directory(graph_dir)
    if not graph:
        logger.error("Failed to load graph, aborting test")
        return
    
    llm_engine = create_remote_llm_engine()
    proposer = TaskProposer(llm_engine=llm_engine, diversity_weight=0.1)
    
    if not graph.nodes:
        logger.warning("Graph has no nodes, cannot test curriculum selection")
        return
    
    test_node_id = list(graph.nodes.keys())[0]
    test_node = graph.nodes[test_node_id]
    
    logger.info(f"\nTesting curriculum selection from node: {test_node_id}")
    logger.info(f"Node has {len(test_node.children)} children")
    
    if test_node.children:
        target_result = proposer.select_target(test_node, graph, horizon_k=1)
        
        if target_result:
            target_node_id, edge_data = target_result
            logger.info(f"\n[SUCCESS] Selected target: {test_node_id} -> {target_node_id}")
            logger.info(f"  Edge data: {edge_data}")
        else:
            logger.warning(f"[FAILED] Could not select target from node {test_node_id}")
    else:
        logger.info(f"Node {test_node_id} has no children, skipping selection")
    
    logger.info(f"{'='*60}\n")


def test_proposer_instruction_generation():
    """Test Proposer's instruction generation capability."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Test: Instruction Generation")
    logger.info(f"{'='*60}")
    
    graph_dir = os.getenv("GRAPH_DIR", "explored_graphs/3_148_75_200_7770__20260103_130754")
    
    if not os.path.exists(graph_dir):
        logger.warning(f"Graph directory not found: {graph_dir}")
        logger.info("Skipping test - please set GRAPH_DIR environment variable")
        return
    
    graph = load_graph_from_directory(graph_dir)
    if not graph:
        logger.error("Failed to load graph, aborting test")
        return
    
    llm_engine = create_remote_llm_engine()
    proposer = TaskProposer(llm_engine=llm_engine, diversity_weight=0.1)
    
    if not graph.nodes:
        logger.warning("Graph has no nodes, cannot test instruction generation")
        return
    
    test_node_id = list(graph.nodes.keys())[0]
    test_node = graph.nodes[test_node_id]
    
    mock_axtree = get_mock_axtree(test_node)
    target_element = "click('256')"
    
    logger.info(f"\nGenerating instruction for target element: {target_element}")
    logger.info(f"Mock AxTree length: {len(mock_axtree)} chars")
    
    try:
        instruction = proposer.generate_instruction(
            obs_axtree=mock_axtree,
            target_element=target_element,
            target_node_id=test_node_id
        )
        
        logger.info(f"\n[SUCCESS] Generated Instruction:")
        logger.info(f"  {instruction}")
        
        assert isinstance(instruction, str)
        assert len(instruction) > 0
        logger.info(f"[PASSED] Instruction is valid")
        
    except Exception as e:
        logger.error(f"[FAILED] Instruction generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info(f"{'='*60}\n")


def test_proposer_full_pipeline():
    """Test Proposer's full pipeline: selection + generation."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Test: Full Pipeline (Selection + Generation)")
    logger.info(f"{'='*60}")
    
    graph_dir = os.getenv("GRAPH_DIR", "explored_graphs/3_148_75_200_7770__20260103_130754")
    
    if not os.path.exists(graph_dir):
        logger.warning(f"Graph directory not found: {graph_dir}")
        logger.info("Skipping test - please set GRAPH_DIR environment variable")
        return
    
    graph = load_graph_from_directory(graph_dir)
    if not graph:
        logger.error("Failed to load graph, aborting test")
        return
    
    llm_engine = create_remote_llm_engine()
    proposer = TaskProposer(llm_engine=llm_engine, diversity_weight=0.1)
    
    if not graph.nodes:
        logger.warning("Graph has no nodes, cannot test full pipeline")
        return
    
    test_node_id = list(graph.nodes.keys())[0]
    test_node = graph.nodes[test_node_id]
    
    if not test_node.children:
        logger.warning(f"Node {test_node_id} has no children, cannot test propose_task")
        return
    
    mock_axtree = get_mock_axtree(test_node)
    
    logger.info(f"\nTesting propose_task from node: {test_node_id}")
    
    try:
        result = proposer.propose_task(
            node=test_node,
            graph=graph,
            obs_axtree=mock_axtree,
            horizon_k=1,
            target_guidance=None
        )
        
        if result:
            instruction, target_node_id, verification_data = result
            
            logger.info(f"\n[SUCCESS] Task Proposal Complete:")
            logger.info(f"  Instruction: {instruction}")
            logger.info(f"  Target: {test_node_id} -> {target_node_id}")
            logger.info(f"  Verification Data: {verification_data}")
            
            assert isinstance(instruction, str)
            assert len(instruction) > 0
            assert isinstance(target_node_id, str)
            assert isinstance(verification_data, dict)
            
            logger.info(f"[PASSED] All assertions passed")
        else:
            logger.warning(f"[FAILED] propose_task returned None")
            
    except Exception as e:
        logger.error(f"[FAILED] Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info(f"{'='*60}\n")


def test_proposer_reward_calculation():
    """Test Proposer's reward calculation."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Test: Reward Calculation")
    logger.info(f"{'='*60}")
    
    graph_dir = os.getenv("GRAPH_DIR", "explored_graphs/3_148_75_200_7770__20260103_130754")
    
    if not os.path.exists(graph_dir):
        logger.warning(f"Graph directory not found: {graph_dir}")
        logger.info("Skipping test - please set GRAPH_DIR environment variable")
        return
    
    graph = load_graph_from_directory(graph_dir)
    if not graph:
        logger.error("Failed to load graph, aborting test")
        return
    
    llm_engine = create_remote_llm_engine()
    proposer = TaskProposer(llm_engine=llm_engine, diversity_weight=0.1)
    
    if not graph.edges:
        logger.warning("Graph has no edges, cannot test reward calculation")
        return
    
    edge_key = list(graph.edges.keys())[0]
    edge_data = graph.edges[edge_key]
    
    logger.info(f"\nTesting reward calculation for edge: {edge_key}")
    logger.info(f"Edge data: {edge_data}")
    
    try:
        reward = proposer.calculate_reward(
            edge_data=edge_data,
            is_valid=True,
            solver_success=None,
            alpha=1.0,
            beta=1.0,
            gamma=0.5
        )
        
        logger.info(f"\n[SUCCESS] Reward Calculated:")
        logger.info(f"  R_total = {reward:.4f}")
        
        assert isinstance(reward, (int, float))
        logger.info(f"[PASSED] Reward is valid")
        
        reward_invalid = proposer.calculate_reward(
            edge_data=edge_data,
            is_valid=False,
            solver_success=None
        )
        
        logger.info(f"\nTesting invalid task penalty:")
        logger.info(f"  R_total (invalid) = {reward_invalid:.4f}")
        assert reward_invalid < 0
        logger.info(f"[PASSED] Invalid task is penalized")
        
    except Exception as e:
        logger.error(f"[FAILED] Reward calculation failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info(f"{'='*60}\n")


def test_proposer_multi_hop_selection():
    """Test Proposer's multi-hop selection (K>1)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Test: Multi-Hop Selection (K=3)")
    logger.info(f"{'='*60}")
    
    graph_dir = os.getenv("GRAPH_DIR", "explored_graphs/3_148_75_200_7770__20260103_130754")
    
    if not os.path.exists(graph_dir):
        logger.warning(f"Graph directory not found: {graph_dir}")
        logger.info("Skipping test - please set GRAPH_DIR environment variable")
        return
    
    graph = load_graph_from_directory(graph_dir)
    if not graph:
        logger.error("Failed to load graph, aborting test")
        return
    
    llm_engine = create_remote_llm_engine()
    proposer = TaskProposer(llm_engine=llm_engine, diversity_weight=0.1)
    
    if not graph.nodes:
        logger.warning("Graph has no nodes, cannot test multi-hop selection")
        return
    
    test_node_id = list(graph.nodes.keys())[0]
    test_node = graph.nodes[test_node_id]
    
    if not test_node.children:
        logger.warning(f"Node {test_node_id} has no children, cannot test multi-hop selection")
        return
    
    logger.info(f"\nTesting multi-hop selection from node: {test_node_id}")
    
    try:
        target_result = proposer.select_target(test_node, graph, horizon_k=3)
        
        if target_result:
            target_node_id, edge_data = target_result
            logger.info(f"\n[SUCCESS] Selected multi-hop target: {test_node_id} -> ... -> {target_node_id}")
            logger.info(f"  Final edge data: {edge_data}")
            logger.info(f"[PASSED] Multi-hop selection succeeded")
        else:
            logger.warning(f"[FAILED] Could not find 3-hop path from node {test_node_id}")
            
    except Exception as e:
        logger.error(f"[FAILED] Multi-hop selection failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info(f"{'='*60}\n")


def test_proposer_with_all_graphs():
    """Test Proposer on all available graphs."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Test: Batch Test on All Graphs")
    logger.info(f"{'='*60}")
    
    base_dir = "explored_graphs"
    
    if not os.path.exists(base_dir):
        logger.warning(f"Base directory not found: {base_dir}")
        logger.info("Skipping batch test")
        return
    
    graph_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not graph_dirs:
        logger.warning(f"No graph directories found in {base_dir}")
        return
    
    logger.info(f"\nFound {len(graph_dirs)} graphs to test")
    
    llm_engine = create_remote_llm_engine()
    proposer = TaskProposer(llm_engine=llm_engine, diversity_weight=0.1)
    
    results = []
    
    for i, graph_dirname in enumerate(graph_dirs):
        graph_path = os.path.join(base_dir, graph_dirname)
        logger.info(f"\n--- Testing Graph {i+1}/{len(graph_dirs)}: {graph_dirname} ---")
        
        graph = load_graph_from_directory(graph_path)
        if not graph:
            logger.warning(f"Failed to load graph: {graph_dirname}")
            results.append({"graph": graph_dirname, "status": "load_failed"})
            continue
        
        if not graph.nodes:
            logger.warning(f"Graph has no nodes: {graph_dirname}")
            results.append({"graph": graph_dirname, "status": "no_nodes"})
            continue
        
        test_node_id = list(graph.nodes.keys())[0]
        test_node = graph.nodes[test_node_id]
        
        if not test_node.children:
            logger.warning(f"Node has no children: {graph_dirname}")
            results.append({"graph": graph_dirname, "status": "no_children"})
            continue
        
        try:
            mock_axtree = get_mock_axtree(test_node)
            result = proposer.propose_task(
                node=test_node,
                graph=graph,
                obs_axtree=mock_axtree,
                horizon_k=1
            )
            
            if result:
                instruction, target_node_id, verification_data = result
                logger.info(f"[SUCCESS] Generated instruction: {instruction[:80]}...")
                results.append({
                    "graph": graph_dirname,
                    "status": "success",
                    "instruction": instruction
                })
            else:
                logger.warning(f"[FAILED] propose_task returned None")
                results.append({"graph": graph_dirname, "status": "no_result"})
                
        except Exception as e:
            logger.error(f"[ERROR] {e}")
            results.append({"graph": graph_dirname, "status": "error", "error": str(e)})
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Batch Test Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total graphs: {len(graph_dirs)}")
    
    status_counts = {}
    for r in results:
        status = r["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count}")
    
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TaskProposer online with remote LLM")
    parser.add_argument("--test", type=str, default="all", 
                       choices=["selection", "generation", "pipeline", "reward", "multihop", "batch", "all"],
                       help="Which test to run")
    parser.add_argument("--graph-dir", type=str, default="explored_graphs/3_148_75_200_7770__20260103_130754",
                       help="Path to graph directory")
    
    args = parser.parse_args()
    
    os.environ["GRAPH_DIR"] = args.graph_dir
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TaskProposer Online Tests")
    logger.info(f"{'='*60}")
    logger.info(f"Graph directory: {args.graph_dir}")
    logger.info(f"LLM API: {os.getenv('LLM_API_BASE', 'http://127.0.0.1:6006')}")
    logger.info(f"{'='*60}\n")
    
    test_functions = {
        "selection": test_proposer_curriculum_selection,
        "generation": test_proposer_instruction_generation,
        "pipeline": test_proposer_full_pipeline,
        "reward": test_proposer_reward_calculation,
        "multihop": test_proposer_multi_hop_selection,
        "batch": test_proposer_with_all_graphs
    }
    
    if args.test == "all":
        for name, func in test_functions.items():
            try:
                func()
            except Exception as e:
                logger.error(f"Test {name} failed with exception: {e}")
                import traceback
                traceback.print_exc()
    else:
        test_functions[args.test]()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"All Tests Complete")
    logger.info(f"{'='*60}\n")

