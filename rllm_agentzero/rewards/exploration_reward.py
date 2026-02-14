"""
Exploration Reward Function

rLLM-compatible reward function for web exploration tasks.
Implements the RewardFunction protocol from rllm.rewards.
"""
import logging
import numpy as np
from typing import Any, Dict

from rllm.rewards.reward_types import RewardOutput

logger = logging.getLogger(__name__)


def exploration_reward_fn(task_info: Dict[str, Any], action: str) -> RewardOutput:
    """
    Calculate exploration reward for web agent training.
    
    This reward function combines:
    1. Novelty Reward: Bonus for discovering new states
    2. Information Gain: Bonus inversely proportional to visit count
    3. Coverage Bonus: Reward for increasing graph coverage
    
    Args:
        task_info: Dictionary containing:
            - source_node: Node before action
            - target_node: Node after action  
            - graph: Graph world model
            - visit_count: Number of times this transition has occurred
            - is_new_node: Whether target_node was newly created
        action: The action string that was executed
        
    Returns:
        RewardOutput with reward value and metadata
    """
    # Extract graph context
    source_node = task_info.get("source_node")
    target_node = task_info.get("target_node")
    graph = task_info.get("graph")
    
    # If no graph context, return zero reward
    if not source_node or not target_node:
        return RewardOutput(
            reward=0.0,
            metadata={"error": "missing_graph_context"}
        )
    
    # 1. Novelty Reward
    r_novelty = 0.0
    is_new_node = task_info.get("is_new_node", False)
    
    if is_new_node:
        r_novelty = 1.0
        logger.debug(f"[Reward] Novelty: New node discovered")
    elif graph and hasattr(graph, 'unexplored_nodes'):
        if target_node in graph.unexplored_nodes:
            r_novelty = 1.0
            logger.debug(f"[Reward] Novelty: Unexplored node visited")
    
    # 2. Information Gain Reward
    r_info_gain = 0.0
    visit_count = task_info.get("visit_count", 1)
    
    if visit_count > 0:
        r_info_gain = 1.0 / np.sqrt(visit_count)
    else:
        r_info_gain = 1.0  # First visit
    
    # 3. Coverage Bonus (optional)
    r_coverage = 0.0
    if graph and hasattr(graph, 'nodes'):
        total_nodes = len(graph.nodes)
        explored_nodes = len(getattr(graph, 'explored_nodes', []))
        if total_nodes > 0:
            coverage_ratio = explored_nodes / total_nodes
            r_coverage = 0.1 * coverage_ratio  # Small bonus for coverage
    
    # Total reward
    total_reward = r_novelty + r_info_gain + r_coverage
    
    logger.debug(
        f"[Reward] Total: {total_reward:.3f} "
        f"(Novelty: {r_novelty}, InfoGain: {r_info_gain:.3f}, Coverage: {r_coverage:.3f})"
    )
    
    return RewardOutput(
        reward=total_reward,
        metadata={
            "r_novelty": r_novelty,
            "r_info_gain": r_info_gain,
            "r_coverage": r_coverage,
            "visit_count": visit_count,
            "is_new_node": is_new_node,
        }
    )


def task_completion_reward_fn(task_info: Dict[str, Any], action: str) -> RewardOutput:
    """
    Calculate task completion reward (sparse, outcome-based).
    
    This is typically provided by the environment but can be wrapped here
    for consistent interface.
    
    Args:
        task_info: Dictionary containing:
            - success: Whether the task was completed successfully
            - done: Whether the episode ended
        action: The final action
        
    Returns:
        RewardOutput with 1.0 for success, 0.0 otherwise
    """
    success = task_info.get("success", False)
    done = task_info.get("done", False)
    
    if done and success:
        reward = 1.0
    else:
        reward = 0.0
    
    return RewardOutput(
        reward=reward,
        metadata={
            "success": success,
            "done": done,
        }
    )
