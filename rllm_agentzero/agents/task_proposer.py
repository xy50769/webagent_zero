import logging
import random
import numpy as np
from typing import Optional, Tuple, Dict, List

from .server.llm_engine import LLMEngine
from .prompt_builders.proposer_prompt_builder import ProposerPromptBuilder
from ..core.node import Node
from ..core.graph import Graph

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class TaskProposer:
    """
    Task Proposer for curriculum learning.
    Selects optimal tasks based on Zone of Proximal Development (ZPD) principle.
    Generates natural language instructions for selected targets.
    """
    def __init__(self, llm_engine: LLMEngine, diversity_weight: float = 0.1):
        """Initialize TaskProposer with LLM engine and diversity weight."""
        self.llm = llm_engine
        self.diversity_weight = diversity_weight
        self.prompt_builder = ProposerPromptBuilder()

    def _calculate_curriculum_score(self, edge_data: Dict) -> float:
        """
        Calculate curriculum score for an edge (lower is better).
        Formula: |P_succ - 0.5| - λ * (1 / sqrt(N + 1))
        """
        success = edge_data.get("success", 0)
        total = edge_data.get("total", 0)
        
        if total == 0:
            p_succ = 0.5
        else:
            p_succ = success / total
        
        difficulty_score = abs(p_succ - 0.5)
        n_visits = total
        diversity_score = 1.0 / np.sqrt(n_visits + 1)
        final_score = difficulty_score - (self.diversity_weight * diversity_score)
        
        return final_score

    def select_target(self, node: Node, graph: Graph, horizon_k: int = 1) -> Optional[Tuple[str, Dict]]:
        """
        Select optimal target edge based on curriculum score.
        For K=1: selects best edge from current node.
        For K>1: performs random walk to find K-hop target.
        """
        if horizon_k == 1:
            if not node.children:
                logger.info(f"[Proposer] No children from node {node.node_id}. Cannot propose curriculum task.")
                return None
            
            best_target = None
            best_score = float('inf')
            
            for child_id in node.children:
                edge_key = f"{node.node_id}|{child_id}"
                edge_data = graph.edges.get(edge_key, {"success": 0, "total": 0, "target_element": "unknown"})
                score = self._calculate_curriculum_score(edge_data)
                
                if score < best_score:
                    best_score = score
                    best_target = (child_id, edge_data)
            
            if best_target is None:
                logger.info(f"[Proposer] No valid target found for node {node.node_id}.")
                return None
            
            target_node_id, edge_data = best_target
            success = edge_data.get("success", 0)
            total = edge_data.get("total", 0)
            p_succ = success / total if total > 0 else 0.5
            
            curriculum_explanation = self.prompt_builder.construct_curriculum_explanation(
                edge_data=edge_data,
                horizon_k=1
            )
            logger.info(f"[Proposer] {curriculum_explanation}")
            
            return best_target
        else:
            current_node_id = node.node_id
            path = []
            
            for step in range(horizon_k):
                current_node = graph.nodes.get(current_node_id)
                if not current_node or not current_node.children:
                    logger.warning(f"[Proposer] Cannot continue path at step {step}, node {current_node_id} has no children.")
                    break
                
                next_node_id = random.choice(current_node.children)
                edge_key = f"{current_node_id}|{next_node_id}"
                edge_data = graph.edges.get(edge_key, {"success": 0, "total": 0, "target_element": "unknown"})
                
                path.append((next_node_id, edge_data))
                current_node_id = next_node_id
            
            if not path:
                logger.info(f"[Proposer] Cannot find {horizon_k}-hop path from node {node.node_id}.")
                return None
            
            target_node_id, final_edge_data = path[-1]
            
            curriculum_explanation = self.prompt_builder.construct_curriculum_explanation(
                edge_data=final_edge_data,
                horizon_k=horizon_k
            )
            logger.info(f"[Proposer] {curriculum_explanation}")
            logger.info(f"Multi-hop path: {node.node_id} -> ... -> {target_node_id} ({len(path)} steps)")
            
            return (target_node_id, final_edge_data)

    def generate_instruction(self, obs_axtree: str, target_element: str, target_node_id: str = None) -> str:
        """Generate natural language instruction for the target element using LLM."""
        system_msg = self.prompt_builder.system_message()['text']
        user_msg = self.prompt_builder.construct_generation_prompt(
            obs_axtree=obs_axtree,
            target_element=target_element,
            target_node_id=target_node_id
        )
        
        try:
            raw_output = self.llm.generate(
                system_msg=system_msg,
                user_msg=user_msg,
                mode="proposer",
                temperature=0.7
            )
            
            instruction = self.prompt_builder.parse_instruction(raw_output)
            logger.info(f"[Proposer] Generated Instruction: {instruction}")
            
        except Exception as e:
            logger.error(f"[Proposer] Failed to generate instruction: {e}")
            instruction = f"Please interact with the element: {target_element}"
        
        return instruction

    def propose_task(
        self, 
        node: Node, 
        graph: Graph, 
        obs_axtree: str, 
        horizon_k: int = 1, 
        target_guidance: Optional[Tuple[str, Dict]] = None
    ) -> Optional[Tuple[str, str, Dict]]:
        """
        Main entry point: generate feasible but difficult task for Solver.
        Returns: (instruction, target_node_id, verification_data) or None
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"[Proposer] Starting Task Proposal")
        logger.info(f"{'='*60}")
        logger.info(f"Current Node: {node.node_id}")
        logger.info(f"Horizon K: {horizon_k}")
        
        if target_guidance:
            target_node_id, edge_data = target_guidance
            logger.info(f"[Proposer] Using Guidance Target: {node.node_id} -> {target_node_id}")
            logger.info(f"Target Element: {edge_data.get('target_element', 'unknown')}")
        else:
            result = self.select_target(node, graph, horizon_k)
            
            if result is None:
                logger.info(f"[Proposer] Cannot select target from node {node.node_id}. Switching to Exploration mode.")
                return None
            
            target_node_id, edge_data = result

        target_element = edge_data.get("target_element", "unknown element")
        instruction = self.generate_instruction(obs_axtree, target_element, target_node_id)
        
        logger.info(f"[Proposer] Task Proposal Complete")
        logger.info(f"Instruction: {instruction}")
        logger.info(f"Target: {node.node_id} -> {target_node_id}")
        logger.info(f"{'='*60}\n")
        
        verification_data = {
            "target_node_id": target_node_id,
            "target_element": target_element,
            "source_node_id": node.node_id,
            "horizon": horizon_k,
            "edge_stats": edge_data
        }
        
        return instruction, target_node_id, verification_data

    def calculate_reward(
        self, 
        edge_data: Dict, 
        is_valid: bool, 
        solver_success: bool = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5
    ) -> float:
        """
        Calculate Proposer reward for RL training.
        Formula: R = α·R_curriculum + β·R_feasibility + γ·R_diversity
        """
        C = 1.0

        if not is_valid:
            logger.warning(f"[Proposer Reward] Task is invalid (hallucination). Penalty: -{C * beta:.2f}")
            return -C * beta

        success = edge_data.get("success", 0)
        total = edge_data.get("total", 0)
        
        if total == 0:
            p_succ = 0.5
        else:
            p_succ = success / total
        
        r_curriculum = 1.0 - 2.0 * abs(p_succ - 0.5)
        n_visits = total
        r_diversity = 1.0 / np.sqrt(n_visits + 1)

        total_reward = (alpha * r_curriculum) + (gamma * r_diversity)
        
        logger.info(f"[Proposer Reward] R_total = {total_reward:.3f}")
        logger.info(f"R_curriculum = {r_curriculum:.3f} (P_succ = {p_succ:.2f})")
        logger.info(f"R_diversity = {r_diversity:.3f} (N = {n_visits})")
        
        return total_reward