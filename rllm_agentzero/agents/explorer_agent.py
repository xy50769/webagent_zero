"""
rllm_agentzero Explorer Agent

Explorer agent for open-ended web exploration, adapted to rLLM's BaseAgent interface.
The Graph/Node logic is designed to work with ExplorationWorkflow.
"""
import copy
import logging
import json
import numpy as np
from typing import Any, Dict, Optional

from rllm.agents.agent import Action, Step
from rllm.rewards.reward_types import RewardOutput
from browsergym.core.action.highlevel import HighLevelActionSet

from .base_agent import AgentFactory
from .solver_agent import SolverAgent, extract_action_and_thought
from .prompt_builders.explorer_prompt_builder import ExplorerPromptBuilder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@AgentFactory.register
class ExplorerAgent(SolverAgent):
    """
    [RLLM Explorer Agent]
    
    Explorer Agent for open-ended web exploration.
    Adapted to rLLM's BaseAgent interface with update_from_env/update_from_model pattern.
    
    Features:
    1. Curiosity-Driven: Prioritize Frontier and Unvisited Edges
    2. Epistemic Uncertainty: Use Mask to filter known actions
    3. Intrinsic Reward: Calculate Novelty and Information Gain
    
    Note: Graph/Node state management is handled by ExplorationWorkflow,
    not by the agent itself. The agent receives exploration context via
    the info dict in update_from_env().
    """
    
    def __init__(
        self, 
        model_id: str = None,
        temperature: float = 0.1,
        max_repeats: int = 3,
        **kwargs
    ):
        """
        Initialize ExplorerAgent.
        
        Args:
            model_id: Model identifier for inference
            temperature: Sampling temperature (lower for more focused exploration)
            max_repeats: Maximum times to repeat same action on a node
        """
        super().__init__(model_id=model_id, temperature=temperature, **kwargs)
        
        self.max_repeats = max_repeats

        # Override action set to include tab actions for exploration
        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "infeas", "nav", "tab"],
            strict=False,
            multiaction=False,
            demo_mode=self.demo_mode,
        )
        
        # Use explorer-specific prompt builder
        self.prompt_builder = ExplorerPromptBuilder(self.action_set)
        
        # Explorer-specific goal
        self._goal = "Explore the website to maximize state coverage. Find new pages and interaction states. Focus on elements that haven't been visited yet."
        
        # Context from workflow (set via update_from_env info)
        self._frontier_info: Optional[Dict] = None
        self._unvisited_elements: list = []
        self._visited_elements: list = []
    
    def reset(self):
        """Reset the agent's state for a new episode."""
        super().reset()
        self._frontier_info = None
        self._unvisited_elements = []
        self._visited_elements = []
        self._goal = "Explore the website to maximize state coverage. Find new pages and interaction states. Focus on elements that haven't been visited yet."
    
    def _format_observation_as_messages(self, obs: Any) -> list[dict]:
        """
        Format observation into chat messages using ExplorerPromptBuilder.
        
        Includes exploration-specific context like frontier info and element masks.
        """
        messages = []
        
        if isinstance(obs, dict):
            # Build exploration prompt using the prompt builder
            prompt_messages = self.prompt_builder.construct_explorer_prompt_messages(
                goal=self._goal,
                obs=obs,
                history=[],  # History is managed by trajectory
                frontier_info=self._frontier_info,
                unvisited_elements=self._unvisited_elements,
                visited_elements=self._visited_elements,
            )
            
            # The prompt builder returns full messages, we use them directly
            # but only take user messages since system is set in reset()
            for msg in prompt_messages:
                if msg.get("role") == "user":
                    messages.append(msg)
                elif msg.get("role") == "system" and not self.messages:
                    # Add system message if not already present
                    self.messages.insert(0, msg)
        
        return messages
    
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Update agent state after environment step.
        
        Extracts exploration context from info dict (provided by ExplorationWorkflow).
        
        Args:
            observation: Environment observation
            reward: Reward from environment
            done: Whether episode is done
            info: Additional info including:
                - frontier_info: Dict with next frontier node info
                - unvisited_elements: List of unvisited interactive elements
                - visited_elements: List of visited element IDs
                - node: Current graph node (optional)
                - graph: Graph reference (optional)
        """
        # Extract exploration context from info
        self._frontier_info = info.get("frontier_info")
        self._unvisited_elements = info.get("unvisited_elements", [])
        self._visited_elements = info.get("visited_elements", [])
        
        # Also check if elements are in observation
        if not self._unvisited_elements and isinstance(observation, dict):
            self._unvisited_elements = observation.get("interactive_elements", [])
        
        logger.info(f"Element Exploration Stats: "
                    f"{len(self._unvisited_elements)} unvisited, "
                    f"{len(self._visited_elements)} visited")
        
        # Call parent implementation
        super().update_from_env(observation, reward, done, info, **kwargs)
    
    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Update agent state after model generates a response.
        
        Args:
            response: Raw response string from the model
            
        Returns:
            Action object containing the parsed action
        """
        # Parse response
        thought, action_str = extract_action_and_thought(response)
        
        if self.debug_mode != "off":
            self._log_response(response, thought, action_str)
        
        # Add assistant message to history
        assistant_message = {"role": "assistant", "content": response}
        self.messages.append(assistant_message)
        
        # Extract ModelOutput from kwargs if present (provided by rLLM rollout engine via workflow)
        model_output = kwargs.get("model_output")

        # Create trajectory step with exploration-specific info
        new_step = Step(
            chat_completions=copy.deepcopy(self.messages),
            observation=self.current_observation,
            thought=thought,
            action=action_str,
            model_response=response,
            model_output=model_output,  # Store full model output (needed for PPO)
            prompt_ids=model_output.prompt_ids if model_output else [],
            response_ids=model_output.completion_ids if model_output else [],
            logprobs=model_output.logprobs if model_output else [],
            info={
                "raw_action": action_str,
                "frontier_info": self._frontier_info,
                "unvisited_count": len(self._unvisited_elements),
                "visited_count": len(self._visited_elements),
            }
        )
        self._trajectory.steps.append(new_step)
        
        logger.info(f"Explorer: {action_str}")
        
        return Action(action=action_str)
    
    @staticmethod
    def calculate_exploration_reward(
        source_node: Any,
        target_node: Any, 
        action_str: str,
        graph: Any
    ) -> RewardOutput:
        """
        Calculate Explorer's Intrinsic Reward.
        
        Math: R_explore = R_novelty + R_info_gain
        
        This is a static method that can be called by ExplorationWorkflow
        after env.step() to compute exploration rewards.
        
        Args:
            source_node: S_t (node before action)
            target_node: S_{t+1} (node after action)
            action_str: a_t (action executed)
            graph: World Model
            
        Returns:
            RewardOutput with reward value and metadata
        """
        r_novelty = 0.0
        r_info_gain = 0.0
        
        if not source_node or not target_node or not graph:
            return RewardOutput(reward=0.0, metadata={"error": "missing_nodes"})
        
        # 1. Novelty Reward: I(S_{t+1} is new)
        edge_key = (source_node.node_id, target_node.node_id)
        
        # Check if target_node is newly discovered
        if hasattr(graph, 'unexplored_nodes') and target_node in graph.unexplored_nodes:
            r_novelty = 1.0
            logger.info(f"[Reward] Novelty Discovery! Node {target_node.node_id}")
        
        # Alternative: Check edge visit count
        if hasattr(graph, 'edges') and edge_key in graph.edges:
            visit_count = graph.edges[edge_key].get("total", 1)
            if visit_count == 1:
                r_novelty = 1.0
        elif hasattr(graph, 'edges'):
            # Edge doesn't exist yet
            r_novelty = 1.0
        
        # 2. Information Gain Reward: 1 / sqrt(N(S_t, a_t))
        visit_count = 1
        if hasattr(graph, 'edges') and edge_key in graph.edges:
            visit_count = graph.edges[edge_key].get("total", 1)
        
        r_info_gain = 1.0 / np.sqrt(visit_count)
        
        total_reward = r_novelty + r_info_gain
        
        logger.info(f"[Reward] Explore Total: {total_reward:.3f} "
                    f"(Novelty: {r_novelty}, InfoGain: {r_info_gain:.3f})")
        
        return RewardOutput(
            reward=total_reward,
            metadata={
                "r_novelty": r_novelty,
                "r_info_gain": r_info_gain,
                "edge_key": str(edge_key),
                "visit_count": visit_count,
            }
        )
    
    # === Backward compatibility method (deprecated) ===
    def get_action(self, obs: dict, oracle_action=None, node=None, graph=None, **kwargs) -> tuple[str, dict]:
        """
        Legacy method for backward compatibility with existing test scripts.
        
        DEPRECATED: Use update_from_env/update_from_model pattern with ExplorationWorkflow.
        """
        logger.warning("get_action() is deprecated. Use update_from_env/update_from_model with ExplorationWorkflow.")
        
        # Build exploration context from node/graph
        info = {}
        
        if graph and node:
            # Get frontier info
            if not getattr(node, 'interactive_elements', []) and hasattr(graph, 'get_next_node'):
                next_node = graph.get_next_node()
                if next_node:
                    info["frontier_info"] = {
                        "node_id": next_node.node_id,
                        "url": getattr(next_node, 'url', ''),
                    }
            
            # Get element masks
            if hasattr(node, 'interactive_elements'):
                info["unvisited_elements"] = node.interactive_elements or []
            if hasattr(node, 'interactive_elements_visited'):
                info["visited_elements"] = [
                    elem.get("bid", "") for elem in node.interactive_elements_visited
                ]
        
        # Simulate env update
        self.update_from_env(obs, reward=0.0, done=False, info=info)
        
        if oracle_action is not None:
            action, thought = oracle_action
            raw_output = json.dumps({"thought": thought, "action": action})
            result = self.update_from_model(raw_output)
            return result.action, {"thought": thought, "raw_action": action}
        
        # Return chat_completions for external LLM call
        return "", {"messages": self.chat_completions, "needs_model_response": True}
    
    def calculate_reward(self, source_node, target_node, action_str: str, graph) -> float:
        """
        Backward compatible reward calculation method.
        
        DEPRECATED: Use calculate_exploration_reward() static method.
        """
        result = self.calculate_exploration_reward(source_node, target_node, action_str, graph)
        return result.reward