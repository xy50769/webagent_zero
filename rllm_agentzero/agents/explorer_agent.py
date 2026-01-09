import logging
import json
import numpy as np
from collections import Counter
from browsergym.core.action.highlevel import HighLevelActionSet
from .base_agent import AgentFactory
from .solver_agent import SolverAgent, extract_action_and_thought
from .prompt_builders.explorer_prompt_builder import ExplorerPromptBuilder 
from .trajectory_data import BrowserGymAgentStepData

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
    [RLLM Explorer] 
    Explorer Agent for open-ended exploration
    
    Features:
    1. Curiosity-Driven: Prioritize Frontier and Unvisited Edges
    2. Epistemic Uncertainty: Use Mask to filter known actions
    3. Intrinsic Reward: Calculate Novelty and Information Gain
    """
    
    def __init__(
        self, 
        model_id: str, 
        temperature: float = 0.1,
        **kwargs
    ):
        super().__init__(model_id=model_id, temperature=temperature, **kwargs)

        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "infeas", "nav", "tab"],
            strict=False,
            multiaction=False,
            demo_mode=self.demo_mode,
        )
        self.prompt_builder = ExplorerPromptBuilder(self.action_set)
        self._goal = "Explore the website to maximize state coverage. Find new pages and interaction states. Focus on elements that haven't been visited yet."

    def get_action(self, obs: dict, oracle_action=None, node=None, graph=None, **kwargs) -> tuple[str, dict]:
        """
        Explorer decision logic
        
        Input: 
            - obs: Current Observation (O_t)
            - node: Current Abstract Node (S_t)
            - graph: World Model (for Frontier and Mask)
        """
        
        # === 1. Input Processing: Frontier Queue & Mask ===
        
        # A. Get Frontier info
        frontier_info = None
        if graph and node and not node.interactive_elements and graph.get_next_node():
            frontier_info = {
                "node_id":  graph.get_next_node().node_id,
                "url": getattr( graph.get_next_node(), 'url', ''),
            }
        
        # B. Element-level Exploration Mask
        unvisited_elements = []
        visited_elements = []
        if node:
            unvisited_elements = node.interactive_elements or []
            visited_elements = [
                elem.get("bid", "") 
                for elem in (node.interactive_elements_visited if hasattr(node, 'interactive_elements_visited') else [])
            ]
        else:
            unvisited_elements = obs.get("interactive_elements", [])
            visited_elements = []

        
        logger.info(f"Element Exploration Stats: "
                    f"{len(unvisited_elements)} unvisited, "
                    f"{len(visited_elements)} visited")

        current_step = BrowserGymAgentStepData(
            action=None, 
            thought=None, 
            axtree=obs["axtree_txt"], 
            last_action_error=obs.get("last_action_error"), 
            misc={}
        )

        raw_output = ""
        action = ""
        thought = ""

        if oracle_action is None:
            # === 2. Semantic Selection via LLM ===
            # Math: a_t ~ Softmax(Score_novelty | O_t, A_candidate)
            
            messages = self.prompt_builder.construct_explorer_prompt_messages(
                goal=self._goal,
                obs=obs,
                history=self.history,
                frontier_info=frontier_info,
                unvisited_elements=unvisited_elements,
                visited_elements=visited_elements,
            )
            
            self._record_prompt(messages, len(self.history) + 1)
            
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        temperature=self.temperature, 
                        max_tokens=2048
                    )
                    raw_output = response.choices[0].message.content
                    thought, action = extract_action_and_thought(raw_output)
                    
                    self._record_response(raw_output, thought, action, len(self.history) + 1)
                    if action and action.strip():
                        if response.usage:
                            current_step.misc["model_usage"] = response.usage.model_dump()
                        break
                    else:
                        logger.warning(f"Attempt {attempt + 1}: Failed to extract action. Retrying...")
                except Exception as e:
                    logger.error(f"Explorer Inference Error: {e}")
                    if attempt == self.max_retries - 1:
                        thought = f"Error after {self.max_retries} retries: {e}"
                        action = "report_infeasible('Model failed to generate a valid action after multiple retries.')"
            
        else:
            action, thought = oracle_action
            raw_output = json.dumps({"thought": thought, "action": action})

        logger.info(f"Explorer: {action}")

        # === 3. Update History ===
        # BrowserGym env.step() will handle action parsing, no need to call action_processor
        current_step.action = action
        current_step.thought = thought
        current_step.misc.update({
            "thought": thought, 
            "raw_action": action,
            "raw_output": raw_output,
            "frontier_info": frontier_info
        })
        self.history.append(current_step)

        return action, current_step.misc

    def calculate_reward(self, source_node, target_node, action_str: str, graph) -> float:
        """
        Calculate Explorer's Intrinsic Reward
        Math: R_explore = R_novelty + R_info_gain
        
        This method should be called by Main Loop after env.step()
        
        Args:
            source_node: S_t (node before action)
            target_node: S_{t+1} (node after action)
            action_str: a_t (action executed)
            graph: World Model
        """
        r_novelty = 0.0
        r_info_gain = 0.0
        
        if not source_node or not target_node or not graph:
            return 0.0
        
        # 1. Novelty Reward
        # Math: I(S_{t+1} is new)
        # Check if target_node is newly created (in unexplored_nodes)
        if target_node in graph.unexplored_nodes:
            r_novelty = 1.0
            logger.info(f"[Reward] Novelty Discovery! Node {target_node.node_id}")
        
        # Alternative: Check if this is the first time visiting this node
        # by checking if the edge is new (total == 1)
        edge_key = (source_node.node_id, target_node.node_id)
        if edge_key in graph.edges:
            visit_count = graph.edges[edge_key].get("total", 1)
            # If this is the first transition (visit_count == 1), it's novel
            if visit_count == 1:
                r_novelty = 1.0
        else:
            # Edge doesn't exist yet, will be created, so it's novel
            r_novelty = 1.0

        # 2. Information Gain Reward
        # Math: 1 / sqrt(N(S_t, a_t))
        # Get visit count for edge (S_t, S_{t+1})
        visit_count = 1
        if edge_key in graph.edges:
            visit_count = graph.edges[edge_key].get("total", 1)
        
        # Calculate Info Gain (fewer visits = higher reward)
        r_info_gain = 1.0 / np.sqrt(visit_count)
        
        total_reward = r_novelty + r_info_gain
        
        logger.info(f"[Reward] Explore Total: {total_reward:.3f} (Novelty: {r_novelty}, InfoGain: {r_info_gain:.3f})")
        return total_reward