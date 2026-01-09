import logging
import numpy as np
from .base_agent import AgentFactory
from .solver_agent import SolverAgent, extract_action_and_thought
from .prompt_builders.explorer_prompt_builder import ExplorerPromptBuilder 
from .trajectory_data import BrowserGymAgentStepData

logger = logging.getLogger(__name__)

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
        max_repeats: int = 3,
        **kwargs
    ):
        super().__init__(model_id=model_id, **kwargs)
        
        self.prompt_builder = ExplorerPromptBuilder(self.action_set)
        self.max_repeats = max_repeats 
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
        if graph:
            frontier_node = graph.get_next_node()
            if frontier_node:
                frontier_info = {
                    "node_id": frontier_node.node_id,
                    "url": getattr(frontier_node, 'url', ''),
                }

        # B. Candidate Filtering (Mask M_t)
        # Math: A_candidate = {a | VisitCount(S_t, a) < tau}
        visited_actions = []
        if node and hasattr(node, "action_history"):
            # action_history 现在是 list[str]，需要统计每个动作出现的次数
            from collections import Counter
            action_counts = Counter(node.action_history)
            visited_actions = [
                act for act, count in action_counts.items() 
                if count >= self.max_repeats
            ]
        
        # C. Element-level Exploration Mask（元素级别的探索 Mask）
        unvisited_elements = []
        visited_element_bids = []
        exploration_stats = None
        
        # 优先从 node.interactive_elements 获取未访问的元素
        if node and hasattr(node, 'interactive_elements'):
            # 使用 node 中已经过滤掉访问过元素的列表（直接使用，不需要转换）
            unvisited_elements = node.interactive_elements
            
            # 获取已访问的元素 IDs
            visited_element_bids = [
                elem.get("bid", "") 
                for elem in (node.interactive_elements_visited if hasattr(node, 'interactive_elements_visited') else [])
            ]
            
            total_elements = len(unvisited_elements) + len(visited_element_bids)
            exploration_stats = {
                'total': total_elements,
                'visited': len(visited_element_bids),
                'unvisited': len(unvisited_elements),
                'coverage': len(visited_element_bids) / total_elements if total_elements > 0 else 0.0,
                'failed': 0
            }
            
            logger.info(f"Element Exploration Stats: "
                       f"{len(unvisited_elements)} unvisited, "
                       f"{len(visited_element_bids)} visited, "
                       f"coverage: {exploration_stats['coverage']:.1%}")
        elif "interactive_elements" in obs:
            # 如果 node 没有元素信息，回退到使用观察中的全部元素
            current_elements = obs["interactive_elements"]
            
            unvisited_elements = [
                {
                    "bid": elem.get("bid", ""),
                    "text": elem.get("text", ""),
                    "role": elem.get("role", "")
                }
                for elem in current_elements
                if elem.get('visible') and elem.get('clickable')
            ]
            
            visited_element_bids = []
            exploration_stats = {
                'total': len(unvisited_elements),
                'visited': 0,
                'unvisited': len(unvisited_elements),
                'coverage': 0.0,
                'failed': 0
            }
            
            logger.info(f"Element Exploration Stats (from obs): "
                       f"{len(unvisited_elements)} interactive elements available")

        current_step = BrowserGymAgentStepData(
            action=None, 
            thought=None, 
            axtree=obs["axtree_txt"], 
            last_action_error=obs.get("last_action_error"), 
            misc={}
        )

        response_text = ""
        action = ""
        thought = ""

        if oracle_action is None:
            # === 2. Semantic Selection via LLM ===
            # Math: a_t ~ Softmax(Score_novelty | O_t, A_candidate)
            
            messages = self.prompt_builder.construct_explorer_prompt_messages(
                goal=self._goal,
                obs=obs,
                history=self.history,
                visited_actions=visited_actions,
                frontier_info=frontier_info,
                unvisited_elements=unvisited_elements,  # 新增：未访问元素
                visited_element_bids=visited_element_bids,  # 新增：已访问元素 bid
                exploration_stats=exploration_stats  # 新增：探索统计
            )

            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=1.0, 
                    max_tokens=1024
                )
                response_text = response.choices[0].message.content
                
                if response.usage:
                    current_step.misc["model_usage"] = response.usage.model_dump()
            except Exception as e:
                logger.error(f"Explorer Inference Error: {e}")
                
            thought, action = extract_action_and_thought(response_text)
            
        else:
            action, thought = oracle_action
            response_text = f'{{"thought": "{thought}", "action": "{action}"}}'

        logger.info(f"Explorer: {action}")

        # === 3. Update History ===
        # BrowserGym env.step() will handle action parsing, no need to call action_processor
        current_step.action = action
        current_step.thought = thought
        current_step.misc.update({
            "thought": thought, 
            "raw_action": action,
            "visited_actions": visited_actions,
            "frontier_info": frontier_info,
            "response_text": response_text
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