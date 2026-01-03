import logging
import json
from .base_agent import AgentFactory
from .solver_agent import SolverAgent, extract_action_and_thought
from .prompt_builders.explorer_prompt_builder import RLLMExplorerPromptBuilder
from .server.llm_engine import LLMEngine
from browsergym.core.action.highlevel import HighLevelActionSet
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
    Explorer Agent for open-ended website exploration.
    Inherits from SolverAgent to reuse parsing logic.
    Uses Base Model to discover new nodes in the graph.
    """
    def __init__(self, llm_engine: LLMEngine, **kwargs):
        """Initialize ExplorerAgent with fixed exploration goal."""
        super().__init__(llm_engine=llm_engine, **kwargs)
        self.prompt_builder = RLLMExplorerPromptBuilder(self.action_set)
        self._goal = "Explore the website. Click on links, buttons, or interact with elements to discover new pages or state changes."
    
    def obs_preprocessor(self, obs: dict) -> dict:
        """Preprocess observation with fixed exploration goal."""
        from browsergym.utils.obs import flatten_axtree_to_str
        
        return {
            "axtree_txt": flatten_axtree_to_str(
                obs["axtree_object"], 
                filter_visible_only=False, 
                extra_properties=obs.get("extra_element_properties", {})
            ),
            "last_action_error": obs.get("last_action_error", ""),
            "url": obs.get("url", ""),
            "goal_object": [{"text": self._goal}]
        }


    def get_action(self, obs: dict, oracle_action=None, node=None, **kwargs) -> tuple[str, dict]:
        """Generate action for exploration with visited actions filtering."""
        visited_actions = []
        if node and hasattr(node, "action_history"):
            visited_actions = [act for act, count in node.action_history.items() if count > 0]

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
            user_msg = self.prompt_builder.construct_explorer_prompt(
                goal=self._goal,
                obs=obs,
                history=self.history,
                visited_actions=visited_actions
            )

            response_text = self.llm.generate(
                system_msg=self.prompt_builder.system_message()['text'],
                user_msg=user_msg,
                mode="base",
                temperature=1.0
            )
            
            thought, action = extract_action_and_thought(response_text)
            current_step.misc["model_usage"] = {"completion_tokens": len(response_text)//4}
        else:
            action, thought = oracle_action
            response_text = json.dumps({"thought": thought, "action": action})
            
        logger.info(f"Explorer Output: Thought: {thought} Action: {action}")

        parsed_action = self.action_processor(action) if action else ""

        current_step.action = action
        current_step.thought = thought
        current_step.misc.update({
            "thought": thought, 
            "raw_action": action,
            "parsed_action": parsed_action,
            "raw_output": response_text,
            "visited_actions": visited_actions
        })
        
        self.history.append(current_step)

        return parsed_action, {
            "raw_action": action,
            "parsed_action": parsed_action,
            "thought": thought,
            "raw_output": response_text
        }