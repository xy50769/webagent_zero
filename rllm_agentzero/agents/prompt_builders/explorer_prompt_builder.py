from .solver_prompt_builder import SolverPromptBuilder
from browsergym.core.action.base import AbstractActionSet

class RLLMExplorerPromptBuilder(SolverPromptBuilder):
    """
    RLLM Explorer Prompt Builder.
    Inherits from SolverPromptBuilder to reuse its ability to build context.
    """
    def __init__(self, action_set: AbstractActionSet):
        super().__init__(action_set)

    def construct_explorer_prompt(self, 
                                  goal: str, 
                                  obs: dict, 
                                  history: list, 
                                  visited_actions: list = None) -> str:
        """
        Build prompt string for local LLM.
        """
        from ..trajectory_data import BrowserGymAgentStepData
        
        current_step = BrowserGymAgentStepData(
            action=None,
            thought=None,
            axtree=obs.get("axtree_txt", ""),
            last_action_error=obs.get("last_action_error", ""),
            misc={}
        )

        messages_dict = self.build_messages(
            goal=goal,
            current_step=current_step,
            history=history
        )
        
        visited_hint = ""
        if visited_actions:
            actions_str = "\n".join(f"- {act}" for act in visited_actions[-5:]) # 只取最近5个避免过长
            visited_hint = f"\n\nIMPORTANT: You have already tried the following actions on this page. DO NOT repeat them:\n{actions_str}\nPlease try something different to find new states."


        user_content = messages_dict['prompt'][1]['content']
        

        if visited_hint:
            return user_content + visited_hint
        else:
            return user_content