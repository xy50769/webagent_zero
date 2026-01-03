from .solver_prompt_builder import SolverPromptBuilder
from browsergym.core.action.base import AbstractActionSet

class RLLMExplorerPromptBuilder(SolverPromptBuilder):
    """
    RLLM 专用的 Explorer Prompt Builder。
    继承自 SolverPromptBuilder，复用其构建 Context 的能力。
    """
    def __init__(self, action_set: AbstractActionSet):
        # 初始化父类
        super().__init__(action_set)

    def construct_explorer_prompt(self, 
                                  goal: str, 
                                  obs: dict, 
                                  history: list, 
                                  visited_actions: list = None) -> str:
        """
        构建适用于本地 LLM 的 Prompt 字符串。
        """
        # 1. 构造一个伪造的 current_step 对象以复用父类逻辑
        # BrowserGymAgentStepData 需要这些字段
        from ..trajectory_data import BrowserGymAgentStepData
        
        current_step = BrowserGymAgentStepData(
            action=None,
            thought=None,
            axtree=obs.get("axtree_txt", ""),
            last_action_error=obs.get("last_action_error", ""),
            misc={}
        )

        # 2. 调用父类方法获取标准 Message List (OpenAI 格式)
        messages_dict = self.build_messages(
            goal=goal,
            current_step=current_step,
            history=history
        )
        
        # messages_dict['prompt'] 结构是: [[system_msg], [user_msg_1, user_msg_2...]]
        
        # 3. [关键] 注入 Visited Actions (Candidate Filtering)
        # 这是一个 RLLM 特有的逻辑，原版 SolverPromptBuilder 没有
        visited_hint = ""
        if visited_actions:
            actions_str = "\n".join(f"- {act}" for act in visited_actions[-5:]) # 只取最近5个避免过长
            visited_hint = f"\n\nIMPORTANT: You have already tried the following actions on this page. DO NOT repeat them:\n{actions_str}\nPlease try something different to find new states."

        # 4. 将 Message List 压扁成单一 User String
        # 我们只取 User 部分，System 部分由 LLMEngine 统一管理
        # build_messages 返回的是 flatten 后的消息，content 是字符串
        user_content = messages_dict['prompt'][1]['content']
        
        # 加入过滤提示
        if visited_hint:
            return user_content + visited_hint
        else:
            return user_content