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
    [RLLM Explorer]
    统一的探索 Agent，继承自 SolverAgent。
    合并了 Nav 和 Page Explorer 的功能。
    职责：利用 Base Model (Merged) 在图中发现新节点。
    
    与 SolverAgent 的区别：
    1. 使用 RLLMExplorerPromptBuilder 而不是 SolverPromptBuilder
    2. 使用固定的探索目标而不是任务指令
    3. 使用 mode="base" 调用 LLM
    4. 支持 visited_actions 过滤
    """
    def __init__(self, llm_engine: LLMEngine, **kwargs):
        # 调用父类（SolverAgent）的 __init__
        # 注意：SolverAgent.__init__ 已经初始化了 action_set, history 等
        super().__init__(llm_engine=llm_engine, **kwargs)
        
        # 覆盖 prompt_builder 为 Explorer 专用的 Builder
        self.prompt_builder = RLLMExplorerPromptBuilder(self.action_set)
        
        # 固定的探索目标（覆盖 SolverAgent 的任务目标）
        self._goal = "Explore the website. Click on links, buttons, or interact with elements to discover new pages or state changes."
    
    # 继承 SolverAgent 的方法：
    # - reset()
    # - action_processor()
    # 无需重新定义
    
    def obs_preprocessor(self, obs: dict) -> dict:
        """
        覆盖 SolverAgent 的 obs_preprocessor，强制使用 Explorer 的固定目标。
        确保 goal_object 始终使用 self._goal，防止外部干扰。
        """
        from browsergym.utils.obs import flatten_axtree_to_str
        
        return {
            "axtree_txt": flatten_axtree_to_str(
                obs["axtree_object"], 
                filter_visible_only=False, 
                extra_properties=obs.get("extra_element_properties", {})
            ),
            "last_action_error": obs.get("last_action_error", ""),
            "url": obs.get("url", ""),
            "goal_object": [{"text": self._goal}]  # 强制使用 Explorer 的固定目标
        }


    def get_action(self, obs: dict, oracle_action=None, node=None, **kwargs) -> tuple[str, dict]:
        """
        Explorer 核心决策
        覆盖 SolverAgent.get_action()，使用 Explorer 特定的逻辑
        """
        # 1. 获取已访问动作 (Candidate Filtering) - Explorer 特有
        visited_actions = []
        if node and hasattr(node, "action_history"):
            visited_actions = [act for act, count in node.action_history.items() if count > 0]

        # 2. 构建当前步骤数据（用于 axtree 传递）
        current_step = BrowserGymAgentStepData(
            action=None,
            thought=None,
            axtree=obs["axtree_txt"],
            last_action_error=obs.get("last_action_error"),
            misc={}
        )

        response_text = ""
        action = ""  # semantic action (语义动作)
        thought = ""

        if oracle_action is None:
            # === LLM 推理分支 ===
            
            # 3. 构建 Prompt String (使用 Explorer 的特殊构建方式)
            user_msg = self.prompt_builder.construct_explorer_prompt(
                goal=self._goal,
                obs=obs,
                history=self.history,
                visited_actions=visited_actions
            )

            # 4. 调用 LLM Engine (使用 mode="base")
            response_text = self.llm.generate(
                system_msg=self.prompt_builder.system_message()['text'],
                user_msg=user_msg,
                mode="base",  # Explorer 使用 base model
                temperature=1.0  # Explorer 使用更高的温度以增加探索性
            )
            
            # 5. 解析结果 - 复用 SolverAgent 的 extract_action_and_thought
            thought, action = extract_action_and_thought(response_text)
            
            # 记录 Token 使用量
            current_step.misc["model_usage"] = {"completion_tokens": len(response_text)//4}
        
        else:
            # === Oracle 分支 ===
            action, thought = oracle_action
            response_text = json.dumps({"thought": thought, "action": action})
            
        logger.info(f"🧭 Explorer Output:\nThought: {thought}\nAction: {action}")

        # 6. 转换为可执行代码
        parsed_action = self.action_processor(action) if action else ""

        # 7. 更新当前步骤数据
        # 注意：存储语义动作（action）而不是完整代码（parsed_action）
        # 这样在构建下一轮 Prompt 时，history 中是简洁的 click('12') 而不是冗长的 Python 代码
        current_step.action = action  # 存储语义动作：click('12')
        current_step.thought = thought
        current_step.misc.update({
            "thought": thought, 
            "raw_action": action,
            "parsed_action": parsed_action,
            "raw_output": response_text,
            "visited_actions": visited_actions  # Explorer 特有：记录已访问动作
        })
        
        # 8. 存入历史
        self.history.append(current_step)

        # 9. 返回 - 第一个返回值必须是可执行代码（parsed_action）
        # BrowserGym 环境会直接执行第一个返回值
        # 虽然环境设置了 action_mapping，但为了保持与 SolverAgent 一致，
        # 我们统一返回 parsed_action
        return parsed_action, {
            "raw_action": action,
            "parsed_action": parsed_action,
            "thought": thought,
            "raw_output": response_text
        }