import logging
import re
import json
from typing import Tuple, Dict, Optional

from .base_agent import AgentFactory, BaseAgent
from .prompt_builders.solver_prompt_builder import SolverPromptBuilder
from .trajectory_data import BrowserGymAgentStepData
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str
from openai import OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def extract_action_and_thought(raw_string):
    """提取 Thought 和 Action (保持原样)"""
    thought = ""
    action = ""
    
    try:
        # 1. JSON 格式优先
        json_match = re.search(r'\{.*\}', raw_string, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                thought = data.get("thought", "")
                action = data.get("action", "")
                return thought, action
            except json.JSONDecodeError:
                pass
        
        # 2. 文本格式兜底
        t_match = re.search(r'Thought:\s*(.*?)(?=Action:|$)', raw_string, re.DOTALL | re.IGNORECASE)
        if t_match:
            thought = t_match.group(1).replace('\\"', '"').strip()

        a_match = re.search(r'Action:\s*(.*)', raw_string, re.DOTALL | re.IGNORECASE)
        if a_match:
            action = a_match.group(1).replace('\\"', '"').strip()
        elif not action and re.search(r'(click|type|scroll|goto|go_back)\(', raw_string):
             # 容错：如果直接输出代码没有 Action: 前缀
             action = raw_string.strip()

    except Exception as e:
        logger.warning(f"Error parsing string: {e}")
        
    return thought, action

@AgentFactory.register
class SolverAgent(BaseAgent):
    """
    [RLLM Solver Agent]
    集成 Consistency Reward 和 Efficiency Penalty 的执行器。
    """

    def __init__(
            self,
            model_id: str = "rllm-model",
            base_url: str = "http://127.0.0.1:6006/v1",
            api_key: str = "EMPTY",
            temperature: float = 0.01,
            char_limit: int = 16000,
            demo_mode: str = 'off',
            
            # [新增] 奖励超参数
            consistency_weight: float = 1.0,
            efficiency_penalty: float = 0.05,
            **kwargs
    ):
        super().__init__(model_id=model_id, temperature=temperature, char_limit=char_limit, demo_mode=demo_mode, **kwargs)
        
        self.model_id = model_id
        self.temperature = temperature
        self.char_limit = char_limit
        
        # RL 参数
        self.consistency_weight = consistency_weight
        self.efficiency_penalty = efficiency_penalty

        self.client = OpenAI(base_url=base_url, api_key=api_key)

        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "infeas", "nav"],
            strict=False,
            multiaction=False,
            demo_mode=demo_mode
        )

        self.prompt_builder = SolverPromptBuilder(self.action_set)
        self.history: list[BrowserGymAgentStepData] = []

    def reset(self):
        self.history.clear()

    def obs_preprocessor(self, obs: dict) -> dict:
        """
        预处理观测，同时提取用于奖励计算的 Target 信息
        """
        # 注意：外层循环(Main Loop)需要负责将 target_info 注入到 obs['misc'] 或 obs 本身中
        target_info = obs.get("target_info", {}) 
        
        return {
            # 基础信息
            "goal_object": obs.get("goal_object", [{"text": "Follow instructions."}]),
            "axtree_txt": flatten_axtree_to_str(
                obs["axtree_object"], 
                filter_visible_only=False, 
                extra_properties=obs.get("extra_element_properties", {})
            ),
            "last_action_error": obs.get("last_action_error", ""),
            "open_pages_urls": obs.get("open_pages_urls", []),
            
            # [新增] 关键信息传递
            "target_info": target_info,  # Proposer 指定的目标 (用于 Consistency Reward)
            "step_count": obs.get("step_count", len(self.history)), # 当前步数 (用于 Efficiency Penalty)
            "extra_element_properties": obs.get("extra_element_properties", {}),
        }
    
    def action_processor(self, action: str) -> str:
        return self.action_set.to_python_code(action)

    def _extract_target_id_from_action(self, action_str: str) -> Optional[str]:
        """
        辅助函数：从动作字符串中提取操作对象的 Element ID。
        例如: "click('45')" -> "45"
        """
        if not action_str:
            return None
        # 匹配 click('123') 或 type('123', ...) 中的数字 ID
        match = re.search(r"['\"](\d+)['\"]", action_str)
        if match:
            return match.group(1)
        return None

    def _verify_target_consistency(self, action_id: str, target_desc: str, axtree_txt: str) -> float:
        """
        增强版 Consistency Check
        检查 action_id 对应的 AxTree 节点文本是否包含 target_desc 的关键词
        """
        if not action_id or not target_desc:
            return 0.0
            
        # 1. 在 AxTree 文本中定位 ID
        # AxTree 格式通常是: [ID] Role "Name"
        pattern = re.compile(fr"\[{re.escape(action_id)}\]\s*(.+)")
        match = pattern.search(axtree_txt)
        
        if not match:
            return 0.0  # 没找到 ID，可能是幻觉
            
        element_line = match.group(1).lower()
        target_keywords = [w.lower() for w in target_desc.split() if len(w) > 2]  # 过滤短词
        
        if not target_keywords:
            return 0.0
        
        # 2. 关键词匹配
        matches = sum(1 for word in target_keywords if word in element_line)
        match_rate = matches / len(target_keywords) if target_keywords else 0
        
        if match_rate > 0.6:
            return 1.0  # Strong Match
        elif match_rate > 0.3:
            return 0.5  # Weak Match
            
        return 0.0

    def _calculate_inner_reward(self, raw_action: str, target_info: Dict, step_count: int, axtree_txt: str) -> Tuple[float, Dict]:
        """
        计算 Agent 内部的 Dense Reward (Consistency & Efficiency)
        注意：Outcome Reward (成败) 通常由环境在 Episode 结束时给出，这里只计算过程奖励。
        
        Returns:
            total_inner_reward, info_dict
        """
        # 1. Efficiency Penalty
        r_efficiency = -self.efficiency_penalty * step_count
        
        # 2. Consistency Reward
        r_consistency = 0.0
        target_element_desc = target_info.get("target_element", "")
        action_id = self._extract_target_id_from_action(raw_action)
        
        if target_element_desc and action_id:
            # 调用增强版验证逻辑
            match_score = self._verify_target_consistency(action_id, target_element_desc, axtree_txt)
            r_consistency = self.consistency_weight * match_score
        
        total = r_consistency + r_efficiency
        return total, {
            "r_consistency": r_consistency, 
            "r_efficiency": r_efficiency,
            "action_id": action_id,
            "target_match_score": r_consistency / (self.consistency_weight + 1e-6)
        }

    def get_action(self, obs: dict, oracle_action: tuple[str, str] = None, **kwargs) -> tuple[str, dict]:
        
        current_step = BrowserGymAgentStepData(
            action=None,
            thought=None,
            axtree=obs["axtree_txt"],
            last_action_error=obs.get("last_action_error"),
            misc={}
        )

        action = "" 
        thought = ""
        raw_output = ""

        if oracle_action is None:
            # === LLM Generation ===
            try:
                # 1. Prompt Construction
                messages = self.prompt_builder.build_messages(
                    goal=obs["goal_object"][0]["text"],
                    current_step=current_step,
                    history=self.history,
                    char_limit=self.char_limit
                )['prompt']

                # 2. Inference
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=1024
                )
                
                raw_output = response.choices[0].message.content
                thought, action = extract_action_and_thought(raw_output)
                
                if response.usage:
                    current_step.misc["model_usage"] = response.usage.model_dump()
            
            except Exception as e:
                logger.error(f"Inference Error: {e}")
                thought = f"Error: {e}"
        
        else:
            action, thought = oracle_action
            raw_output = json.dumps({"thought": thought, "action": action})

        logger.info(f"🤖 Solver: {action}")

        # === Grounding ===
        parsed_action = self.action_processor(action) if action else ""

        # === [核心] 计算奖励 ===
        inner_reward, reward_details = self._calculate_inner_reward(
            raw_action=action,
            target_info=obs.get("target_info", {}),
            step_count=obs.get("step_count", 0),
            axtree_txt=obs["axtree_txt"]
        )

        # Update Step Data
        current_step.action = action 
        current_step.thought = thought
        current_step.misc.update({
            "thought": thought, 
            "raw_action": action, 
            "parsed_action": parsed_action,
            "raw_output": raw_output,
            # [新增] 奖励信息
            "inner_reward": inner_reward,
            "reward_details": reward_details,
            "target_info_snapshot": obs.get("target_info", {})  # 记录当步的目标，方便调试
        })
        
        self.history.append(current_step)

        # 返回 parsed_action 给环境执行，extras 包含所有训练所需数据
        return parsed_action, current_step.misc