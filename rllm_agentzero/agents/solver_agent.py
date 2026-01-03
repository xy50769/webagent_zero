import logging
import re
import json
from .base_agent import AgentFactory, BaseAgent
from .prompt_builders.solver_prompt_builder import SolverPromptBuilder
from .prompt_builders import flatten_messages # 引用我们在 __init__.py 里写的辅助函数
from .server.llm_engine import LLMEngine
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str, prune_html, flatten_dom_to_str
from .trajectory_data import BrowserGymAgentStepData

logger = logging.getLogger(__name__)

def extract_action_and_thought(raw_string):
    """
    从模型输出中提取 Thought 和 Action。
    兼容 JSON 格式和 Thought: ... Action: ... 格式。
    增强容错：自动修复未闭合的引号、括号等常见错误。
    返回: (thought, action) - 注意返回顺序
    """
    thought = None
    action = None
    
    try:
        # 1. 尝试完整 JSON 解析（最可靠）
        try:
            json_match = re.search(r'\{.*\}', raw_string, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                thought = data.get("thought", "")
                action = data.get("action", "")
                # 如果成功解析 JSON 且有 action，直接返回
                if action:
                    return thought, action
        except json.JSONDecodeError as e:
            # JSON 解析失败，尝试修复
            logger.debug(f"JSON parse error, attempting to fix: {e}")
            try:
                json_match = re.search(r'\{.*', raw_string, re.DOTALL)  # 允许不完整的 JSON
                if json_match:
                    json_str = json_match.group(0)
                    # 修复常见问题
                    json_str = _fix_json_string(json_str)
                    data = json.loads(json_str)
                    thought = data.get("thought", "")
                    action = data.get("action", "")
                    if action:
                        logger.info("Successfully fixed and parsed malformed JSON")
                        return thought, action
            except Exception:
                pass
        
        # 2. 尝试正则提取 JSON 字段（即使 JSON 不完整）
        # 查找 "thought": "..."
        thought_match = re.search(r'"thought"\s*:\s*"(.*?)"(?=\s*[,}])', raw_string, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1)
            # 处理转义字符
            thought = thought.replace('\\\\', '\\').replace('\\"', '"')
            
        # 查找 "action": "..." - 支持未闭合的引号和截断
        action_patterns = [
            r'"action"\s*:\s*"(.*?)"(?=\s*[,}])',  # 标准格式：完整闭合
            r'"action"\s*:\s*"([^"]*?)(?:"|,|})',  # 允许引号未闭合但有分隔符
            r'"action"\s*:\s*"([^"]+)',            # 允许完全截断（贪婪匹配到字符串末尾）
            r"'action'\s*:\s*'([^']*?)(?:'|,|})",  # 单引号版本
            r"'action'\s*:\s*'([^']+)",            # 单引号截断版本
        ]
        
        for pattern in action_patterns:
            action_match = re.search(pattern, raw_string, re.DOTALL)
            if action_match:
                action = action_match.group(1)
                # 处理转义字符
                action = action.replace('\\\\', '\\').replace('\\"', '"')
                # 清理尾部可能的不完整内容
                action = action.rstrip('\\').rstrip()
                
                # 如果 action 看起来是截断的（例如 click("b258\ ），尝试补全
                if action and not action.endswith(')'):
                    # 检查是否是函数调用格式
                    func_match = re.match(r'(\w+)\s*\(\s*["\']?([^"\')\]]*)["\']?', action)
                    if func_match:
                        func_name = func_match.group(1)
                        param = func_match.group(2)
                        # 如果参数看起来不完整，保留原样但记录警告
                        if param and len(param) > 0:
                            logger.warning(f"Detected potentially truncated action: {action}")
                            # 尝试补全（假设是字符串参数）
                            action = f"{func_name}('{param}')"
                            logger.info(f"Auto-completed action to: {action}")
                
                if action:  # 只有非空才 break
                    break

        # 3. 如果 JSON 提取失败，尝试文本格式
        if not action:
            # Thought: ...
            t_match = re.search(r'Thought:\s*(.*?)(?=Action:|$)', raw_string, re.DOTALL | re.IGNORECASE)
            if t_match:
                thought = t_match.group(1).strip()
            
            # Action: ...
            a_match = re.search(r'Action:\s*(.*?)(?=\n\n|$)', raw_string, re.DOTALL | re.IGNORECASE)
            if a_match:
                action = a_match.group(1).strip()
                # 移除可能的引号包裹
                action = action.strip('"').strip("'")

    except Exception as e:
        logger.warning(f"Error parsing string: {e}")
        return None, None
        
    return thought, action  # 返回顺序：(thought, action)


def _fix_json_string(json_str: str) -> str:
    """
    修复常见的 JSON 格式错误。
    - 未闭合的引号
    - 未闭合的括号
    - 尾部逗号
    """
    fixed = json_str
    
    # 1. 修复 action 字段的未闭合引号
    # 查找 "action": "..." 模式
    action_match = re.search(r'"action"\s*:\s*"([^"]*?)(?:"|,|}|$)', fixed, re.DOTALL)
    if action_match:
        action_start = action_match.start(1)
        action_end = action_match.end(1)
        # 检查是否有闭合引号
        if action_end < len(fixed) and fixed[action_end] not in ['"', ',', '}']:
            # 引号未闭合，在下一个 , 或 } 之前插入引号
            next_delimiter = min(
                (fixed.find(',', action_end) if fixed.find(',', action_end) != -1 else len(fixed)),
                (fixed.find('}', action_end) if fixed.find('}', action_end) != -1 else len(fixed))
            )
            fixed = fixed[:next_delimiter] + '"' + fixed[next_delimiter:]
    
    # 2. 修复未闭合的大括号
    open_braces = fixed.count('{')
    close_braces = fixed.count('}')
    if open_braces > close_braces:
        fixed += '}' * (open_braces - close_braces)
    
    # 3. 移除尾部的逗号（在 } 之前）
    fixed = re.sub(r',\s*}', '}', fixed)
    
    # 4. 移除尾部的无效字符
    fixed = fixed.rstrip().rstrip(',').rstrip()
    
    return fixed

@AgentFactory.register
class SolverAgent(BaseAgent):
    """
    [RLLM Solver Agent]
    角色: 学生 (执行者)
    职责: 接收 Proposer 的 Instruction，利用 LLM 执行具体操作。
    """

    def __init__(
            self,
            llm_engine: LLMEngine,
            temperature: float = 0.01, # Solver 需要精准，温度调低
            char_limit: int = 16000,   # 上下文限制
            demo_mode: str = 'off',
            action_timeout: int = 2000,  # Playwright 动作超时时间（毫秒），默认 2000ms
            **kwargs
    ):
        super().__init__(**kwargs)
        
        self.llm = llm_engine
        self.temperature = temperature
        self.char_limit = char_limit
        self.action_timeout = action_timeout
        
        # 1. 动作空间定义
        # 注意：BrowserGym 的 HighLevelActionSet 不直接支持 timeout 参数
        # timeout 需要在环境级别设置，或通过 demo_mode 的配置传递
        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "infeas", "nav"],
            strict=False,
            multiaction=False,
            demo_mode=demo_mode
        )

        # 2. 初始化原版 SolverPromptBuilder
        self.prompt_builder = SolverPromptBuilder(self.action_set)

        # 3. 历史记录
        self.history: list[BrowserGymAgentStepData] = []

    def reset(self):
        self.history = []

    def obs_preprocessor(self, obs: dict) -> dict:
        """
        数据预处理，提取 Prompt 需要的所有字段
        """
        # 提取 Instruction (Goal)
        # 注意：在 RLLM 架构中，Goal 通常由 Proposer 生成并传入 obs['goal_object']
        # 或者在 reset 时设定。这里假设环境标准格式。
        
        return {
            "goal_object": obs.get("goal_object", [{"text": "Follow instructions."}]),
            "axtree_txt": flatten_axtree_to_str(
                obs["axtree_object"], 
                filter_visible_only=False, 
                extra_properties=obs.get("extra_element_properties", {})
            ),
            "last_action_error": obs.get("last_action_error", ""),
            "open_pages_urls": obs.get("open_pages_urls", []),
            # 保留其他可能需要的字段
            "extra_element_properties": obs.get("extra_element_properties", {}),
        }
    
    def action_processor(self, action: str) -> str:
        """将语义动作字符串转换为可执行的 Python 代码"""
        # 直接转换，不再解析（action 已经是解析好的语义动作字符串）
        return self.action_set.to_python_code(action)

    
    def get_action(self, obs: dict, oracle_action: tuple[str, str] = None, **kwargs) -> tuple[str, dict]:
        """
        核心决策逻辑
        返回: (response_text, extras_dict)
        其中 extras_dict 包含:
            - raw_action: 语义动作字符串，如 click('12')
            - parsed_action: 完整的可执行 Python 代码
            - thought: 推理思维链
        """
        # 1. 构造当前步骤数据
        current_step = BrowserGymAgentStepData(
            action=None,
            thought=None,
            axtree=obs["axtree_txt"],
            last_action_error=obs.get("last_action_error"),
            misc={}
        )

        response_text = ""
        action = ""  # 语义动作字符串，如 click('12')
        thought = ""

        if oracle_action is None:
            # === LLM 推理分支 ===
            
            # 2. 调用 Builder 构建消息列表
            messages_dict = self.prompt_builder.build_messages(
                goal=obs["goal_object"][0]["text"],
                current_step=current_step,
                history=self.history,
                char_limit=self.char_limit
            )
            
            raw_messages = messages_dict['prompt']
            
            # 3. 使用 flatten_messages 压扁消息
            flat_msgs = flatten_messages(raw_messages)
            
            system_msg = ""
            user_msg = ""
            
            # 提取 System 和 User 内容
            for m in flat_msgs:
                if m['role'] == 'system':
                    system_msg = m['content']
                elif m['role'] == 'user':
                    user_msg = m['content']

            # 4. 发送给远程 LLM
            # Solver 使用 base model
            response_text = self.llm.generate(
                system_msg=system_msg,
                user_msg=user_msg,
                mode="base",
                temperature=self.temperature
            )
            
            # 5. 解析结果 - 返回顺序为 (thought, action)
            thought, action = extract_action_and_thought(response_text)
            
            # 记录 Token 使用量
            current_step.misc["model_usage"] = {"completion_tokens": len(response_text)//4}
        
        else:
            # === Oracle 分支 ===
            action, thought = oracle_action  # 假设 oracle_action 是 (action, thought)
            response_text = json.dumps({"thought": thought, "action": action})
            
        logger.info(f"🤖 Solver Output:\nThought: {thought}\nAction: {action}")

        # 6. 转换为可执行代码（仅用于 extras，供调试）
        parsed_action = self.action_processor(action) if action else ""

        # 7. 更新当前步骤数据
        # 注意：存储语义动作（raw_action）而不是完整代码（parsed_action）
        # 这样在构建下一轮 Prompt 时，history 中是简洁的 click('12') 而不是冗长的 Python 代码
        current_step.action = action  # 存储语义动作：click('12')
        current_step.thought = thought
        current_step.misc.update({
            "thought": thought, 
            "raw_action": action,              # 语义动作：click('12')
            "parsed_action": parsed_action,    # 完整 Python 代码（仅用于调试）
            "raw_output": response_text
        })
        
        # 8. 存入历史
        self.history.append(current_step)

        # 9. 返回 - 第一个返回值是语义动作（raw_action）
        # 环境会使用 action_mapping (agent.action_processor) 将其转换为可执行代码
        # 参考 episode.py line 38: browser_env.action_mapping = agent.action_processor
        return action, {
            "raw_action": action,           # 语义动作，用于记录 action_history
            "parsed_action": parsed_action, # 完整代码，用于调试
            "thought": thought,
            "raw_output": response_text     # LLM 原始输出，用于调试
        }