"""
Proposer Prompt Builder
为 Task Proposer 构建条件生成提示词
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ProposerPromptBuilder:
    """
    [Proposer Prompt Builder]
    为 Task Proposer 构建条件生成提示词
    
    核心功能：
    T ~ π_proposer(T | O_t, Target(e*))
    
    根据当前观测和目标元素，生成自然语言任务指令
    """
    
    def __init__(self):
        """初始化 Proposer Prompt Builder"""
        pass
    
    def system_message(self) -> Dict[str, str]:
        """
        返回 Proposer 的 System Message
        
        定义 Proposer 的角色和任务生成规则
        """
        system_text = (
            "You are a helpful web agent expert and task designer. "
            "Your goal is to propose natural, user-friendly task instructions based on the current webpage context.\n\n"
            "Given:\n"
            "1. The current page content (AxTree)\n"
            "2. A specific target element to interact with\n\n"
            "Generate a concise, high-level user intent (task instruction) that:\n"
            "- Requires interacting with the target element\n"
            "- Sounds like a natural user request\n"
            "- Is feasible but challenging (requires some reasoning)\n"
            "- Follows the Zone of Proximal Development (ZPD) principle: difficult but achievable\n\n"
            "Output ONLY the task instruction, without explanation or additional text."
        )
        
        return {"text": system_text}
    
    def construct_generation_prompt(
        self, 
        obs_axtree: str, 
        target_element: str,
        target_node_id: Optional[str] = None,
        max_axtree_length: int = 8000
    ) -> str:
        """
        构建任务生成提示词
        
        Args:
            obs_axtree: 当前页面的 AxTree 观测
            target_element: 目标元素描述（来自边数据）
            target_node_id: 目标节点 ID（可选，用于调试）
            max_axtree_length: AxTree 最大长度限制
        
        Returns:
            str: 完整的 User Message
        """
        # 截断 AxTree 以防止超出上下文长度
        truncated_axtree = obs_axtree[:max_axtree_length]
        if len(obs_axtree) > max_axtree_length:
            truncated_axtree += "\n... (truncated)"
            logger.debug(f"AxTree truncated from {len(obs_axtree)} to {max_axtree_length} chars")
        
        # 构造 User Message
        # 显式地将 Target 放入 Prompt，强制模型关注特定元素
        # 符合 "Directed Exploration" 和 "Conditional Generation" 的定义
        user_msg = (
            f"Current Page Content (AxTree):\n"
            f"{truncated_axtree}\n\n"
            f"Target Element to Interact With:\n"
            f"{target_element}\n\n"
            f"Please generate a natural user task instruction that requires interacting with the target element."
        )
        
        # 如果提供了 target_node_id，添加到日志（不添加到 prompt）
        if target_node_id:
            logger.debug(f"Generating instruction for target node: {target_node_id}")
        
        return user_msg
    
    def parse_instruction(self, raw_output: str) -> str:
        """
        解析和清理 LLM 生成的任务指令
        
        Args:
            raw_output: LLM 的原始输出
        
        Returns:
            str: 清理后的任务指令
        """
        instruction = raw_output.strip()
        
        # 移除可能的前缀（如 "Task:", "Instruction:", etc.）
        prefixes_to_remove = [
            "Task:", 
            "Instruction:", 
            "User Intent:", 
            "Goal:",
            "Task Instruction:",
            "The task is to",
            "The task is:"
        ]
        
        for prefix in prefixes_to_remove:
            if instruction.startswith(prefix):
                instruction = instruction[len(prefix):].strip()
                break
        
        # 移除可能的引号包裹
        if (instruction.startswith('"') and instruction.endswith('"')) or \
           (instruction.startswith("'") and instruction.endswith("'")):
            instruction = instruction[1:-1].strip()
        
        # 确保指令不为空
        if not instruction:
            logger.warning("Generated instruction is empty after parsing")
            instruction = "Interact with the specified element"
        
        # 确保指令以句号结尾（如果还没有标点符号）
        if instruction and instruction[-1] not in ['.', '!', '?']:
            instruction += '.'
        
        return instruction
    
    def construct_curriculum_explanation(
        self,
        edge_data: Dict,
        horizon_k: int = 1
    ) -> str:
        """
        构建课程选择的解释（用于日志和调试）
        
        Args:
            edge_data: 边的统计数据
            horizon_k: 规划视野
        
        Returns:
            str: 课程选择的解释文本
        """
        success = edge_data.get("success", 0)
        total = edge_data.get("total", 0)
        target_element = edge_data.get("target_element", "unknown")
        
        if total == 0:
            p_succ = 0.5
            exploration_status = "unexplored"
        else:
            p_succ = success / total
            if p_succ < 0.3:
                exploration_status = "challenging"
            elif p_succ > 0.7:
                exploration_status = "easy"
            else:
                exploration_status = "optimal ZPD"
        
        explanation = (
            f"Curriculum Selection (K={horizon_k}):\n"
            f"  Target Element: {target_element}\n"
            f"  Success Rate: {p_succ:.2f} ({success}/{total})\n"
            f"  Status: {exploration_status}\n"
        )
        
        if horizon_k > 1:
            explanation += f"  Task Type: Multi-step Planning ({horizon_k} hops)\n"
        else:
            explanation += f"  Task Type: Atomic Operation\n"
        
        return explanation

