import logging
import random
import numpy as np
from typing import Optional, Tuple, Dict, List

from .server.llm_engine import LLMEngine
from .prompt_builders.proposer_prompt_builder import ProposerPromptBuilder
from ..core.node import Node
from ..core.graph import Graph

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class TaskProposer:
    """
    [Task Proposer]
    角色: Teacher (课程设计师)
    
    核心职责:
    1. Curriculum Selection: 基于 P_succ ~ 0.5 和 Horizon K 选择 Graph 中的目标边/节点。
    2. Conditional Generation: 基于选定的 Target 生成自然语言指令。
    
    数学描述:
    - Input: O_t (观测), S_t (抽象状态), E_out (候选边), g* (目标引导), K (视野)
    - Output: T (任务指令), Verification Ground Truth
    - Process: A) Curriculum Selection, B) Conditional Generation
    - Reward: R = α·R_curriculum + β·R_feasibility + γ·R_diversity
    """
    def __init__(self, llm_engine: LLMEngine, diversity_weight: float = 0.1):
        """
        初始化 Task Proposer
        
        Args:
            llm_engine: LLM 引擎，用于生成任务指令（mode="proposer"）
            diversity_weight: 多样性权重，用于平衡难度和探索
        """
        self.llm = llm_engine
        self.diversity_weight = diversity_weight
        
        # 初始化 Prompt Builder
        self.prompt_builder = ProposerPromptBuilder()

    def _calculate_curriculum_score(self, edge_data: Dict) -> float:
        """
        计算边的课程分数 (Score 越小越好)
        
        优化目标: 
        1. 难度适中: |P_succ - 0.5| 越小越好 (Zone of Proximal Development)
        2. 多样性: N(e) 越小越好 (鼓励探索冷门边)
        
        数学公式:
        Score(e) = |P_succ(e) - 0.5| - λ * (1 / sqrt(N(e) + 1))
        
        Args:
            edge_data: 边的统计数据，包含 success, total 等信息
        
        Returns:
            float: 课程分数，越小越好
        """
        # 1. 计算成功率 P_succ
        success = edge_data.get("success", 0)
        total = edge_data.get("total", 0)
        
        if total == 0:
            # 未探索的边，假设初始成功率为 0.5 (最大不确定性)
            p_succ = 0.5
        else:
            p_succ = success / total
        
        # 2. 难度分数 (Regret Minimization / ZPD)
        # P_succ 接近 0.5 时，difficulty_score 接近 0（最优）
        difficulty_score = abs(p_succ - 0.5)
        
        # 3. 多样性分数 (Diversity Reward)
        # 访问次数越少，diversity_score 越大
        n_visits = total
        diversity_score = 1.0 / np.sqrt(n_visits + 1)
        
        # 4. 综合打分
        # 我们希望 difficulty_score 小（接近 0.5）且 diversity_score 大（访问少）
        # Score = Difficulty - λ * Diversity（越小越好）
        final_score = difficulty_score - (self.diversity_weight * diversity_score)
        
        return final_score

    def select_target(self, node: Node, graph: Graph, horizon_k: int = 1) -> Optional[Tuple[str, Dict]]:
        """
        Phase A: Curriculum Selection via World Model
        根据 Horizon K 和 P_succ 选择最佳目标边 (Target Edge)。
        
        数学公式:
        e* = argmin_{e ∈ E_out(S_t)} |P_succ(e) - 0.5|
        
        Args:
            node: 当前节点 (S_t)
            graph: 世界模型 (Skill Graph)
            horizon_k: 规划视野
                - K=1: 练习原子操作 (Atomic Skills)
                - K>1: 练习规划 (Planning)
        
        Returns:
            Optional[Tuple[target_node_id, edge_data]]: 目标节点 ID 和边数据
        """
        if horizon_k == 1:
            # === K=1: 练习原子操作 (Atomic Skills) ===
            # 获取从当前节点出发的所有边
            if not node.children:
                logger.info(f"[Proposer] No children from node {node.node_id}. Cannot propose curriculum task.")
                return None
            
            # 遍历所有子节点，找到最佳的边
            best_target = None
            best_score = float('inf')
            
            for child_id in node.children:
                # 获取边的统计数据
                edge_key = f"{node.node_id}|{child_id}"
                edge_data = graph.edges.get(edge_key, {"success": 0, "total": 0, "target_element": "unknown"})
                
                # 计算课程分数
                score = self._calculate_curriculum_score(edge_data)
                
                if score < best_score:
                    best_score = score
                    best_target = (child_id, edge_data)
            
            if best_target is None:
                logger.info(f"[Proposer] No valid target found for node {node.node_id}.")
                return None
            
            target_node_id, edge_data = best_target
            success = edge_data.get("success", 0)
            total = edge_data.get("total", 0)
            p_succ = success / total if total > 0 else 0.5
            
            # 使用 Prompt Builder 生成课程选择的解释
            curriculum_explanation = self.prompt_builder.construct_curriculum_explanation(
                edge_data=edge_data,
                horizon_k=1
            )
            logger.info(f"🎯 [Proposer] {curriculum_explanation}")
            
            return best_target
            
        else:
            # === K>1: 练习规划 (Planning) ===
            # 在图上进行随机游走，寻找 K 步之外的节点
            current_node_id = node.node_id
            path = []
            
            for step in range(horizon_k):
                current_node = graph.nodes.get(current_node_id)
                if not current_node or not current_node.children:
                    logger.warning(f"[Proposer] Cannot continue path at step {step}, node {current_node_id} has no children.")
                    break
                
                # 随机选择一个子节点（可以改进为选择高成功率的边作为"桥梁"）
                next_node_id = random.choice(current_node.children)
                edge_key = f"{current_node_id}|{next_node_id}"
                edge_data = graph.edges.get(edge_key, {"success": 0, "total": 0, "target_element": "unknown"})
                
                path.append((next_node_id, edge_data))
                current_node_id = next_node_id
            
            if not path:
                logger.info(f"[Proposer] Cannot find {horizon_k}-hop path from node {node.node_id}.")
                return None
            
            # 多步任务的目标是最后一条边的终点
            target_node_id, final_edge_data = path[-1]
            
            # 使用 Prompt Builder 生成课程选择的解释
            curriculum_explanation = self.prompt_builder.construct_curriculum_explanation(
                edge_data=final_edge_data,
                horizon_k=horizon_k
            )
            logger.info(f"🎯 [Proposer] {curriculum_explanation}")
            logger.info(f"   Multi-hop path: {node.node_id} -> ... -> {target_node_id} ({len(path)} steps)")
            
            return (target_node_id, final_edge_data)

    def generate_instruction(self, obs_axtree: str, target_element: str, target_node_id: str = None) -> str:
        """
        Phase B: Conditional Generation via LLM
        
        数学公式:
        T ~ π_proposer(T | O_t, Target(e*))
        
        Args:
            obs_axtree: 当前页面的 AxTree 观测
            target_element: 目标元素描述（来自边数据）
            target_node_id: 目标节点 ID（用于调试）
        
        Returns:
            str: 生成的任务指令
        """
        # 使用 Prompt Builder 构造提示词
        system_msg = self.prompt_builder.system_message()['text']
        user_msg = self.prompt_builder.construct_generation_prompt(
            obs_axtree=obs_axtree,
            target_element=target_element,
            target_node_id=target_node_id
        )
        
        # 调用 LLM (使用 proposer adapter)
        try:
            raw_output = self.llm.generate(
                system_msg=system_msg,
                user_msg=user_msg,
                mode="proposer",
                temperature=0.7  # 保持一定的多样性
            )
            
            # 使用 Prompt Builder 解析和清理输出
            instruction = self.prompt_builder.parse_instruction(raw_output)
            
            logger.info(f"📝 [Proposer] Generated Instruction: {instruction}")
            
        except Exception as e:
            logger.error(f"[Proposer] Failed to generate instruction: {e}")
            # 回退：生成一个简单的指令
            instruction = f"Please interact with the element: {target_element}"
        
        return instruction

    def propose_task(
        self, 
        node: Node, 
        graph: Graph, 
        obs_axtree: str, 
        horizon_k: int = 1, 
        target_guidance: Optional[Tuple[str, Dict]] = None
    ) -> Optional[Tuple[str, str, Dict]]:
        """
        主入口函数：为 Solver 生成 Feasible but Difficult 的任务
        
        Args:
            node: 当前节点 (S_t)
            graph: 世界模型 (Skill Graph)
            obs_axtree: 当前观测 (O_t)
            horizon_k: 规划视野 (K)
                - K=1: 原子操作任务
                - K>1: 多步规划任务
            target_guidance: 外部强制指定的 Target (g*)
                格式: (target_node_id, edge_data)
        
        Returns:
            Optional[Tuple[instruction, target_node_id, verification_data]]:
                - instruction: 任务指令 (T)
                - target_node_id: 目标节点 ID
                - verification_data: 验证信息（包含 target_element 等）
                如果无法生成任务，返回 None
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"[Proposer] Starting Task Proposal")
        logger.info(f"{'='*60}")
        logger.info(f"Current Node: {node.node_id}")
        logger.info(f"Horizon K: {horizon_k}")
        
        # 1. Selection Phase: Curriculum Selection via World Model
        if target_guidance:
            # 如果有外部指导 (比如来自人工干预或特定的探索策略)，直接使用
            target_node_id, edge_data = target_guidance
            logger.info(f"🎯 [Proposer] Using Guidance Target: {node.node_id} -> {target_node_id}")
            logger.info(f"   Target Element: {edge_data.get('target_element', 'unknown')}")
        else:
            # 否则使用内部 Curriculum 策略选择
            result = self.select_target(node, graph, horizon_k)
            
            if result is None:
                # 如果选不出 Target (例如新节点无边)，返回 None
                # 外层循环应转为 Exploration 模式
                logger.info(f"[Proposer] Cannot select target from node {node.node_id}. Switching to Exploration mode.")
                return None
            
            target_node_id, edge_data = result

        # 2. Generation Phase: Conditional Generation via LLM
        target_element = edge_data.get("target_element", "unknown element")
        instruction = self.generate_instruction(obs_axtree, target_element, target_node_id)
        
        logger.info(f"✅ [Proposer] Task Proposal Complete")
        logger.info(f"   Instruction: {instruction}")
        logger.info(f"   Target: {node.node_id} -> {target_node_id}")
        logger.info(f"{'='*60}\n")
        
        # 3. 返回 (指令, 目标节点 ID, 验证数据)
        verification_data = {
            "target_node_id": target_node_id,
            "target_element": target_element,
            "source_node_id": node.node_id,
            "horizon": horizon_k,
            "edge_stats": edge_data
        }
        
        return instruction, target_node_id, verification_data

    def calculate_reward(
        self, 
        edge_data: Dict, 
        is_valid: bool, 
        solver_success: bool = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5
    ) -> float:
        """
        计算 Proposer 的奖励 (用于 RL 阶段更新 Proposer 参数)
        
        数学公式:
        R_proposer = α·R_curriculum + β·R_feasibility + γ·R_diversity
        
        Args:
            edge_data: 边的统计数据
            is_valid: 任务是否有效（无幻觉）
            solver_success: Solver 是否成功（可选，用于更新 P_succ）
            alpha: Curriculum Reward 权重
            beta: Feasibility Penalty 权重
            gamma: Diversity Reward 权重
        
        Returns:
            float: 总奖励值
        """
        C = 1.0  # Feasibility Penalty 常数

        # 1. Feasibility Penalty
        # 严厉惩罚幻觉：如果生成的任务描述了页面上不存在的元素
        if not is_valid:
            logger.warning(f"[Proposer Reward] Task is invalid (hallucination). Penalty: -{C * beta:.2f}")
            return -C * beta

        # 2. Curriculum Reward
        # R_curriculum = 1 - 2 * |P_succ - 0.5|
        # 奖励那些让 Solver 处于"懂与不懂之间"的任务
        success = edge_data.get("success", 0)
        total = edge_data.get("total", 0)
        
        if total == 0:
            p_succ = 0.5  # 初始假设
        else:
            p_succ = success / total
        
        r_curriculum = 1.0 - 2.0 * abs(p_succ - 0.5)
        # P_succ = 0.5 -> R = 1.0 (最优)
        # P_succ = 0 or 1 -> R = 0.0 (太简单或太难)

        # 3. Diversity Reward
        # R_diversity = 1 / sqrt(N(e) + 1)
        # 鼓励探索 Skill Graph 中被冷落的边
        n_visits = total
        r_diversity = 1.0 / np.sqrt(n_visits + 1)

        # 4. 总奖励
        total_reward = (alpha * r_curriculum) + (gamma * r_diversity)
        
        logger.info(f"[Proposer Reward] R_total = {total_reward:.3f}")
        logger.info(f"  R_curriculum = {r_curriculum:.3f} (P_succ = {p_succ:.2f})")
        logger.info(f"  R_diversity = {r_diversity:.3f} (N = {n_visits})")
        
        return total_reward