from __future__ import annotations
from .node import Node
from .trace import Trace
from .trajectory import TrajectoryStep
from typing import Sequence, Callable, Tuple
import json
import logging
import os
import glob
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class Graph:
    def __init__(
            self, 
            root_url: str, 
            exp_dir: str, 
            encoder_fn: Callable[[dict], list[float]], # [必须] 外部传入的 Encoder 函数
            encoder_name: str = "",                     # [新增] Encoder 名称
            similarity_threshold: float = 0.95,        # [参数] 判定同一状态的阈值
            allowlist_patterns: Sequence[str] = tuple(), 
            denylist_patterns: Sequence[str] = tuple(), 
            resume: bool = False
        ):
        """
        World Model Graph
        :param encoder_fn: 函数，接收 obs(dict) 返回 embedding(list[float])
        :param encoder_name: Encoder 的名称（例如 "openai/clip-vit-base-patch32"）
        :param similarity_threshold: 余弦相似度阈值，大于此值视为同一节点
        """
        self.nodes: dict[str, Node] = {} # Key: node_id (e.g., "node_0")
        # Key: (source_id, target_id), Value: stats dict
        self.edges: dict[tuple[str, str], dict] = {}
        
        self.explored_nodes = []
        self.unexplored_nodes = []
        self.exp_dir = os.path.join(exp_dir, "graph")
        self.allowlist_patterns = allowlist_patterns
        self.denylist_patterns = denylist_patterns
        
        # World Model 核心组件
        self.encoder_fn = encoder_fn
        self.encoder_name = encoder_name
        self.similarity_threshold = similarity_threshold
        self.current_node: Node | None = None # 指针：Agent 当前认为自己所在的抽象节点
        
        # 统计信息
        self.total_steps: int = 0  # 全局累计步数

        if not resume:
            os.makedirs(self.exp_dir, exist_ok=True)
            self._save_graph_info(root_url)

    def _save_graph_info(self, root_url: str = ""):
        from datetime import datetime
        
        graph_info = {
            "root_node_id": "node_0" if "node_0" in self.nodes else "",
            "encoder_name": self.encoder_name,
            "similarity_threshold": self.similarity_threshold,
            
            "total_steps": self.total_steps,
            "total_nodes": len(self.nodes),
            "last_updated": datetime.now().isoformat()
        }
        with open(os.path.join(self.exp_dir, "graph_info.json"), "w") as f:
            json.dump(graph_info, f, indent=4)

    # =========================================================
    #  Core World Model Logic: Encoding & Matching
    # =========================================================

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """计算两个向量的余弦相似度"""
        if not vec_a or not vec_b:
            return 0.0
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def _find_matching_node(self, embedding: list[float]) -> Tuple[Node | None, float]:
        """
        [核心逻辑] 在现有节点中寻找最相似的节点 (KNN Search)
        返回: (BestNode, SimilarityScore)
        """
        best_sim = -1.0
        best_node = None
        
        # 注意：对于几千个节点的规模，线性扫描速度足够快。
        # 如果规模达到百万级，此处应替换为 FAISS 索引查询。
        for node in self.nodes.values():
            if node.embedding is None:
                continue
            sim = self._cosine_similarity(embedding, node.embedding)
            if sim > best_sim:
                best_sim = sim
                best_node = node
        
        return best_node, best_sim

    def _create_node(self, obs: dict, embedding: list[float], hint: str = "", parent_node: Node = None) -> Node:
        """创建一个新的抽象状态节点"""
        # 生成独立于内容的 ID: node_0, node_1, ...
        node_id = f"node_{len(self.nodes)}"
        url = obs.get("url", "")
        
        # 计算 depth：如果有父节点，则为父节点 depth + 1，否则为 0（root 节点）
        depth = parent_node.depth + 1 if parent_node else 0
        
        node_exp_dir = os.path.join(self.exp_dir, node_id)
        
        new_node = Node(
            node_id=node_id,
            embedding=embedding,
            url=url,
            hint=hint,
            tasks={},
            exploration_tasks={},
            children=[],
            depth=depth,
            visit_count=0,
            traces=[],
            visited=False,
            exp_dir=node_exp_dir
        )
        
        self.nodes[node_id] = new_node
        self.unexplored_nodes.append(new_node)
        
        # 立即注册可交互元素（包括 root 节点）
        from .element_utils import extract_interactive_elements
        
        # 优先使用 obs 中已提取的 interactive_elements
        if "interactive_elements" in obs:
            elements = obs["interactive_elements"]
        else:
            # 如果没有，从 axtree 中提取
            elements = extract_interactive_elements(
                obs.get("axtree_txt", ""),
                obs.get("extra_element_properties", {})
            )
        
        if elements:
            new_node.register_elements(elements)
            logger.info(f"Created NEW Abstract State {node_id} (URL: {url}, Depth: {depth}) with {len(elements)} interactive elements")
        else:
            logger.info(f"Created NEW Abstract State {node_id} (URL: {url}, Depth: {depth})")
        
        # 保存截图（如果有）
        if "screenshot" in obs:
            new_node.update_save(
                save_traces=False, 
                save_info=True,
                save_screenshot=True,
                screenshot=obs["screenshot"]
            )
        else:
            new_node.update_save(save_traces=False, save_info=True)
        
        return new_node

    # =========================================================
    #  Phase 1-3 Transition Processing (Input -> Process)
    # =========================================================

    def process_transition(self, obs_t: dict, action: str, obs_t1: dict, success: bool, hint: str = "", thought: str = None) -> Node:
        """
        [核心接口] 处理 (O_t, a_t, O_{t+1}) 三元组
        
        Process:
        1. Encode: 计算向量 V_t, V_{t+1}
        2. Match: 匹配或创建节点 S_t, S_{t+1}
        3. Update: 更新边 (S_t -> S_{t+1}) 的统计数据
        4. Create Trace: 创建从 source_node 到 target_node 的 Trace 并添加到 traces
        
        Output:
        返回当前的新状态节点 S_{t+1}
        """
        
        # --- 0. 更新步数统计 ---
        self.total_steps += 1
        
        # --- 1. 确定源节点 S_t ---
        # 理想情况下，S_t 应该是 self.current_node。
        # 但为了处理冷启动或定位漂移，我们检查 V_t
        if self.current_node:
            source_node = self.current_node
        else:
            # 冷启动：寻找或创建起点
            vec_t = self.encoder_fn(obs_t)
            match_node, sim = self._find_matching_node(vec_t)
            if match_node and sim >= self.similarity_threshold:
                source_node = match_node
                logger.info(f"Localized to existing start node: {source_node.node_id}")
            else:
                source_node = self._create_node(obs_t, vec_t, hint="Start/Root", parent_node=None)
        
        # 确保 source_node 有元素注册（兼容旧数据或首次访问）
        if len(source_node.interactive_elements) == 0:
            from .element_utils import extract_interactive_elements
            
            # 优先使用 obs 中已提取的 interactive_elements
            if "interactive_elements" in obs_t:
                elements = obs_t["interactive_elements"]
            else:
                elements = extract_interactive_elements(
                    obs_t.get("axtree_txt", ""),
                    obs_t.get("extra_element_properties", {})
                )
            
            if elements:
                source_node.register_elements(elements)
                logger.info(f"Registered {len(elements)} interactive elements for source node {source_node.node_id}")

        # --- 2. 确定目标节点 S_{t+1} ---
        vec_t1 = self.encoder_fn(obs_t1)
        match_node, sim = self._find_matching_node(vec_t1)
        
        is_new_node = False
        if match_node and sim >= self.similarity_threshold:
            logger.info(f"Matched existing state {match_node.node_id} (Sim: {sim:.4f})")
            target_node = match_node
            # 更新访问计数
            target_node.visit_count += 1
            target_node.update_save(save_traces=False, save_info=True)
        else:
            logger.info(f"State not found (Max Sim: {sim:.4f}). Creating new state.")
            target_node = self._create_node(obs_t1, vec_t1, hint, parent_node=source_node)
            is_new_node = True

        # --- 3. 更新图结构与统计 (Edges) ---
        
        # 3.1 维护父子引用
        if target_node.node_id not in source_node.children:
            source_node.children.append(target_node.node_id)
            source_node.update_save(save_traces=False, save_info=True) 
            
        # 3.2 更新边统计信息 (Action Success Rate)
        self._update_edge_stats(source_node.node_id, target_node.node_id, success, action=action, target_element=action)
        
        # 3.2.5 记录 action 到 source_node 的 action_history（用于 exploration mask）
        source_node.record_action(action)

        # 3.3 创建 Trace 并添加到 target_node 的 traces
        # 为新节点创建 trace（无论成功与否），用于后续 replay
        try:
            # 创建 TrajectoryStep
            step = TrajectoryStep(
                action=action,
                parsed_action=action,
                thought=thought or "",
                observation=obs_t1,
                misc={"hint": hint, "success": success, "source_node": source_node.node_id}
            )
            
            # 创建 Trace
            start_url = obs_t.get("url", "")
            end_url = obs_t1.get("url", "")
            trace = Trace.from_trajectory_steps(
                steps=[step],
                start_url=start_url,
                end_url=end_url,
                misc={"source_node": source_node.node_id, "target_node": target_node.node_id, "is_new_node": is_new_node}
            )
            
            # 新节点必须添加 trace，已存在的节点如果还没有 trace 也添加
            if is_new_node:
                target_node.add_trace(trace)
                logger.info(f"Created trace for NEW node {target_node.node_id}: {start_url} -> {end_url} (success={success})")
            elif len(target_node.traces) == 0:
                target_node.add_trace(trace)
                logger.info(f"Created trace for existing node {target_node.node_id}: {start_url} -> {end_url} (success={success})")
        except Exception as e:
            logger.warning(f"Failed to create trace: {e}")

        # 3.4 状态管理
        self.add_to_explored(source_node)
        
        # 3.5 移动指针
        self.current_node = target_node
        
        # 3.6 保存全局信息
        self._save_graph_info()
        
        return target_node
    
    def _update_edge_stats(self, source_id: str, target_id: str, success: bool, action: str = "", target_element: str = ""):
        """更新边的统计信息 (Success Rate)"""
        key = (source_id, target_id)
        if key not in self.edges:
            self.edges[key] = {
                "action": action,
                "target_element": target_element,
                "success": 0,
                "total": 0
            }
        
        self.edges[key]["total"] += 1
        if success:
            self.edges[key]["success"] += 1
        
        # 实时保存边数据
        self._save_edges()

    def _save_edges(self):
        """保存边为数组格式"""
        edges_array = []
        for (source_id, target_id), stats in self.edges.items():
            edges_array.append({
                "source_node_id": source_id,
                "target_node_id": target_id,
                "action": stats.get("action", ""),
                "target_element": stats.get("target_element", ""),
                "stats": {
                    "success": stats.get("success", 0),
                    "total": stats.get("total", 0)
                }
            })
        
        with open(os.path.join(self.exp_dir, "edges.json"), "w") as f:
            json.dump(edges_array, f, indent=4)
    
    # =========================================================
    #  Outputs for Proposer & Solver
    # =========================================================

    def get_out_edges(self, node_id: str) -> list[dict]:
        """
        Output: 为 Proposer 提供当前节点可达的边及其成功率
        """
        out_edges = []
        for (u, v), stats in self.edges.items():
            if u == node_id:
                total = stats["total"]
                success = stats["success"]
                
                # 贝叶斯平滑 (Bayesian Smoothing)
                # 初始先验概率为 0.5 (1/2)，随着 total 增加，概率趋向于真实值
                p_succ = (success + 1) / (total + 2)
                
                out_edges.append({
                    "target_node_id": v,
                    "succ_rate": p_succ,       # Proposer 使用此指标判断是否为 50%
                    "total_attempts": total,
                    "action_skill": stats.get("target_element", "")
                })
        return out_edges

    def add_to_explored(self, node: Node):
        if node in self.unexplored_nodes:
            self.unexplored_nodes.remove(node)
        
        if node not in self.explored_nodes:
            self.explored_nodes.append(node)
            node.visited = True
            node.update_save(save_traces=False, save_info=True)

    def get_next_node(self) -> Node | None:
        """获取 Frontier 节点"""
        if len(self.unexplored_nodes) == 0:
            return None
        # 这里可以使用更复杂的策略（例如优先探索 embedding 距离远的节点）
        return self.unexplored_nodes[0]

    # =========================================================
    #  Loading
    # =========================================================

    @staticmethod
    def load(path: str, encoder_fn: Callable, encoder_name: str = "", load_steps: bool=True, load_traces: bool=True, load_images: bool=True) -> Graph:
        """从硬盘加载图"""
        logger.info(f"Loading graph from {path}")
        
        graph_info_path = os.path.join(path, "graph_info.json")
        # 兼容路径处理
        if not os.path.exists(graph_info_path):
            path = os.path.join(path, "graph")
            graph_info_path = os.path.join(path, "graph_info.json")
            
        if not os.path.exists(graph_info_path):
             raise FileNotFoundError(f"Graph info not found at {graph_info_path}")

        with open(graph_info_path, "r") as f:
            graph_info = json.load(f)
            
        # 初始化 Graph，必须重新注入 encoder_fn
        graph = Graph(
            root_url="",  # 不再需要 root_url
            exp_dir=os.path.dirname(path), 
            encoder_fn=encoder_fn,
            encoder_name=encoder_name or graph_info.get("encoder_name", ""),
            similarity_threshold=graph_info.get("similarity_threshold", 0.95),
            resume=True
        )
        
        # 恢复统计信息
        graph.total_steps = graph_info.get("total_steps", 0)
        
        # 1. 加载 Nodes
        node_dirs = glob.glob(os.path.join(path, "node_*"))
        
        for node_dir in node_dirs:
            try:
                # Node.load 已经适配了新版的 node_id 和 embedding
                node = Node.load(node_dir, load_steps=load_steps, load_traces=load_traces, load_images=load_images)
                graph.nodes[node.node_id] = node 
                
                if node.visited:
                    graph.explored_nodes.append(node)
                else:
                    graph.unexplored_nodes.append(node)
            except Exception as e:
                logger.error(f"Error loading node from {node_dir}: {e}")
        
        # 2. 加载 Edges（新格式：数组）
        edges_path = os.path.join(path, "edges.json")
        if os.path.exists(edges_path):
            try:
                with open(edges_path, "r") as f:
                    edges_array = json.load(f)
                    
                # 判断格式：如果是数组则为新格式，否则为旧格式
                if isinstance(edges_array, list):
                    # 新格式：数组
                    for edge in edges_array:
                        key = (edge["source_node_id"], edge["target_node_id"])
                        graph.edges[key] = {
                            "action": edge.get("action", ""),
                            "target_element": edge.get("target_element", ""),
                            "success": edge.get("stats", {}).get("success", 0),
                            "total": edge.get("stats", {}).get("total", 0)
                        }
                else:
                    # 旧格式：对象（向后兼容）
                    graph.edges = {
                        tuple(k.split("|")): v for k, v in edges_array.items()
                    }
            except Exception as e:
                logger.error(f"Error loading edges: {e}")
        
        # 恢复 Current Node 指针
        if graph.explored_nodes:
            graph.current_node = graph.explored_nodes[-1]
        elif graph.nodes:
            # 排序确保确定性
            sorted_ids = sorted(graph.nodes.keys(), key=lambda x: int(x.split('_')[-1]))
            graph.current_node = graph.nodes[sorted_ids[0]]

        logger.info(f"Loaded graph with {len(graph.nodes)} nodes, {graph.total_steps} total steps.")
        return graph