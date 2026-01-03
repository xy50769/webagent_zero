from __future__ import annotations
from .node import Node
from .trace import Trace
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
            similarity_threshold: float = 0.95,        # [参数] 判定同一状态的阈值
            allowlist_patterns: Sequence[str] = tuple(), 
            denylist_patterns: Sequence[str] = tuple(), 
            resume: bool = False
        ):
        """
        World Model Graph
        :param encoder_fn: 函数，接收 obs(dict) 返回 embedding(list[float])
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
        self.similarity_threshold = similarity_threshold
        self.current_node: Node | None = None # 指针：Agent 当前认为自己所在的抽象节点

        if not resume:
            os.makedirs(self.exp_dir, exist_ok=True)
            self._save_graph_info(root_url)

    def _save_graph_info(self, root_url: str):
        graph_info = {
            "root_url": root_url,
            "allowlist_patterns": self.allowlist_patterns,
            "denylist_patterns": self.denylist_patterns,
            "similarity_threshold": self.similarity_threshold
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

    def _create_node(self, obs: dict, embedding: list[float], hint: str = "") -> Node:
        """创建一个新的抽象状态节点"""
        # 生成独立于内容的 ID: node_0, node_1, ...
        node_id = f"node_{len(self.nodes)}"
        url = obs.get("url", "")
        
        node_exp_dir = os.path.join(self.exp_dir, node_id)
        
        new_node = Node(
            node_id=node_id,
            embedding=embedding,
            url=url,
            hint=hint,
            tasks={},
            exploration_tasks={},
            children=[],
            prefixes=[],
            visited=False,
            exp_dir=node_exp_dir
        )
        
        self.nodes[node_id] = new_node
        self.unexplored_nodes.append(new_node)
        
        logger.info(f"✨ Created NEW Abstract State {node_id} (URL: {url})")
        return new_node

    # =========================================================
    #  Phase 1-3 Transition Processing (Input -> Process)
    # =========================================================

    def process_transition(self, obs_t: dict, action: str, obs_t1: dict, success: bool, hint: str = "") -> Node:
        """
        [核心接口] 处理 (O_t, a_t, O_{t+1}) 三元组
        
        Process:
        1. Encode: 计算向量 V_t, V_{t+1}
        2. Match: 匹配或创建节点 S_t, S_{t+1}
        3. Update: 更新边 (S_t -> S_{t+1}) 的统计数据
        
        Output:
        返回当前的新状态节点 S_{t+1}
        """
        
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
                source_node = self._create_node(obs_t, vec_t, hint="Start/Root")

        # --- 2. 确定目标节点 S_{t+1} ---
        vec_t1 = self.encoder_fn(obs_t1)
        match_node, sim = self._find_matching_node(vec_t1)
        
        if match_node and sim >= self.similarity_threshold:
            logger.info(f"Matched existing state {match_node.node_id} (Sim: {sim:.4f})")
            target_node = match_node
        else:
            logger.info(f"State not found (Max Sim: {sim:.4f}). Creating new state.")
            target_node = self._create_node(obs_t1, vec_t1, hint)

        # --- 3. 更新图结构与统计 (Edges) ---
        
        # 3.1 维护父子引用
        if target_node.node_id not in source_node.children:
            source_node.children.append(target_node.node_id)
            source_node.update_save(save_prefix=False) 
            
        # 3.2 更新边统计信息 (Action Success Rate)
        self._update_edge_stats(source_node.node_id, target_node.node_id, success, target_element=action)

        # 3.3 状态管理
        self.add_to_explored(source_node)
        
        # 3.4 移动指针
        self.current_node = target_node
        
        return target_node

    def _update_edge_stats(self, source_id: str, target_id: str, success: bool, target_element: str = None):
        """更新边的统计信息 (Success Rate)"""
        key = (source_id, target_id)
        if key not in self.edges:
            self.edges[key] = {
                "success": 0,
                "total": 0,
                "target_element": target_element
            }
        
        self.edges[key]["total"] += 1
        if success:
            self.edges[key]["success"] += 1
        
        # 实时保存边数据
        self._save_edges()

    def _save_edges(self):
        # JSON 不支持 Tuple key，转换为 "u|v" 字符串格式
        serializable_edges = {f"{k[0]}|{k[1]}": v for k, v in self.edges.items()}
        with open(os.path.join(self.exp_dir, "edges.json"), "w") as f:
            json.dump(serializable_edges, f, indent=4)

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
            node.update_save(save_prefix=False)

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
    def load(path: str, encoder_fn: Callable, load_steps: bool=True, load_prefixes: bool=True, load_images: bool=True) -> Graph:
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
            root_url=graph_info.get("root_url", ""), 
            exp_dir=os.path.dirname(path), 
            encoder_fn=encoder_fn, 
            similarity_threshold=graph_info.get("similarity_threshold", 0.95),
            allowlist_patterns=graph_info.get("allowlist_patterns", []),
            denylist_patterns=graph_info.get("denylist_patterns", []),
            resume=True
        )
        
        # 1. 加载 Nodes
        node_dirs = glob.glob(os.path.join(path, "node_*"))
        
        for node_dir in node_dirs:
            try:
                # Node.load 已经适配了新版的 node_id 和 embedding
                node = Node.load(node_dir, load_steps=load_steps, load_prefix=load_prefixes, load_images=load_images)
                graph.nodes[node.node_id] = node 
                
                if node.visited:
                    graph.explored_nodes.append(node)
                else:
                    graph.unexplored_nodes.append(node)
            except Exception as e:
                logger.error(f"Error loading node from {node_dir}: {e}")
        
        # 2. 加载 Edges
        edges_path = os.path.join(path, "edges.json")
        if os.path.exists(edges_path):
            try:
                with open(edges_path, "r") as f:
                    raw_edges = json.load(f)
                    # 还原 Tuple Key
                    graph.edges = {
                        tuple(k.split("|")): v for k, v in raw_edges.items()
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

        logger.info(f"Loaded graph with {len(graph.nodes)} nodes.")
        return graph