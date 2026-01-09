from __future__ import annotations
import json
import logging
import os
import glob
import numpy as np
from datetime import datetime
from typing import Callable, Tuple
from .trajectory import TrajectoryStep
from .node import Node
from .trace import Trace
from .element_utils import extract_interactive_elements
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@dataclass
class Graph:
    root_url: str
    exp_dir: str
    encoder_fn: Callable[[dict], list[float]]
    encoder_name: str = ""
    similarity_threshold: float = 0.95
    resume: bool = False
    
    # Internal State 
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[tuple[str, str], dict] = field(default_factory=dict)
    explored_nodes: list[Node] = field(default_factory=list)
    unexplored_nodes: list[Node] = field(default_factory=list)
    current_node: Node | None = None
    total_steps: int = 0

    def __post_init__(self):
        """
        World Model Graph Initialization
        """
        if not self.exp_dir.endswith("graph"):
            self.exp_dir = os.path.join(self.exp_dir, "graph")

        if self.resume and os.path.exists(os.path.join(self.exp_dir, "graph_info.json")):
            logger.info(f"Resuming from existing graph at {self.exp_dir}")
            self._load_from_disk()
        else:
            os.makedirs(self.exp_dir, exist_ok=True)
            self._save_graph_info()
    
    def _load_from_disk(self):
        """Load graph data from disk"""
        graph_info_path = os.path.join(self.exp_dir, "graph_info.json")
        
        with open(graph_info_path, "r") as f:
            graph_info = json.load(f)
        
        self.total_steps = graph_info.get("total_steps", 0)
        
        node_dirs = glob.glob(os.path.join(self.exp_dir, "node_*"))
        for node_dir in sorted(node_dirs):
            node = Node.load(node_dir, load_steps=False, load_traces=True, load_images=False)
            self.nodes[node.node_id] = node 
            if node.visited:
                self.explored_nodes.append(node)
            else:
                self.unexplored_nodes.append(node)
        
        edges_path = os.path.join(self.exp_dir, "edges.json")
        if os.path.exists(edges_path):
            with open(edges_path, "r") as f:
                edges_array = json.load(f)
                for edge in edges_array:
                    key = (edge["source_node_id"], edge["target_node_id"])
                    self.edges[key] = {
                        "action": edge.get("action", ""),
                        "target_element": edge.get("target_element", ""),
                        "explorer_execute_success": edge.get("stats", {}).get("explorer_execute_success", 0),
                        "explorer_execute_total": edge.get("stats", {}).get("explorer_execute_total", 0)
                    }
        
        if self.explored_nodes:
            self.current_node = self.explored_nodes[-1]
        elif self.nodes:
            sorted_ids = sorted(self.nodes.keys(), key=lambda x: int(x.split('_')[-1]))
            self.current_node = self.nodes[sorted_ids[0]]
        
        logger.info(f"Loaded {len(self.nodes)} nodes, {len(self.edges)} edges, {self.total_steps} total steps")

    def _save_graph_info(self):
        graph_info = {
            "root_node_id": "node_0" if "node_0" in self.nodes else "",
            "root_url": self.root_url,
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
        """Cosine similarity between two vectors"""
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
        Find the most similar node to the given embedding
        """
        best_sim = -1.0
        best_node = None
        
        for node in self.nodes.values():
            if node.embedding is None:
                continue
            sim = self._cosine_similarity(embedding, node.embedding)
            if sim > best_sim:
                best_sim = sim
                best_node = node
        
        return best_node, best_sim

    def _create_node(self, obs: dict, embedding: list[float], hint: str = "", parent_node: Node = None) -> Node:
        """Create a new abstract state node"""
        node_id = f"node_{len(self.nodes)}"
        url = obs.get("url", "")
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
        elements = extract_interactive_elements(
            obs.get("axtree_txt", ""),
            obs.get("extra_element_properties", {})
        )
        
        new_node.register_elements(elements)
        logger.info(f"Created NEW Abstract State {node_id} (URL: {url}, Depth: {depth}) with {len(elements)} interactive elements")
        new_node.update_save(
            save_screenshot=True,
            screenshot=obs["screenshot"]
        )
        return new_node

    # =========================================================
    #  Phase 1-3 Transition Processing (Input -> Process)
    # =========================================================

    def process_transition(self, obs_t: dict, action: str, obs_t1: dict, explorer_execute_success: bool, hint: str = "", thought: str = None) -> Node:
        """
        Process transition (O_t, a_t, O_{t+1})
        
        Process:
        1. Encode: Calculate V_t, V_{t+1}
        2. Match: Match or create nodes S_t, S_{t+1}
        3. Update: Update edge (S_t -> S_{t+1}) statistics
        4. Create Trace: Create trace from source_node to target_node and add to traces
        
        Output:
        Return the new state node S_{t+1}
        """
        # --- 0. Update step count ---
        self.total_steps += 1
        
        # --- 1. Determine source node S_t ---
        # Ideally, S_t should be self.current_node.
        # But to handle cold start or location drift, we check V_t
        if self.current_node:
            source_node = self.current_node
        else:
            vec_t = self.encoder_fn(obs_t)
            match_node, sim = self._find_matching_node(vec_t)
            if match_node and sim >= self.similarity_threshold:
                source_node = match_node
                logger.info(f"Localized to existing start node: {source_node.node_id}")
            else:
                source_node = self._create_node(obs_t, vec_t, hint="Start/Root", parent_node=None)
        
        # Ensure source_node has element registration (for old data or first visit)
        if len(source_node.interactive_elements) == 0:
            elements = extract_interactive_elements(
                obs_t.get("axtree_txt", ""),
                obs_t.get("extra_element_properties", {})
            )
            if source_node.interactive_elements_visited:
                visited_bids = {e.get("bid") for e in source_node.interactive_elements_visited}
                elements = [e for e in elements if e.get("bid") not in visited_bids]
            
            if elements:
                source_node.register_elements(elements)
                logger.info(f"Registered {len(elements)} interactive elements for source node {source_node.node_id}")

        # --- 2. Determine target node S_{t+1} ---
        vec_t1 = self.encoder_fn(obs_t1)
        match_node, sim = self._find_matching_node(vec_t1)
        
        is_new_node = False
        if match_node and sim >= self.similarity_threshold:
            logger.info(f"Matched existing state {match_node.node_id} (Sim: {sim:.4f})")
            target_node = match_node
            target_node.visit_count += 1
            target_node.update_save()
        else:
            logger.info(f"State not found (Max Sim: {sim:.4f}). Creating new state.")
            target_node = self._create_node(obs_t1, vec_t1, hint, parent_node=source_node)
            is_new_node = True

        # --- 3. Update graph structure and statistics (Edges) ---
        

        if target_node.node_id not in source_node.children:
            source_node.children.append(target_node.node_id)
            source_node.update_save() 
        self._update_edge_stats(source_node.node_id, target_node.node_id, explorer_execute_success, action=action, target_element=action)
        source_node.record_action(action)
        step = TrajectoryStep(
            action=action,
            parsed_action=action,
            thought=thought or "",
            observation=obs_t1,
            misc={"hint": hint, "explorer_execute_success": explorer_execute_success, "source_node": source_node.node_id}
        )
        
        start_url = obs_t.get("url", "")
        end_url = obs_t1.get("url", "")
        trace = Trace.from_trajectory_steps(
            steps=[step],
            start_url=start_url,
            end_url=end_url,
            misc={"source_node": source_node.node_id, "target_node": target_node.node_id, "is_new_node": is_new_node}
        )
        
        if is_new_node:
            target_node.add_trace(trace)
            logger.info(f"Created trace for NEW node {target_node.node_id}: {start_url} -> {end_url} (explorer_execute_success={explorer_execute_success})")
        elif len(target_node.traces) == 0:
            target_node.add_trace(trace)
            logger.info(f"Created trace for existing node {target_node.node_id}: {start_url} -> {end_url} (explorer_execute_success={explorer_execute_success})")
        
        self.add_to_explored(source_node)
        
        self.current_node = target_node

        self._save_graph_info()
        
        return target_node
    
    def _update_edge_stats(self, source_id: str, target_id: str, explorer_execute_success: bool, action: str = "", target_element: str = ""):
        """Update edge statistics"""
        key = (source_id, target_id)
        if key not in self.edges:
            self.edges[key] = {
                "action": action,
                "target_element": target_element,
                "explorer_execute_success": 0,
                "explorer_execute_total": 0
            }
        
        self.edges[key]["explorer_execute_total"] += 1
        if explorer_execute_success:
            self.edges[key]["explorer_execute_success"] += 1
        
        self._save_edges()

    def _save_edges(self):
        """Save edges as array format"""
        edges_array = []
        for (source_id, target_id), stats in self.edges.items():
            edges_array.append({
                "source_node_id": source_id,
                "target_node_id": target_id,
                "action": stats.get("action", ""),
                "target_element": stats.get("target_element", ""),
                "stats": {
                    "explorer_execute_success": stats.get("explorer_execute_success", 0),
                    "explorer_execute_total": stats.get("explorer_execute_total", 0)
                }
            })
        
        with open(os.path.join(self.exp_dir, "edges.json"), "w") as f:
            json.dump(edges_array, f, indent=4)
    
    # =========================================================
    #  Outputs for Proposer & Solver
    # =========================================================

    def get_out_edges(self, node_id: str) -> list[dict]:
        """
        Output: provide Proposer with the edges that are reachable from the current node
        """
        out_edges = []
        for (u, v), stats in self.edges.items():
            if u == node_id:
                explorer_execute_total = stats["explorer_execute_total"]
                explorer_execute_success = stats["explorer_execute_success"]
                
                # Bayesian Smoothing
                # init prior = 0.5 (1/2), as total increases, the probability approaches the true value
                p_succ = (explorer_execute_success + 1) / (explorer_execute_total + 2)
                
                out_edges.append({
                    "target_node_id": v,
                    "explorer_execute_success_rate": p_succ,       # Not used by Proposer, just for information
                    "explorer_execute_total_attempts": explorer_execute_total,
                    "action_skill": stats.get("target_element", "")
                })
        return out_edges

    def add_to_explored(self, node: Node):
        if node in self.unexplored_nodes:
            self.unexplored_nodes.remove(node)
        
        if node not in self.explored_nodes:
            self.explored_nodes.append(node)
            node.visited = True
            node.update_save()

    def get_next_node(self) -> Node | None:
        """get frontier node to explore, if there is no interractive elements"""
        if len(self.unexplored_nodes) == 0:
            return None
        return self.unexplored_nodes[0]

    # =========================================================
    #  Loading
    # =========================================================

    @staticmethod
    def load(path: str, load_steps: bool=True, load_traces: bool=True, load_images: bool=True) -> Graph:
        logger.info(f"Loading graph from {path}")
        path = os.path.join(path, "graph")
        graph_info_path = os.path.join(path, "graph_info.json")

        with open(graph_info_path, "r") as f:
            graph_info = json.load(f)

        graph = Graph(
            root_url=graph_info.get("root_url"),
            exp_dir=os.path.dirname(path), 
            encoder_fn=graph_info.get("encoder_fn"),
            encoder_name=graph_info.get("encoder_name"),
            similarity_threshold=graph_info.get("similarity_threshold"),
            resume=True
        )
        graph.total_steps = graph_info.get("total_steps")
        
        # 1. load nodes
        node_dirs = glob.glob(os.path.join(path, "node_*"))
        for node_dir in node_dirs:
            node = Node.load(node_dir, load_steps=load_steps, load_traces=load_traces, load_images=load_images)
            graph.nodes[node.node_id] = node 
            if node.visited:
                graph.explored_nodes.append(node)
            else:
                graph.unexplored_nodes.append(node)
        
        # 2. load edges
        edges_path = os.path.join(path, "edges.json")
        with open(edges_path, "r") as f:
            edges_array = json.load(f)
            for edge in edges_array:
                key = (edge["source_node_id"], edge["target_node_id"])
                graph.edges[key] = {
                    "action": edge.get("action", ""),
                    "target_element": edge.get("target_element", ""),
                    "explorer_execute_success": edge.get("stats", {}).get("explorer_execute_success", 0),
                    "explorer_execute_total": edge.get("stats", {}).get("explorer_execute_total", 0)
                }
        
        if graph.explored_nodes:
            graph.current_node = graph.explored_nodes[-1]
        elif graph.nodes:
            sorted_ids = sorted(graph.nodes.keys(), key=lambda x: int(x.split('_')[-1]))
            graph.current_node = graph.nodes[sorted_ids[0]]

        logger.info(f"Loaded graph with {len(graph.nodes)} nodes, {graph.total_steps} total steps.")
        return graph