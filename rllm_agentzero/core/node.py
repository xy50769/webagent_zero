from .task import Task
from .trace import Trace
from .trajectory import Trajectory
from dataclasses import dataclass, field
import json
import logging
import os
import shutil
import glob
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def delete_folder_contents(folder_path: str):
    """Delete all contents of a folder without deleting the folder itself."""
    if not os.path.exists(folder_path):
        logger.warning(f"Folder {folder_path} does not exist. Nothing to delete.")
        return
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                logger.info(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                logger.info(f"Deleted directory and its contents: {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete {file_path}. Reason: {e}")

@dataclass
class Node:
    # ========== 必需字段 (无默认值) ==========
    node_id: str                    # 唯一标识符 (例如 "node_0")
    url: str
    embedding: list[float] | None   # Vector Embedding for Semantic Matching
    hint: str 
    exp_dir: str
    tasks: dict[str, Task]
    exploration_tasks: dict[str, Task]
    children: list[str]             # List of child node_ids
    traces: list[Trace]             # 到达该节点的路径集合（存储在 traces/ 文件夹）

    # ========== 可选字段 (有默认值) ==========
    depth: int = 0                  # 从 Root 到此的最短跳数
    visit_count: int = 0            # 访问次数
    visited: bool = False
    action_history: list[str] = field(default_factory=list)  # 已尝试过的动作列表
    misc: dict | None = None
    interactive_elements: list[dict] = field(default_factory=list)  # 未探索的元素: [{"id": "42", "text": "Cart", "type": "button"}]
    interactive_elements_visited: list[dict] = field(default_factory=list)  # 已探索的元素

    def __post_init__(self):
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
            self.update_save(save_traces=True, save_info=True)

    @staticmethod
    def load(load_dir: str, load_steps: bool=True, load_traces: bool=True, load_images: bool=True):
        info_path = os.path.join(load_dir, "node_info.json")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Node info file not found at {info_path}")
        
        with open(info_path, "r") as f:
            node_info = json.load(f)
        
        # 加载 embedding
        embedding = None
        embedding_path = os.path.join(load_dir, "embedding.npy")
        if os.path.exists(embedding_path):
            embedding = np.load(embedding_path).tolist()
        
        visited = node_info.get("visited", False)
        tasks = {}
        exploration_tasks = {}

        # load micro-tasks
        if visited and os.path.exists(os.path.join(load_dir, "tasks")):
            task_dirs = sorted(glob.glob(os.path.join(load_dir, "tasks", "task_*")), key=lambda x: int(os.path.basename(x).split("_")[-1]))
            for task_dir in task_dirs:
                try:
                    task = Task.load(task_dir, load_steps=load_steps, load_images=load_images)
                    tasks[task.goal] = task
                except Exception as e:
                    logger.error(f"Error loading task from {task_dir}: {e}")
        
        # load exploration micro-tasks
        if visited and os.path.exists(os.path.join(load_dir, "exploration_tasks")):
            exploration_task_dirs = sorted(glob.glob(os.path.join(load_dir, "exploration_tasks", "task_*")), key=lambda x: int(os.path.basename(x).split("_")[-1]))
            for task_dir in exploration_task_dirs:
                try:
                    task = Task.load(task_dir, load_steps=load_steps, load_images=load_images)
                    exploration_tasks[task.goal] = task
                except Exception as e:
                    logger.error(f"Error loading exploration task from {task_dir}: {e}")
        
        # load traces（从 traces/ 文件夹）
        traces = []
        traces_dir = os.path.join(load_dir, "traces")
        if load_traces and os.path.exists(traces_dir):
            trace_dirs = sorted(
                glob.glob(os.path.join(traces_dir, "trace_*")),
                key=lambda x: int(os.path.basename(x).split("_")[-1])
            )
            for trace_dir in trace_dirs:
                try:
                    trace = Trace.load(trace_dir, load_steps=load_steps, load_images=load_images)
                    traces.append(trace)
                except Exception as e:
                    logger.error(f"Error loading trace from {trace_dir}: {e}")

        # 读取新格式字段
        return Node(
            node_id=node_info["node_id"],
            embedding=embedding,
            url=node_info.get("url", ""),
            hint=node_info.get("hint", ""),
            tasks=tasks,
            exploration_tasks=exploration_tasks,
            children=node_info.get("children", []),
            depth=node_info.get("depth", 0),
            visit_count=node_info.get("visit_count", 0),
            traces=traces,
            visited=visited,
            exp_dir=load_dir,
            action_history=node_info.get("action_history", []),
            interactive_elements=node_info.get("interactive_elements", []),
            interactive_elements_visited=node_info.get("interactive_elements_visited", []),
            misc=node_info.get("misc", None)
        )

    def update_save(self, save_traces=False, save_info=True, save_screenshot=False, screenshot=None):
        if save_info:
            node_info = {
                "node_id": self.node_id,
                "url": self.url,
                
                # Graph structure
                "depth": self.depth,
                "visit_count": self.visit_count,
                "children": self.children,
                
                # Interactive space
                "interactive_elements": self.interactive_elements,
                "interactive_elements_visited": self.interactive_elements_visited,
                "action_history": self.action_history,
                
                # Metadata
                "hint": self.hint,
                "visited": self.visited,
                "misc": self.misc
            }
            with open(os.path.join(self.exp_dir, "node_info.json"), "w") as f:
                json.dump(node_info, f, indent=4)
        
        # 保存 embedding 到单独的 .npy 文件
        if self.embedding:
            np.save(os.path.join(self.exp_dir, "embedding.npy"), np.array(self.embedding))
        
        # 保存截图
        if save_screenshot and screenshot is not None:
            from PIL import Image
            if isinstance(screenshot, np.ndarray):
                img = Image.fromarray(screenshot)
                img.save(os.path.join(self.exp_dir, "screenshot.png"))
        
        if save_traces:
            # 使用 traces/ 文件夹
            traces_dir = os.path.join(self.exp_dir, "traces")
            os.makedirs(traces_dir, exist_ok=True)
            delete_folder_contents(traces_dir)
            
            # 保存所有 traces
            for i, trace in enumerate(self.traces):
                trace_save_dir = os.path.join(traces_dir, f"trace_{i}")
                os.makedirs(trace_save_dir, exist_ok=True)
                trace.save(trace_save_dir)
    
    def record_action(self, action: str):
        """记录已尝试的动作"""
        self.action_history.append(action)
        self.visit_count += 1
        self.update_save(save_traces=False, save_info=True)

    def add_task(self, goal: str, instruction: str = "", target_edge: dict = None, task_misc: dict = None) -> Task:
        if target_edge is None:
            target_edge = {}
        task_dir = os.path.join(self.exp_dir, "tasks", f"task_{len(self.tasks)}")
        processed_goal, _ = Task.process_raw_goal(goal)
        if processed_goal not in self.tasks:
            self.tasks[processed_goal] = Task.from_goal(goal=goal, instruction=instruction, target_edge=target_edge, exp_dir=task_dir, misc=task_misc)
            logger.info(f"Added new task '{processed_goal}'")
        return self.tasks[processed_goal]
    
    def add_trajectory(self, traj: Trajectory, task_misc: dict = None):
        task = self.add_task(traj.goal, task_misc=task_misc)
        task.add_trajectory(traj)

    def add_trace(self, trace: Trace):
        """添加一个 trace 到节点的 traces 列表"""
        self.traces.append(trace)
        # 立即保存
        trace_save_dir = os.path.join(self.exp_dir, "traces", f"trace_{len(self.traces) - 1}")
        os.makedirs(trace_save_dir, exist_ok=True)
        trace.save(trace_save_dir)

    def add_exploration_task(self, goal: str, instruction: str = "", target_edge: dict = None, task_misc: dict = None) -> Task:
        if target_edge is None:
            target_edge = {}
        task_dir = os.path.join(self.exp_dir, "exploration_tasks", f"task_{len(self.exploration_tasks)}")
        processed_goal, _ = Task.process_raw_goal(goal)
        if processed_goal not in self.exploration_tasks:
            self.exploration_tasks[processed_goal] = Task.from_goal(goal=goal, instruction=instruction, target_edge=target_edge, exp_dir=task_dir, misc=task_misc)
            logger.info(f"Added new exploration task '{processed_goal}'")
        return self.exploration_tasks[processed_goal]
    
    def add_exploration_traj(self, traj: Trajectory):
        task = self.add_exploration_task(traj.goal)
        task.add_trajectory(traj)
    
    # ========== Element-level Tracking Methods ==========
    
    def register_elements(self, elements: list[dict]):
        """
        注册页面上的可交互元素（首次访问节点时调用）
        
        Args:
            elements: 元素列表，已经过滤为可见且可点击的元素
                      格式 [{"bid": "123", "role": "button", "text": "...", "visible": True, "clickable": True}]
        """
        # 使用验证好的过滤条件
        filtered_elements = [
            {
                "bid": item.get("bid", ""),
                "text": item.get("text", ""),
                "role": item.get("role", "unknown")
            }
            for item in elements 
            if item.get('visible') and item.get('clickable')
        ]
        
        self.interactive_elements = filtered_elements
        logger.info(f"Registered {len(filtered_elements)} interactive elements for node {self.node_id}")
        self.update_save(save_traces=False, save_info=True)