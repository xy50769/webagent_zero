from .task import Task
from .trace import Trace
from .trajectory import Trajectory
from dataclasses import dataclass, field
import json
import logging
import os
import shutil
import glob
from collections import Counter

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
    # ========== Node Attributes ==========
    node_id: str                    # 唯一标识符 (例如 "node_0")
    url: str
    embedding: list[float] | None   # Vector Embedding for Semantic Matching
    hint: str 
    exp_dir: str

    # ========== data collections ==========
    tasks: dict[str, Task]
    exploration_tasks: dict[str, Task]
    children: list[str]             # List of child node_ids
    prefixes: list[Trace]

    # ========== State Management ==========
    visited: bool = False
    action_history: Counter = field(default_factory=Counter)
    misc: dict | None = None

    def __post_init__(self):
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
            self.update_save(save_prefix=True, save_info=True)

    @staticmethod
    def load(load_dir: str, load_steps: bool=True, load_prefix: bool=True, load_images: bool=True):
        info_path = os.path.join(load_dir, "node_info.json")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Node info file not found at {info_path}")
        
        with open(info_path, "r") as f:
            node_info = json.load(f)
        
        visited = node_info.get("visited", False)
        tasks = {}
        exploration_tasks = {}
        action_history = Counter(node_info.get("action_history", {}))

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
        
        # load prefixes
        prefixes = []
        if load_prefix and os.path.exists(os.path.join(load_dir, "prefixes")):
            prefix_dirs = sorted(glob.glob(os.path.join(load_dir, "prefixes", "prefix_*")), key=lambda x: int(os.path.basename(x).split("_")[-1]))
            for prefix_dir in prefix_dirs:
                try:
                    prefix = Trace.load(prefix_dir, load_steps=load_steps, load_images=load_images)
                    prefixes.append(prefix)
                except Exception as e:
                    logger.error(f"Error loading prefix trace from {prefix_dir}: {e}")

        # 直接读取必需字段，不再做兼容性检查
        return Node(
            node_id=node_info["node_id"],         # 必须存在，否则报错
            embedding=node_info.get("embedding"), # 允许为 None (如果尚未计算)
            url=node_info.get("url", ""),
            hint=node_info.get("hint", ""),
            tasks=tasks,
            exploration_tasks=exploration_tasks,
            children=node_info.get("children", []),
            prefixes=prefixes,
            visited=visited,
            exp_dir=load_dir,
            action_history=action_history,
            misc=node_info.get("misc", None)
        )

    def update_save(self, save_prefix=False, save_info=True):
        if save_info:
            node_info = {
                "node_id": self.node_id,
                "embedding": self.embedding,
                "url": self.url,
                "hint": self.hint,
                "children": self.children,
                "visited": self.visited,
                "action_history": dict(self.action_history),
                "misc": self.misc
            }
            with open(os.path.join(self.exp_dir, "node_info.json"), "w") as f:
                json.dump(node_info, f, indent=4)
        
        if save_prefix:
            prefix_save_dir = os.path.join(self.exp_dir, "prefixes")
            os.makedirs(prefix_save_dir, exist_ok=True)
            delete_folder_contents(prefix_save_dir)
            for i, trace in enumerate(self.prefixes):
                trace_save_dir = os.path.join(prefix_save_dir, f"prefix_{i}")
                os.makedirs(trace_save_dir, exist_ok=True)
                trace.save(trace_save_dir)
    
    def record_action(self, action: str):
        self.action_history[action] += 1
        self.update_save(save_prefix=False, save_info=True)

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

    def add_prefix(self, prefix: Trace):
        prefix_save_dir = os.path.join(self.exp_dir, "prefixes")
        os.makedirs(prefix_save_dir, exist_ok=True)
        existing_prefixes = glob.glob(os.path.join(prefix_save_dir, "prefix_*"))
        next_num = len(existing_prefixes)
        trace_save_dir = os.path.join(prefix_save_dir, f"prefix_{next_num}")
        os.makedirs(trace_save_dir, exist_ok=True)
        prefix.save(trace_save_dir)
        self.prefixes.append(prefix)

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