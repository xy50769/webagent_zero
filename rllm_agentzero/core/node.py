from .task import Task
from .trace import Trace
from .trajectory import Trajectory
from dataclasses import dataclass, field
import json
import logging
import os
from PIL import Image
import glob
import re
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@dataclass
class Node:
    # ========== Required Fields (No Default Value) ==========
    node_id: str                    # unique identifier
    url: str
    embedding: list[float] | None  
    hint: str 
    exp_dir: str
    tasks: dict[str, Task]
    exploration_tasks: dict[str, Task]
    children: list[str]             # List of child node_ids
    traces: list[Trace]             

    # ========== Optional Fields (With Default Value) ==========
    depth: int = 0                  # minimal steps from root
    visit_count: int = 0            
    visited: bool = False
    action_history: list[str] = field(default_factory=list)  
    # unexplored elements: [{"bid": "42", "text": "Cart", "role": "button"}]
    interactive_elements: list[dict] = field(default_factory=list)  
    interactive_elements_visited: list[dict] = field(default_factory=list)  
    misc: dict | None = None

    def __post_init__(self):
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
            self.update_save()

    @staticmethod
    def load(load_dir: str, load_steps: bool=True, load_traces: bool=True, load_images: bool=True):
        info_path = os.path.join(load_dir, "node_info.json")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Node info file not found at {info_path}")
        
        with open(info_path, "r") as f:
            node_info = json.load(f)
        
        embedding = None
        embedding_path = os.path.join(load_dir, "embedding.npy")
        if os.path.exists(embedding_path):
            embedding = np.load(embedding_path).tolist()
        
        visited = node_info.get("visited", False)
        tasks = {}
        exploration_tasks = {}

        if visited and os.path.exists(os.path.join(load_dir, "tasks")):
            task_dirs = sorted(glob.glob(os.path.join(load_dir, "tasks", "task_*")), key=lambda x: int(os.path.basename(x).split("_")[-1]))
            for task_dir in task_dirs:
                task = Task.load(task_dir, load_steps=load_steps, load_images=load_images)
                tasks[task.goal] = task
        
        if visited and os.path.exists(os.path.join(load_dir, "exploration_tasks")):
            exploration_task_dirs = sorted(glob.glob(os.path.join(load_dir, "exploration_tasks", "task_*")), key=lambda x: int(os.path.basename(x).split("_")[-1]))
            for task_dir in exploration_task_dirs:
                task = Task.load(task_dir, load_steps=load_steps, load_images=load_images)
                exploration_tasks[task.goal] = task
        
        traces = []
        traces_dir = os.path.join(load_dir, "traces")
        if load_traces and os.path.exists(traces_dir):
            trace_dirs = sorted(
                glob.glob(os.path.join(traces_dir, "trace_*")),
                key=lambda x: int(os.path.basename(x).split("_")[-1])
            )
            for trace_dir in trace_dirs:
                trace = Trace.load(trace_dir, load_steps=load_steps, load_images=load_images)
                traces.append(trace)

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

    def update_save(self,save_screenshot=False, screenshot:np.ndarray|None = None):
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
        if self.embedding:
            np.save(os.path.join(self.exp_dir, "embedding.npy"), np.array(self.embedding))
        if save_screenshot and screenshot is not None:
            Image.fromarray(screenshot).save(os.path.join(self.exp_dir, "screenshot.png"))
    
    def record_action(self, action: str):
        """
            record_action records an action taken by the agent on the node.
            Find and move the element from unvisited to visited
        """
        match = re.search(r"['\"](.*?)['\"]", action)
        bid = match.group(1) if match else None

        self.action_history.append(action)
        self.visit_count += 1
        
        if bid:
            element = next((e for e in self.interactive_elements if e.get("bid") == bid), None)
            if element:
                self.interactive_elements.remove(element)
                self.interactive_elements_visited.append(element)
                logger.info(f"Moved element bid={bid} from unvisited to visited in node {self.node_id}")
            else:
                logger.warning(f"Element bid={bid} not found in interactive_elements of node {self.node_id} (may have been filtered out)")
                already_visited = any(e.get("bid") == bid for e in self.interactive_elements_visited)
                if not already_visited:
                    self.interactive_elements_visited.append({
                        "bid": bid,
                        "text": "",
                        "role": "unknown"
                    })
                    logger.info(f"Added placeholder for element bid={bid} to visited list")
        
        self.update_save()

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
        self.traces.append(trace)
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
    
    def register_elements(self, elements: list[dict]):
        """
            register_elements registers the interactive elements on the page (called when first visiting the node)
        """
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
        self.update_save()