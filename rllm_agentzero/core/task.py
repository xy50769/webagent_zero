from .trajectory import Trajectory
from dataclasses import dataclass
import json
import logging
import os
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class Task:
    goal: str # global target
    instruction: str   # local target
    target_edge: dict 
    positive_trajs: list[Trajectory]
    negative_trajs: list[Trajectory]
    exp_dir: str
    misc: dict = None
    
    def __post_init__(self):
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
            os.makedirs(os.path.join(self.exp_dir, "positive_trajs"))
            os.makedirs(os.path.join(self.exp_dir, "negative_trajs"))
            
            task_info = {
                "goal": self.goal,
                "instruction": self.instruction,
                "target_edge": self.target_edge,
                "misc": self.misc,
            }
            with open(os.path.join(self.exp_dir, "task_info.json"), "w") as f:
                json.dump(task_info, f, indent=4)

    def is_feasible(self) -> bool:
        return len(self.positive_trajs) > 0
    
    def add_trajectory(self, traj: Trajectory, subdirectory=None):
        
        exp_dir = self.exp_dir
        
        if subdirectory is not None:
            exp_dir = os.path.join(exp_dir, subdirectory)
        
        if traj.success:
            traj_save_dir = os.path.join(exp_dir, "positive_trajs", f"{len(self.positive_trajs)}")
            self.positive_trajs.append(traj)
            logger.info(f"Saving positive trajectory to {traj_save_dir}")
            os.makedirs(traj_save_dir, exist_ok=True)
            traj.save(traj_save_dir)
        else:
            traj_save_dir = os.path.join(exp_dir, "negative_trajs", f"{len(self.negative_trajs)}")
            self.negative_trajs.append(traj)
            logger.info(f"Saving negative trajectory to {traj_save_dir}")
            os.makedirs(traj_save_dir, exist_ok=True)
            traj.save(traj_save_dir)

    @staticmethod
    def load(load_dir: str, load_steps: bool=True, load_images: bool=True):
        info_path = os.path.join(load_dir, "task_info.json")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Task info not found at {info_path}")

        with open(info_path, "r") as f:
            task_info = json.load(f)
        
        positive_trajs = []
        for i in range(len(os.listdir(os.path.join(load_dir, "positive_trajs")))):
            traj_load_dir = os.path.join(load_dir, "positive_trajs", f"{i}")
            positive_trajs.append(Trajectory.load(traj_load_dir, load_steps=load_steps, load_images=load_images))
        
        negative_trajs = []
        for i in range(len(os.listdir(os.path.join(load_dir, "negative_trajs")))):
            traj_load_dir = os.path.join(load_dir, "negative_trajs", f"{i}")
            negative_trajs.append(Trajectory.load(traj_load_dir, load_steps=load_steps, load_images=load_images))
        
        return Task(
            goal=task_info["goal"],
            instruction=task_info["instruction"], 
            target_edge=task_info,
            positive_trajs=positive_trajs,
            negative_trajs=negative_trajs,
            exp_dir=load_dir,
            misc=task_info["misc"]
        )
    
    @staticmethod
    def process_raw_goal(goal: str) -> tuple[str, list[str]]:
        """
        Process the raw goal string to remove any tags and return the cleaned goal and any tags.
        """
        tags = []
        tag_pattern = r'\[([A-Za-z0-9_-]+)\]'
        
        found_tags = re.findall(tag_pattern, goal)
        if found_tags:
            tags = found_tags
            goal = re.sub(r'\[[A-Za-z0-9_-]+\]', '', goal).strip()
        return goal, tags

    @staticmethod
    def from_goal(goal: str, instruction: str, target_edge: dict, exp_dir: str, misc: dict = None):
        if misc is None:
            misc = {}
        
        goal, tags = Task.process_raw_goal(goal)
    
        if tags:
            misc["tags"] = tags
        
        return Task(goal, instruction, target_edge, [], [], exp_dir, misc)