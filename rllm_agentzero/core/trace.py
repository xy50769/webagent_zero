from .trajectory import TrajectoryStep
from dataclasses import dataclass
import json
import os
import glob
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def execute_python_code(code: str, page):
    """
    Executes the given Python code string in the context of the provided page.

    Args:
        code (str): The Python code to execute.
        page: The browser page object where the code will be executed.
    """
    def dummy_send_message_to_user(*args, **kwargs):
        pass
    def dummy_report_infeasible_instructions(*args, **kwargs):
        pass
    local_scope = {'page': page,
                   'send_message_to_user': dummy_send_message_to_user,
                   'report_infeasible_instructions': dummy_report_infeasible_instructions}
    try:
        exec(code, {}, local_scope)
    except Exception as e:
        logger.error(f"Error executing code: {e}")
        raise e

@dataclass
class Trace:
    """A trace of a sequence of steps"""

    steps: list[TrajectoryStep]
    start_url: str
    end_url: str
    misc: dict | None = None

    @classmethod
    def from_trajectory_steps(
        cls,
        steps: list[TrajectoryStep| str],
        start_url: str,
        end_url: str,
        misc: dict | None = None,
    ):
        return cls(steps, start_url, end_url, misc)
    
    def save(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        trace_info = {
            "start_url": self.start_url,
            "end_url": self.end_url,
            "misc": self.misc
        }
        
        with open(os.path.join(save_dir, "trace_info.json"), "w") as f:
            json.dump(trace_info, f, indent=4)
        
        for i, step in enumerate(self.steps):
            step_save_dir = os.path.join(save_dir, f"step_{i}")
            os.makedirs(step_save_dir, exist_ok=True)
            step.save(step_save_dir, keep_image_in_memory=True, save_image=False)

    def load(load_dir: str, load_steps: bool=True, load_images: bool=False):
        with open(os.path.join(load_dir, "trace_info.json"), "r") as f:
            trace_info = json.load(f)
        
        steps = []
        if load_steps:
            steps_dirs = sorted(glob.glob(os.path.join(load_dir, "step_*")),key=lambda x: int(os.path.basename(x).split("_")[-1]))
            for step_dir in steps_dirs:
                step = TrajectoryStep.load(step_dir, load_image=load_images)
                steps.append(step)
        else: 
            steps = os.listdir(load_dir)
        
        return Trace(steps, trace_info["start_url"], trace_info["end_url"], trace_info["misc"])
    
    def __len__(self):
        return len(self.steps)
    
    # New method to replay the trace
    def replay(self,env):
        logger.info(f"Replaying trace from {self.start_url} to {self.end_url} with {len(self.steps)} steps.")
        env.page.goto(self.start_url)
        env.page.wait_for_load_state("networkidle",timeout=5000)
        for i, step in enumerate(self.steps):
            action_code = step.action
            if not action_code:
                continue
            try:
                execute_python_code(action_code,env.page)
            except Exception as e:
                logger.error(f"Error executing action at step {i}: {e}")
                break

        logger.info(f"Finished replaying trace. Current URL: {env.page.url}")
        return True