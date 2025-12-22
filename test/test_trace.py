'''
A test script to verify the Trace replay functionality in a browser environment.
This script creates a test trace, replays it in a browser environment, and checks if the
final state matches the expected outcome.'''

import logging
import gymnasium as gym
import browsergym.core
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rllm_agentzero.core.trace import Trace
from rllm_agentzero.core.trajectory import TrajectoryStep
from register_env import register_test_env
register_test_env()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def build_test_trace() -> list[TrajectoryStep]:
    steps = [
        # Step 0: type in the search box
        TrajectoryStep(
            action="page.locator('#sb_form_q').fill('RLLM')", 
            parsed_action=None, thought="Typing query", observation={}, misc={}
        ),
        # Step 1: press Enter
        TrajectoryStep(
            action="page.keyboard.press('Enter')", 
            parsed_action=None, thought="Pressing Enter", observation={}, misc={}
        )
    ]
    return steps

def test_trace_replay(steps) -> bool:
    start_url = "https://www.bing.com"
    target_url = "https://www.bing.com/search?q=RLLM" 

    # Test env (GenericTestTask) does not accept task kwargs like start_url in its constructor.
    # Don't pass task_kwargs here; GenericTestTask.setup() will navigate to a default page.
    env = gym.make("Test", headless=False)
    env.reset()
    env = env.unwrapped
    
    # creat sim trace
    trace = Trace.from_trajectory_steps(
        steps=steps, 
        start_url=start_url, 
        end_url="unknown_yet"
    )
    
    
    # set default state to something else
    env.page.goto("https://www.google.com")
    logger.info(f"Current URL before replay: {env.page.url}")

    # start replay
    success = trace.replay(env)
    
    # verify success
    logger.info("Verifying final state after replay...")
    env.page.wait_for_timeout(2000) 
    
    current_url = env.page.url
    logger.info(f"Current URL after replay: {current_url}")
    
    # simple check
    if "bing.com" in current_url and "RLLM" in current_url: 
        logger.info("Test PASSED: Browser reached expected state.")
        env.close()
        return True
    else:
        logger.error("Test FAILED: Browser did not reach expected state.")
        env.close()
        return False



if __name__ == "__main__":
    steps = build_test_trace()
    test_trace_replay(steps=steps)