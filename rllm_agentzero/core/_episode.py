from ._evaluator import Evaluator
from .graph import Graph
from .node import Node
from .trajectory import Trajectory, TrajectoryStep
from .task import Task
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from ..agents.base_agent import BaseAgent

from browsergym.core.env import BrowserEnv
from browsergym.experiments.loop import _send_chat_info

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=20),
)
def reset_env_to_node(
        env: BrowserEnv,
        node: Node,
        agent: BaseAgent,
        goal: str,
):
    """
    Reset the environment to a given node using Trace Replay or Direct Navigation.
    """
    logger.info(f"Resetting environment to node: {node.url}...")
    
    # 1. 基础重置 (在 Wrapper 上调用是安全的，也是推荐的)
    env.reset()
    
    # 2. [关键修复] 获取底层环境以访问 page, chat, action_mapping
    browser_env = env.unwrapped if hasattr(env, "unwrapped") else env
    
    # 绑定动作解析器
    browser_env.action_mapping = agent.action_processor
    
    # 3. 导航逻辑
    # 优先使用 Trace Replay
    if node.traces and len(node.traces) > 0:
        best_trace = node.traces[0]
        logger.info(f"Replaying trace ({len(best_trace)} steps) to reach node: {node.url}...")
        try:
            # Trace replay 需要直接操作 browser_env (因为它需要 page 对象)
            # 注意: trace.replay 内部如果用了 env.page，传入 browser_env 最稳妥
            success = best_trace.replay(browser_env) 
            if not success:
                logger.warning(f"Failed to replay trace. Fallback to direct navigation.")
                browser_env.page.goto(node.url)
        except Exception as e:
            logger.error(f"Error replaying trace: {e}. Fallback to direct navigation.")
            browser_env.page.goto(node.url)
    else:
        # 兜底逻辑：直接跳转
        logger.info(f"No trace found. Navigating directly.")
        try:
            browser_env.page.goto(node.url)
            browser_env.page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception as e:
            logger.error(f"Error navigating to {node.url}: {e}")
        
    # 4. 设置 Goal 和 Chat
    browser_env.goal_object = [{"type": "text", "text": goal}]
    browser_env.chat.add_message(role="user", msg=goal)
    
    # 获取 Observation (使用底层 env 也可以，或者 Wrapper 的 env.step 之后也会返回)
    obs = browser_env._get_obs()
    return agent.obs_preprocessor(obs)


def has_new_assistant_message(env: BrowserEnv):
    """Check if there is a new assistant message (indicating task end)."""
    # [关键修复] 解包 env
    browser_env = env.unwrapped if hasattr(env, "unwrapped") else env
    
    if not browser_env.chat.messages:
        return False
    last_msg = browser_env.chat.messages[-1]
    if last_msg["role"] == "assistant" or last_msg["role"] == "infeasible":
        return True
    return False


def get_action(
    env: BrowserEnv,
    agent: BaseAgent,
    obs: dict,
    traj: Trajectory,
    oracle_action: str = None
) -> tuple[str, dict]:
    """Get action from agent and record it."""
    
    action, action_extras = agent.get_action(obs, oracle_action=oracle_action)
    
    thought = action_extras.get("thought", None)
    parsed_action = action_extras.get("parsed_action", None)

    if thought and "think" not in action_extras:
        action_extras["think"] = thought

    logger.info(f"Agent chose action: \n{action}")
    
    # 记录步骤到 Trajectory
    traj.add_step(
        action, 
        parsed_action, 
        thought, 
        obs, 
        {'model_usage': action_extras.get("model_usage", None), 'agent_config': agent.get_config()}
    )

    # 同步到 Chat 界面
    # [关键修复] 解包 env
    browser_env = env.unwrapped if hasattr(env, "unwrapped") else env
    _send_chat_info(browser_env.chat, action, action_extras)
    
    return action, action_extras


def perform_env_step(
    env: BrowserEnv,
    agent: BaseAgent,
    action: str,
    target_info: dict = None,
    step_count: int = 0,
) -> tuple:
    """Execute the action in the environment."""
    # Step 依然要在最外层 env 上调用，以保持 Gym Wrapper 的功能 (如 TimeLimit 计数)
    obs, reward, terminated, truncated, env_info = env.step(action)
    
    # 注入 target_info 和 step_count，确保在 obs_preprocessor 中可用
    if target_info is not None:
        obs["target_info"] = target_info
    obs["step_count"] = step_count
    
    obs = agent.obs_preprocessor(obs)
    return obs, reward, terminated, truncated, env_info


def run_episode(
    goal: str,
    node: Node,
    env: BrowserEnv,
    agent: BaseAgent,
    evaluator: Evaluator,
    graph: Graph,
    max_steps: int,
    task: Task = None,
) -> Trajectory:
    """
    Run a complete RLLM episode: Reset -> Loop(Action->Step) -> Evaluate.
    """
    logger.info(f"Running episode for goal: {goal}, at node {node.url}...")

    # 1. 环境重置 (内部已处理 unwrapped)
    obs = reset_env_to_node(
        env=env,
        node=node,
        agent=agent,
        goal=goal,
    )
    
    # 提取 Target Info
    target_info = {}
    if task and task.target_edge:
        target_info = {
            "target_element": task.target_edge.get("target_element", ""),
            "target_node_id": task.target_edge.get("target_node_id", ""),
            "action_skill": task.target_edge.get("action_skill", ""),
        }
    
    agent.reset()
    traj = Trajectory.from_goal(goal, agent.get_config())
    
    num_steps = 0
    done = False

    while not done and num_steps < max_steps:
        logger.info(f"Step {num_steps} for goal {goal}.")
        num_steps += 1

        # 注入 target_info 和 step_count 到 obs
        obs["target_info"] = target_info
        obs["step_count"] = num_steps

        # 2. 获取动作
        action, action_extras = get_action(
            env=env,
            agent=agent,
            obs=obs,
            traj=traj
        )

        if action is None:
            logger.info("Agent returned None action. Ending episode.")
            break

        # 3. 执行动作 (传递 target_info 和 step_count)
        obs, reward, terminated, truncated, env_info = perform_env_step(
            env=env,
            agent=agent,
            action=action,
            target_info=target_info,
            step_count=num_steps,
        )
        
        # 4. 检查是否结束 (BrowserGym 特有逻辑)
        # 这里传入 env 即可，函数内部会 handle unwrapped
        if has_new_assistant_message(env):
            logger.info("New assistant message received. Task claimed done.")
            terminated = True
            
            # 5. 触发评估 (Evaluator)
            if evaluator:
                logger.info("Evaluating episode with GPT-4V...")
                try:
                    evaluator.evaluate(traj)
                    logger.info(f"Episode evaluated. Success: {traj.success}, Reward: {traj.reward}")
                except Exception as e:
                    logger.error(f"Evaluation failed: {e}")
                    traj.success = False
                    traj.reward = 0.0

        done = terminated or truncated

    # 6. 整理最终结果
    traj.extract_response(env.unwrapped if hasattr(env, "unwrapped") else env)
    traj.final_state = TrajectoryStep(
        action=None,
        parsed_action=None,
        thought=None,
        observation=obs, 
        misc=None,
    )

    return traj