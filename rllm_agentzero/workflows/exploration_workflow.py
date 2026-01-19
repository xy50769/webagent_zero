"""
ExplorationWorkflow

rLLM Workflow implementation for Graph-based web exploration.
Integrates the Graph world model with rLLM's training infrastructure.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

from rllm.agents.agent import Trajectory, Step, Episode
from rllm.engine.rollout.rollout_engine import RolloutEngine, ModelOutput
from rllm.workflows.workflow import Workflow, TerminationReason, TerminationEvent
# 使用 rLLM 的 BrowserGymEnv（subprocess 隔离，已修复兼容性）
from rllm.environments.browsergym.browsergym import BrowserGymEnv
from browsergym.utils.obs import flatten_axtree_to_str

from rllm_agentzero.agents.explorer_agent import ExplorerAgent
from rllm_agentzero.core.graph import Graph

logger = logging.getLogger(__name__)


def _process_browsergym_obs(obs) -> Dict:
    """
    Process raw BrowserGym observation tuple into dict format.
    
    rLLM's BrowserGymEnv returns raw BrowserGym obs which contains
    axtree_object. We need to convert it to axtree_txt.
    """
    # Handle tuple return from reset (obs, info)
    if isinstance(obs, tuple):
        raw_obs = obs[0]
        info = obs[1] if len(obs) > 1 else {}
    else:
        raw_obs = obs
        info = {}
    
    result = {}
    
    # Process AXTree
    if isinstance(raw_obs, dict):
        if "axtree_object" in raw_obs and raw_obs["axtree_object"] is not None:
            result["axtree_txt"] = flatten_axtree_to_str(raw_obs["axtree_object"])
        elif "axtree_txt" in raw_obs:
            result["axtree_txt"] = raw_obs["axtree_txt"]
        else:
            result["axtree_txt"] = ""
        
        # Copy other fields
        for key in ["last_action_error", "open_pages_urls", "active_page_index", "url", "goal"]:
            if key in raw_obs:
                result[key] = raw_obs[key]
        
        # Handle goal from goal_object
        if "goal_object" in raw_obs and raw_obs["goal_object"] is not None:
            goal_obj = raw_obs["goal_object"]
            if isinstance(goal_obj, list) and len(goal_obj) > 0:
                first = goal_obj[0]
                if isinstance(first, dict) and "text" in first:
                    result["goal"] = first["text"]
                else:
                    result["goal"] = str(first)
            else:
                result["goal"] = str(goal_obj)
    else:
        result["axtree_txt"] = ""
    
    return result, info


class ExplorationWorkflow(Workflow):
    """
    Workflow for Graph-based web exploration.
    
    Integrates the Graph world model from rllm_agentzero with rLLM's
    training infrastructure. Manages the exploration loop including:
    
    1. Environment interaction
    2. Graph state matching and updates  
    3. Exploration reward calculation
    4. Trajectory collection for training
    
    Usage with AgentTrainer:
        trainer = AgentTrainer(
            workflow_class=ExplorationWorkflow,
            workflow_args={
                "encoder_fn": my_encoder,
                "max_steps": 20,
                ...
            },
            backend="verl",
        )
    """
    
    def __init__(
        self,
        rollout_engine: RolloutEngine = None,
        executor: ThreadPoolExecutor = None,
        # Graph configuration
        encoder_fn: Callable[[dict], list[float]] = None,
        encoder_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.95,
        # Exploration configuration
        max_steps: int = 20,
        max_nodes: int = 50,
        # Agent configuration
        agent_cls: type = None,
        agent_args: Dict = None,
        # Environment configuration  
        env_cls: type = None,
        env_args: Dict = None,
        # Reward configuration
        gamma: float = 0.99,
        reward_bonus_coeff: float = 1.0,  # 默认使用探索奖励
        **kwargs
    ):
        """
        Initialize ExplorationWorkflow.
        
        Args:
            rollout_engine: rLLM rollout engine for model inference
            executor: Thread pool executor
            encoder_fn: Function to encode observations to embeddings
            encoder_name: Name of encoder model
            similarity_threshold: Threshold for node matching
            max_steps: Maximum steps per episode
            max_nodes: Maximum nodes in graph
            agent_cls: Agent class (default: ExplorerAgent)
            agent_args: Agent initialization arguments
            env_cls: Environment class (default: WebAgentZeroEnv)
            env_args: Environment initialization arguments
            gamma: Discount factor for rewards
            reward_bonus_coeff: Coefficient for exploration reward bonus
        """
        super().__init__(
            rollout_engine=rollout_engine,
            executor=executor,
            gamma=gamma,
            reward_bonus_coeff=reward_bonus_coeff,
            **kwargs
        )
        
        # Graph configuration
        self.encoder_fn = encoder_fn
        self.encoder_name = encoder_name
        self.similarity_threshold = similarity_threshold
        
        # Exploration configuration
        self.max_steps = max_steps
        self.max_nodes = max_nodes
        
        # Agent configuration
        self.agent_cls = agent_cls or ExplorerAgent
        self.agent_args = agent_args or {}
        
        # Environment configuration - 使用 rLLM 的 BrowserGymEnv（subprocess 隔离）
        self.env_cls = env_cls or BrowserGymEnv
        self.env_args = env_args or {}
        
        # State (per-run, not shared)
        self.graph: Optional[Graph] = None
        self.agent: Optional[ExplorerAgent] = None
        self.env: Optional[BrowserGymEnv] = None
        
    def reset(self, task: Dict = None, uid: str = None):
        """
        Reset the workflow for a new episode.
        
        Args:
            task: Task configuration dict
            uid: Unique identifier for this episode
        """
        # 存储 uid 属性，rLLM 的 postprocess_episode 需要它
        self.uid = uid
        self.task = task
        super().reset(task, uid)
        
        # Initialize environment
        # BrowserGymEnv expects task with start_url and goal (not env_id, headless, etc.)
        env_args = {**self.env_args}
        if task:
            # Only pass the fields that BrowserGymEnv/OpenEndedTask expects
            env_args["task"] = {
                "start_url": task.get("url", "https://www.google.com"),
                "goal": task.get("goal", "Explore the website"),
            }
        self.env = self.env_cls(**env_args)
        
        # Initialize agent
        self.agent = self.agent_cls(**self.agent_args)
        self.agent.reset()
        
        # Initialize graph (world model)
        if self.encoder_fn is not None:
            exp_dir = task.get("exp_dir", "./exploration_output") if task else "./exploration_output"
            self.graph = Graph(
                root_url=task.get("url", "") if task else "",
                exp_dir=exp_dir,
                encoder_fn=self.encoder_fn,
                encoder_name=self.encoder_name,
                similarity_threshold=self.similarity_threshold,
                resume=False
            )
        else:
            self.graph = None
            logger.warning("No encoder_fn provided, Graph world model disabled")
    
    def _initialize_graph_root(self, obs: Dict):
        """
        Initialize the graph with the first observation as root node.
        
        Fix for Issue #2: Ensure source_node is valid on first step.
        """
        if self.graph is None:
            return
        
        # Create root node if not exists
        if not self.graph.nodes:
            embedding = self.encoder_fn(obs)
            from rllm_agentzero.core.element_utils import extract_interactive_elements
            
            node_id = "node_0"
            from rllm_agentzero.core.node import Node
            import os
            
            node_exp_dir = os.path.join(self.graph.exp_dir, node_id)
            root_node = Node(
                node_id=node_id,
                embedding=embedding,
                url=obs.get("url", ""),
                hint="Root",
                tasks={},
                exploration_tasks={},
                children=[],
                depth=0,
                visit_count=1,
                traces=[],
                visited=True,
                exp_dir=node_exp_dir
            )
            
            elements = extract_interactive_elements(
                obs.get("axtree_txt", ""),
                obs.get("extra_element_properties", {})
            )
            root_node.register_elements(elements)
            
            self.graph.nodes[node_id] = root_node
            self.graph.explored_nodes.append(root_node)
            self.graph.current_node = root_node
            
            logger.info(f"Initialized root node with {len(elements)} interactive elements")
    
    async def run_async(self, task: Dict, uid: str, **kwargs) -> Episode:
        """
        Execute the exploration workflow on a task (async version).
        
        This is the primary method for use with async rollout engines.
        
        Args:
            task: Task configuration with url, goal, etc.
            uid: Unique identifier for this episode
            
        Returns:
            Episode containing trajectories for training
        """
        # Reset for new episode
        self.reset(task, uid)
        
        # Initial environment reset
        obs, info = self.env.reset()
        done = False
        step_count = 0
        
        # Fix #2: Initialize graph root with first observation
        self._initialize_graph_root(obs)
        
        logger.info(f"Starting exploration: {task.get('url', 'unknown')}")
        
        try:
            while not done and step_count < self.max_steps:
                # Check node limit
                if self.graph and len(self.graph.nodes) >= self.max_nodes:
                    logger.info(f"Reached max nodes limit: {self.max_nodes}")
                    raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)
                
                # Store current node for reward calculation (before step)
                source_node = self.graph.current_node if self.graph else None
                
                # Prepare exploration context for agent
                exploration_info = self._build_exploration_context()
                
                # Update agent with observation
                self.agent.update_from_env(obs, reward=0.0, done=False, info=exploration_info)
                
                # Get model response via rollout engine
                messages = self.agent.chat_completions
                response = await self._get_model_response(messages)
                
                # Update agent with model response
                action = self.agent.update_from_model(response)
                
                # Execute action in environment
                obs_prev = obs
                obs, env_reward, done, info = self.env.step(action.action)
                step_count += 1
                
                # Update graph world model with error handling (Fix #3)
                target_node = None
                if self.graph:
                    try:
                        action_success = not obs.get("last_action_error", "")
                        target_node = self.graph.process_transition(
                            obs_t=obs_prev,
                            action=action.action,
                            obs_t1=obs,
                            explorer_execute_success=action_success,
                            thought=self.agent._trajectory.steps[-1].thought if self.agent._trajectory.steps else ""
                        )
                    except Exception as e:
                        logger.warning(f"Graph process_transition failed: {e}")
                        target_node = None
                
                # Calculate exploration reward with null check (Fix #3)
                explore_reward = 0.0
                if source_node is not None and target_node is not None and self.graph:
                    try:
                        reward_output = ExplorerAgent.calculate_exploration_reward(
                            source_node=source_node,
                            target_node=target_node,
                            action_str=action.action,
                            graph=self.graph
                        )
                        explore_reward = reward_output.reward
                    except Exception as e:
                        logger.warning(f"Reward calculation failed: {e}")
                        explore_reward = 0.0
                
                # Fix #5: Apply reward_bonus_coeff
                total_reward = env_reward + self.reward_bonus_coeff * explore_reward
                
                # Update step reward
                if self.agent._trajectory.steps:
                    self.agent._trajectory.steps[-1].reward = total_reward
                    self.agent._trajectory.steps[-1].done = done
                
                logger.info(f"Step {step_count}: reward={total_reward:.3f}, done={done}")
            
            # Fix #4: Handle terminal observation (process final state)
            if done and self.agent._trajectory.steps:
                # The last step already has done=True set above
                logger.info("Episode completed, final step recorded")
            
            # Commit trajectory
            self.commit(name="explorer", agent=self.agent, reset=False)
            
        except TerminationEvent as e:
            logger.info(f"Terminated: {e.reason}")
            self.commit(name="explorer", agent=self.agent, reset=False)
        
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            self.commit(name="explorer", agent=self.agent, reset=False)
        
        finally:
            # Cleanup
            if self.env:
                self.env.close()
        
        # Collect and return episode
        episode = self.collect_trajectories()
        episode.task = task
        episode.id = uid
        
        return episode
    
    async def run(self, task: Dict, uid: str, **kwargs) -> Episode:
        """
        Execute the exploration workflow on a task (async version).
        
        注意：虽然这是 async 函数，但内部直接调用同步代码。
        这是因为 Playwright Sync API 无法在 run_in_executor 的子线程中正常工作。
        每个 Ray Worker 是独立进程，阻塞式运行不会影响其他 Worker。
        
        Args:
            task: Task configuration with url, goal, etc.
            uid: Unique identifier for this episode
            
        Returns:
            Episode containing trajectories for training
        """
        # 存储 uid 属性
        self.uid = uid
        self.task = task
        
        try:
            # Reset for new episode (直接同步调用)
            self.reset(task, uid)
            
            # Initial environment reset (直接同步调用)
            raw_obs = self.env.reset()
            obs, info = _process_browsergym_obs(raw_obs)
            done = False
            step_count = 0
            
            # Initialize graph root with first observation (直接同步调用)
            self._initialize_graph_root(obs)
            
            logger.info(f"Starting exploration: {task.get('url', 'unknown') if task else 'unknown'}")
            
            while not done and step_count < self.max_steps:
                # Check node limit
                if self.graph and len(self.graph.nodes) >= self.max_nodes:
                    logger.info(f"Reached max nodes limit: {self.max_nodes}")
                    break
                
                # Store current node for reward calculation
                source_node = self.graph.current_node if self.graph else None
                
                # Prepare exploration context for agent
                exploration_info = self._build_exploration_context()
                
                # Update agent with observation
                self.agent.update_from_env(obs, reward=0.0, done=False, info=exploration_info)
                
                # Get model response - 这里仍然用 await，因为 rollout_engine 是 async 的
                messages = self.agent.chat_completions
                response_obj = await self._get_model_response_async(messages)
                
                # Check for model failure BEFORE processing
                # If model failed, terminate episode immediately to avoid empty data in PPO
                finish_reason = getattr(response_obj, 'finish_reason', None)
                if finish_reason == 'error':
                    logger.warning("Model call failed, terminating episode early (finish_reason=error)")
                    break
                
                # Extract string for agent logic
                response_str = ""
                if hasattr(response_obj, 'text') and response_obj.text:
                    response_str = response_obj.text
                elif hasattr(response_obj, 'content') and response_obj.content:
                    response_str = response_obj.content
                else:
                    response_str = str(response_obj) # Fallback
                
                # Update agent with model response AND full model output
                action = self.agent.update_from_model(response_str, model_output=response_obj)
                
                # Execute action in environment (直接同步调用)
                obs_prev = obs
                raw_result = self.env.step(action.action)
                # raw_result is (obs, reward, done, info) from BrowserGymEnv
                raw_obs, env_reward, done, extra_info = raw_result
                obs, _ = _process_browsergym_obs(raw_obs)
                step_count += 1
                
                # Update graph world model (直接同步调用)
                target_node = None
                if self.graph:
                    try:
                        action_success = not obs.get("last_action_error", "")
                        target_node = self.graph.process_transition(
                            obs_t=obs_prev,
                            action=action.action,
                            obs_t1=obs,
                            explorer_execute_success=action_success,
                            thought=self.agent._trajectory.steps[-1].thought if self.agent._trajectory.steps else ""
                        )
                    except Exception as e:
                        logger.warning(f"Graph process_transition failed: {e}")
                        target_node = None
                
                # Calculate exploration reward
                explore_reward = 0.0
                if source_node is not None and target_node is not None and self.graph:
                    try:
                        reward_output = ExplorerAgent.calculate_exploration_reward(
                            source_node=source_node,
                            target_node=target_node,
                            action_str=action.action,
                            graph=self.graph
                        )
                        explore_reward = reward_output.reward
                    except Exception as e:
                        logger.warning(f"Reward calculation failed: {e}")
                
                # Apply reward_bonus_coeff
                total_reward = env_reward + self.reward_bonus_coeff * explore_reward
                
                # Update step reward
                if self.agent._trajectory.steps:
                    self.agent._trajectory.steps[-1].reward = total_reward
                    self.agent._trajectory.steps[-1].done = done
                
                logger.info(f"Step {step_count}: reward={total_reward:.3f}, done={done}")
            
            # Commit trajectory
            self.commit(name="explorer", agent=self.agent, reset=False)
            
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            import traceback
            traceback.print_exc()
            self.commit(name="explorer", agent=self.agent, reset=False)
        
        finally:
            # Cleanup (直接同步调用)
            if self.env:
                try:
                    self.env.close()
                except:
                    pass
        
        # Collect and return episode
        episode = self.collect_trajectories()
        episode.task = task
        episode.id = uid
        
        return episode
    
    async def _get_model_response_async(self, messages: list) -> ModelOutput:
        """
        使用 rollout_engine 获取模型响应（async 版本）。
        
        返回 ModelOutput 对象以便获取 logprobs 和 token ids。
        """
        if self.rollout_engine is None:
            logger.warning("Rollout engine not initialized, returning placeholder response")
            # Return placeholder ModelOutput
            return ModelOutput(
                text='{"thought": "Testing mode - no model", "action": "report_infeasible(\'No model available\')"}',
                content='{"thought": "Testing mode - no model", "action": "report_infeasible(\'No model available\')"}',
                finish_reason="stop",
                prompt_ids=[],
                completion_ids=[],
                logprobs=[],
                prompt_logprobs=[],
                prompt_length=0,
                completion_length=0
            )
        
        try:
            # 调用 rollout_engine 的 async 方法
            result = await self.rollout_engine.get_model_response(messages=messages)
            return result
            
        except Exception as e:
            logger.warning(f"Model call failed: {e}")
            # Return error ModelOutput
            return ModelOutput(
                text='{"thought": "Model call failed", "action": "report_infeasible(\'Model call failed\')"}',
                content='{"thought": "Model call failed", "action": "report_infeasible(\'Model call failed\')"}',
                finish_reason="error",
                prompt_ids=[],
                completion_ids=[],
                logprobs=[],
                prompt_logprobs=[],
                prompt_length=0,
                completion_length=0
            )
    
    
    async def _get_model_response(self, messages: list) -> str:
        """
        Get model response using rollout engine.
        
        Args:
            messages: Chat completion messages
            
        Returns:
            Model response string
        """
        if self.rollout_engine is None:
            # Fallback: return placeholder for testing
            logger.warning("Rollout engine not initialized, returning placeholder response")
            return '{"thought": "Testing mode - no model", "action": "report_infeasible(\'No model available\')"}'
        
        # Use rollout engine for inference
        result = await self.rollout_engine.generate(
            messages=messages,
            max_tokens=2048,
        )
        
        return result.text if hasattr(result, 'text') else str(result)
    
    def _build_exploration_context(self) -> Dict:
        """
        Build exploration context for the agent.
        
        Returns:
            Dict with frontier_info, unvisited_elements, visited_elements
        """
        context = {}
        
        if self.graph and self.graph.current_node:
            node = self.graph.current_node
            
            # Frontier info
            if not node.interactive_elements:
                next_node = self.graph.get_next_node()
                if next_node:
                    context["frontier_info"] = {
                        "node_id": next_node.node_id,
                        "url": getattr(next_node, 'url', ''),
                    }
            
            # Element masks
            context["unvisited_elements"] = node.interactive_elements or []
            context["visited_elements"] = [
                elem.get("bid", "") 
                for elem in getattr(node, 'interactive_elements_visited', [])
            ]
            
            # Graph reference for reward calculation
            context["node"] = node
            context["graph"] = self.graph
        
        return context
    
    def compute_trajectory_reward(self, trajectory: Trajectory) -> float:
        """
        Compute trajectory-level reward.
        
        For exploration, we sum the step rewards which include
        both environment rewards and exploration bonuses.
        """
        return sum(step.reward for step in trajectory.steps)
    
    @staticmethod
    def is_multithread_safe() -> bool:
        """
        rLLM AgentWorkflowEngine 要求返回 True。
        
        每个任务使用独立的 Workflow 实例，所以是线程安全的。
        """
        return True
