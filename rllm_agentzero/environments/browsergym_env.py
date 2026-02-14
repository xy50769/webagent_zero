"""
WebAgentZero Environment

rLLM-compatible BrowserGym environment wrapper for web agent training.
"""
import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from browsergym.utils.obs import flatten_axtree_to_str

from rllm.environments.base.base_env import BaseEnv

logger = logging.getLogger(__name__)


class WebAgentZeroEnv(BaseEnv):
    """
    rLLM-compatible environment wrapper for BrowserGym.
    
    Provides a standardized interface for web agent training with rLLM.
    Handles observation preprocessing and action execution.
    """
    
    def __init__(
        self,
        task: Dict[str, Any] = None,
        env_id: str = "browsergym/openended",
        headless: bool = True,
        timeout: int = 30000,
        **kwargs
    ):
        """
        Initialize the environment.
        
        Args:
            task: Task configuration dict with keys:
                - url: Starting URL
                - goal: Task goal description
                - env_id: Optional override for gym environment ID
            env_id: Gymnasium environment ID
            headless: Whether to run browser in headless mode
            timeout: Browser timeout in milliseconds
            **kwargs: Additional arguments passed to gymnasium
        """
        self.task = task or {}
        self.env_id = task.get("env_id", env_id) if task else env_id
        self.headless = headless
        self.timeout = timeout
        self.extra_kwargs = kwargs
        
        self._env: Optional[gym.Env] = None
        self._current_obs: Dict = {}
        self._step_count: int = 0
        
    def _create_env(self) -> gym.Env:
        """Create the underlying BrowserGym environment."""
        task_kwargs = {}
        
        if self.task:
            if "url" in self.task:
                task_kwargs["start_url"] = self.task["url"]
            if "goal" in self.task:
                task_kwargs["goal"] = self.task["goal"]
        
        env_kwargs = {
            "headless": self.headless,
            "timeout": self.timeout,
            **self.extra_kwargs,
        }
        
        if task_kwargs:
            env_kwargs["task_kwargs"] = task_kwargs
        
        return gym.make(self.env_id, **env_kwargs)
    
    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """
        Reset the environment.
        
        Args:
            **kwargs: 额外参数（如 task）会被忽略，兼容 rLLM 的调用方式
        
        Returns:
            Tuple of (observation, info)
        """
        # 忽略 rLLM 传入的 task 参数
        _ = kwargs.get('task', None)
        
        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:
                logger.warning(f"Error closing previous env: {e}")
        
        self._env = self._create_env()
        self._step_count = 0
        
        obs, info = self._env.reset()
        self._current_obs = self._preprocess_obs(obs)
        
        return self._current_obs, info
    
    def step(self, action: Any) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action to execute (string action code)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        self._step_count += 1
        
        # Handle Action object from rLLM
        if hasattr(action, 'action'):
            action = action.action
        
        # Execute action
        try:
            obs, reward, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            obs = self._current_obs
            reward = 0.0
            done = True
            info = {"error": str(e)}
        
        self._current_obs = self._preprocess_obs(obs)
        self._current_obs["step_count"] = self._step_count
        
        return self._current_obs, reward, done, info
    
    def close(self):
        """Close the environment and cleanup resources."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:
                logger.warning(f"Error closing environment: {e}")
            self._env = None
    
    def _preprocess_obs(self, obs: Dict) -> Dict:
        """
        Preprocess raw observation into agent-friendly format.
        
        Args:
            obs: Raw observation from BrowserGym
            
        Returns:
            Preprocessed observation dict
        """
        processed = {
            "goal_object": obs.get("goal_object", []),
            "last_action_error": obs.get("last_action_error", ""),
            "open_pages_urls": obs.get("open_pages_urls", []),
            "open_pages_titles": obs.get("open_pages_titles", []),
            "active_page_index": obs.get("active_page_index", 0),
            "screenshot": obs.get("screenshot"),
            "step_count": self._step_count,
        }
        
        # Process accessibility tree
        if "axtree_object" in obs:
            extra_props = obs.get("extra_element_properties", {})
            processed["axtree_txt"] = flatten_axtree_to_str(
                obs["axtree_object"],
                filter_visible_only=False,
                extra_properties=extra_props
            )
            
            # Extract interactive elements
            processed["interactive_elements"] = self._extract_interactive_elements(
                processed["axtree_txt"],
                extra_props
            )
        elif "axtree_txt" in obs:
            processed["axtree_txt"] = obs["axtree_txt"]
            processed["interactive_elements"] = obs.get("interactive_elements", [])
        
        # Extract current URL
        if processed["open_pages_urls"]:
            active_idx = processed["active_page_index"]
            # 确保 active_idx 是整数（可能从 numpy 数组返回）
            if hasattr(active_idx, 'item'):
                active_idx = active_idx.item()
            else:
                active_idx = int(active_idx)
            if active_idx < len(processed["open_pages_urls"]):
                processed["url"] = processed["open_pages_urls"][active_idx]
        
        return processed
    
    def _extract_interactive_elements(
        self, 
        axtree_txt: str, 
        extra_properties: Dict
    ) -> list:
        """
        Extract interactive elements from accessibility tree.
        
        Args:
            axtree_txt: Accessibility tree as text
            extra_properties: Additional element properties
            
        Returns:
            List of interactive element dicts
        """
        elements = []
        
        # Import element extraction utility if available
        try:
            from rllm_agentzero.core.element_utils import extract_interactive_elements
            elements = extract_interactive_elements(axtree_txt, extra_properties)
        except ImportError:
            # Fallback: basic extraction from extra_properties
            for bid, props in extra_properties.items():
                if props.get("clickable") or props.get("editable"):
                    elements.append({
                        "bid": bid,
                        "role": props.get("role", ""),
                        "text": props.get("text", ""),
                        "visible": props.get("visible", True),
                        "clickable": props.get("clickable", False),
                    })
        
        return elements
    
    @staticmethod
    def from_dict(info: Dict) -> "WebAgentZeroEnv":
        """
        Create environment from a task dictionary.
        
        This is used by rLLM's AgentExecutionEngine for batch environment creation.
        
        Args:
            info: Task info dictionary with:
                - url: Starting URL
                - goal: Task goal
                - env_id: Optional environment ID
                - headless: Optional headless setting
                
        Returns:
            WebAgentZeroEnv instance
        """
        return WebAgentZeroEnv(
            task=info,
            headless=info.get("headless", True),
            timeout=info.get("timeout", 30000),
        )
    
    @staticmethod
    def is_multithread_safe() -> bool:
        """
        Indicate if this environment is safe for multithreading.
        
        BrowserGym environments use separate browser processes,
        so they can run in parallel.
        """
        return True
    
    @property
    def current_observation(self) -> Dict:
        """Get the current observation."""
        return self._current_obs
    
    @property
    def step_count(self) -> int:
        """Get the current step count."""
        return self._step_count
