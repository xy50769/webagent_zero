"""
rllm_agentzero Agent Base Classes

Adapts the original agent interfaces to be compatible with rLLM's BaseAgent pattern.
"""
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# Import rLLM's base classes
from rllm.agents.agent import (
    Action,
    BaseAgent as RLLMBaseAgent,
    Step,
    Trajectory,
)


@runtime_checkable
class AgentProtocol(Protocol):
    """
    Protocol for an agent (for type checking).
    """
    
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        ...
    
    def update_from_model(self, response: str, **kwargs) -> Action:
        ...
    
    def reset(self):
        ...
    
    @property
    def chat_completions(self) -> list[dict[str, str]]:
        ...
    
    @property
    def trajectory(self) -> Trajectory:
        ...


class BaseAgent(RLLMBaseAgent):
    """
    Base agent class for rllm_agentzero, extending rLLM's BaseAgent.
    
    Provides common functionality for both SolverAgent and ExplorerAgent.
    Implements the rLLM agent interface: update_from_env(), update_from_model().
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the agent.
        
        Args:
            **kwargs: Keyword arguments for the agent configuration.
        """
        self.config = {
            "name": self.__class__.__name__,
            **kwargs
        }
        
        # Internal state
        self._trajectory = Trajectory(name=self.__class__.__name__)
        self.messages: list[dict[str, Any]] = []
        self.current_observation: dict = {}
        self.history: list[Step] = []
        
    def reset(self):
        """
        Reset the agent's state for a new episode.
        """
        self._trajectory = Trajectory(name=self.__class__.__name__)
        self.messages = []
        self.current_observation = {}
        self.history = []
        self._init_system_message()
    
    def _init_system_message(self):
        """
        Initialize the system message. Override in subclasses.
        """
        pass
    
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Updates the agent's internal state after an environment step.
        
        Args:
            observation: The observation from the environment.
            reward: The reward received after taking the action.
            done: Whether the episode has ended.
            info: Additional metadata from the environment.
        """
        # Store the current observation
        self.current_observation = observation
        
        # Format observation as messages
        obs_messages = self._format_observation_as_messages(observation)
        self.messages.extend(obs_messages)
        
        # Update the last step in trajectory with reward/done info
        if self._trajectory.steps:
            self._trajectory.steps[-1].reward = reward
            self._trajectory.steps[-1].done = done
            self._trajectory.steps[-1].info.update(info)
    
    def _format_observation_as_messages(self, obs: Any) -> list[dict]:
        """
        Format observation into chat messages. Override in subclasses for custom formatting.
        
        Args:
            obs: The observation to format.
            
        Returns:
            List of message dictionaries.
        """
        messages = []
        if isinstance(obs, dict):
            # Build user message from observation components
            content_parts = []
            
            if "goal_object" in obs:
                goals = obs["goal_object"]
                if isinstance(goals, list) and goals:
                    goal_text = goals[0].get("text", "") if isinstance(goals[0], dict) else str(goals[0])
                    content_parts.append(f"Goal: {goal_text}")
            
            if "axtree_txt" in obs:
                content_parts.append(f"Page Content:\n{obs['axtree_txt']}")
            
            if "url" in obs:
                content_parts.append(f"Current URL: {obs['url']}")
            
            if "last_action_error" in obs and obs["last_action_error"]:
                content_parts.append(f"Last Action Error: {obs['last_action_error']}")
            
            if content_parts:
                messages.append({
                    "role": "user",
                    "content": "\n\n".join(content_parts)
                })
        elif isinstance(obs, str):
            messages.append({"role": "user", "content": obs})
        
        return messages
    
    @abstractmethod
    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's internal state after the model generates a response.
        
        Args:
            response: The response from the model.
            
        Returns:
            Action: The action to execute in the environment.
        """
        raise NotImplementedError("Subclasses must implement update_from_model")
    
    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """
        Returns the current message history for the model.
        """
        return self.messages
    
    @property
    def trajectory(self) -> Trajectory:
        """
        Returns the trajectory recorded so far.
        """
        return self._trajectory
    
    def get_config(self) -> dict:
        """
        Get the agent's configuration.
        
        Returns:
            dict: The agent's configuration.
        """
        return self.config
    
    def obs_preprocessor(self, obs: dict) -> dict:
        """
        Preprocess the observation before it is used.
        
        Args:
            obs: The observation from the environment.
            
        Returns:
            dict: The preprocessed observation.
        """
        return obs
    
    def action_processor(self, action: str) -> str:
        """
        Process the action before it is passed to the environment.
        
        Args:
            action: The action to process.
            
        Returns:
            str: The processed action.
        """
        return action
    
    @classmethod
    def create_agent(cls, *args, **kwargs):
        """
        Factory method to create an agent instance.
        """
        return cls(*args, **kwargs)


# Agent Factory Registry
AGENT_FACTORY_REGISTRY: dict[str, type[BaseAgent]] = {}


class AgentFactory:
    """
    Factory class for creating registered agents.
    """
    
    @staticmethod
    def create_agent(name: str, *args, **kwargs) -> BaseAgent:
        """
        Create an agent instance by name.
        
        Args:
            name: The registered name of the agent.
            *args: Positional arguments for the agent.
            **kwargs: Keyword arguments for the agent.
            
        Returns:
            BaseAgent: An instance of the requested agent.
        """
        if name not in AGENT_FACTORY_REGISTRY:
            raise ValueError(f"Unknown agent: {name}")
        return AGENT_FACTORY_REGISTRY[name].create_agent(*args, **kwargs)
    
    @staticmethod
    def register(cls=None, aliases: str | tuple[str] = tuple()):
        """
        Register an agent class with the factory.
        
        Can be used as a decorator:
            @AgentFactory.register
            class MyAgent(BaseAgent):
                ...
        
        Or with aliases:
            @AgentFactory.register(aliases=("my_agent", "myagent"))
            class MyAgent(BaseAgent):
                ...
        """
        def decorator(agent_cls):
            AGENT_FACTORY_REGISTRY[agent_cls.__name__] = agent_cls
            
            if isinstance(aliases, str):
                AGENT_FACTORY_REGISTRY[aliases] = agent_cls
            else:
                for name in aliases:
                    AGENT_FACTORY_REGISTRY[name] = agent_cls
            
            return agent_cls
        
        if cls is not None:
            # Called without arguments: @AgentFactory.register
            return decorator(cls)
        else:
            # Called with arguments: @AgentFactory.register(aliases=...)
            return decorator
