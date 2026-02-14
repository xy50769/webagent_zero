"""
rllm_agentzero Solver Agent

Solver agent for task completion, adapted to rLLM's BaseAgent interface.
"""
import copy
import logging
import re
import json
from typing import Any, Dict, Optional, Tuple

from rllm.agents.agent import Action, Step, Trajectory

from .base_agent import AgentFactory, BaseAgent
from .prompt_builders.solver_prompt_builder import SolverPromptBuilder
from browsergym.core.action.highlevel import HighLevelActionSet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def extract_action_and_thought(raw_string: str) -> Tuple[str, str]:
    """
    Extract Thought and Action from raw LLM response string.
    
    Attempts multiple parsing strategies:
    1. Pure JSON format
    2. JSON embedded in text
    3. Thought: / Action: text format
    4. Direct action function call
    
    Returns:
        Tuple of (thought, action)
    """
    thought = ""
    action = ""
    
    try:
        raw_string = raw_string.strip()
        
        # Strategy 1: Pure JSON
        if raw_string.startswith('{'):
            try:
                data = json.loads(raw_string)
                thought = data.get("thought", "")
                action = data.get("action", "")
                if thought or action:
                    return thought, action
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: JSON embedded in text
        json_match = re.search(r'\{.*\}', raw_string, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                thought = data.get("thought", "")
                action = data.get("action", "")
                if thought or action:
                    return thought, action
            except json.JSONDecodeError as e:
                logger.debug(f"JSON decode failed: {e}, trying text format")
        
        # Strategy 3: Thought: / Action: format
        t_match = re.search(r'Thought:\s*(.*?)(?=Action:|$)', raw_string, re.DOTALL | re.IGNORECASE)
        if t_match:
            thought = t_match.group(1).replace('\\"', '"').strip()

        a_match = re.search(r'Action:\s*(.*)', raw_string, re.DOTALL | re.IGNORECASE)
        if a_match:
            action = a_match.group(1).replace('\\"', '"').strip()
        
        # Strategy 4: Direct function call
        elif not action and re.search(r'(click|type|scroll|goto|go_back)\(', raw_string):
            action = raw_string.strip()

    except Exception as e:
        logger.warning(f"Error parsing string: {e}")
        
    return thought, action


@AgentFactory.register
class SolverAgent(BaseAgent):
    """
    [RLLM Solver Agent]
    
    Task completion agent with consistency reward and efficiency penalty.
    Adapted to rLLM's BaseAgent interface with update_from_env/update_from_model pattern.
    """

    def __init__(
            self,
            model_id: str | None = None,
            temperature: float = 1.0,
            char_limit: int = 16000,
            demo_mode: str = 'off',
            max_retries: int = 3,
            debug_mode: str = 'off',
            consistency_weight: float = 1.0,
            efficiency_penalty: float = 0.05,
            **kwargs
    ):
        """
        Initialize SolverAgent.
        
        Args:
            model_id: Model identifier (used by rLLM's engine for inference)
            temperature: Sampling temperature
            char_limit: Character limit for prompts
            demo_mode: Demo mode setting for action set
            max_retries: Maximum retry attempts for action extraction
            debug_mode: Debug logging mode
            consistency_weight: Weight for consistency reward
            efficiency_penalty: Step penalty coefficient
        """
        super().__init__(
            model_id=model_id, 
            temperature=temperature, 
            char_limit=char_limit, 
            demo_mode=demo_mode, 
            **kwargs
        )
        
        self.model_id = model_id
        self.temperature = temperature
        self.char_limit = char_limit
        self.max_retries = max_retries
        self.demo_mode = demo_mode
        self.debug_mode = debug_mode
        self.consistency_weight = consistency_weight
        self.efficiency_penalty = efficiency_penalty

        # Action set for BrowserGym
        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "infeas", "nav"],
            strict=False,
            multiaction=False,
            demo_mode=self.demo_mode
        )

        # Prompt builder
        self.prompt_builder = SolverPromptBuilder(self.action_set)
        
        # Goal storage
        self._goal = ""
        
    def reset(self):
        """Reset the agent's state for a new episode."""
        super().reset()
        self._goal = ""
        self._init_system_message()
    
    def _init_system_message(self):
        """Initialize system message with action set description."""
        # prompt_builder.system_message() 返回 dict，需要提取 text 字段
        system_msg_obj = self.prompt_builder.system_message()
        system_prompt = system_msg_obj.get("text", "") if isinstance(system_msg_obj, dict) else str(system_msg_obj)
        if system_prompt:
            self.messages = [{"role": "system", "content": system_prompt}]
    
    def _format_observation_as_messages(self, obs: Any) -> list[dict]:
        """
        Format observation into chat messages for the model.
        
        Uses SolverPromptBuilder for consistent formatting.
        """
        messages = []
        
        if isinstance(obs, dict):
            # Extract goal if available
            if "goal_object" in obs:
                goals = obs["goal_object"]
                if isinstance(goals, list) and goals:
                    self._goal = goals[0].get("text", "") if isinstance(goals[0], dict) else str(goals[0])
            
            # Build user message using prompt builder logic
            content_parts = []
            
            # Add goal
            if self._goal:
                content_parts.append(f"## Goal\n{self._goal}")
            
            # Add current page info
            if "url" in obs:
                content_parts.append(f"## Current URL\n{obs['url']}")
            
            # Add page content (axtree)
            if "axtree_txt" in obs:
                axtree = obs["axtree_txt"]
                if len(axtree) > self.char_limit:
                    axtree = axtree[:self.char_limit] + "\n... [truncated]"
                content_parts.append(f"## Page Content (Accessibility Tree)\n{axtree}")
            
            # Add error info if any
            if obs.get("last_action_error"):
                content_parts.append(f"## Last Action Error\n{obs['last_action_error']}")
            
            if content_parts:
                messages.append({
                    "role": "user",
                    "content": "\n\n".join(content_parts)
                })
        
        return messages
    
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Update agent state after environment step.
        
        Args:
            observation: Environment observation
            reward: Reward from environment
            done: Whether episode is done
            info: Additional info from environment
        """
        # Preprocess observation
        processed_obs = self.obs_preprocessor(observation) if isinstance(observation, dict) else observation
        
        # Store current observation
        self.current_observation = processed_obs
        
        # Format and add observation messages
        obs_messages = self._format_observation_as_messages(processed_obs)
        self.messages.extend(obs_messages)
        
        # Update last step with reward/done
        if self._trajectory.steps:
            self._trajectory.steps[-1].reward = reward
            self._trajectory.steps[-1].done = done
            self._trajectory.steps[-1].info.update(info)
    
    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Update agent state after model generates a response.
        
        This method:
        1. Parses the model response to extract thought and action
        2. Updates message history with assistant response
        3. Creates a trajectory step
        4. Calculates inner reward if applicable
        
        Args:
            response: Raw response string from the model
            
        Returns:
            Action object containing the parsed action
        """
        # Parse response
        thought, action_str = extract_action_and_thought(response)
        
        if self.debug_mode != "off":
            self._log_response(response, thought, action_str)
        
        # Add assistant message to history
        assistant_message = {"role": "assistant", "content": response}
        self.messages.append(assistant_message)
        
        # Calculate inner reward
        inner_reward = 0.0
        reward_details = {}
        if isinstance(self.current_observation, dict):
            inner_reward, reward_details = self._calculate_inner_reward(
                raw_action=action_str,
                target_info=self.current_observation.get("target_info", {}),
                step_count=len(self._trajectory.steps),
                axtree_txt=self.current_observation.get("axtree_txt", "")
            )
        
        # Extract ModelOutput from kwargs if present (provided by rLLM rollout engine via workflow)
        model_output = kwargs.get("model_output")
        
        # Create trajectory step
        new_step = Step(
            chat_completions=copy.deepcopy(self.messages),
            observation=self.current_observation,
            thought=thought,
            action=action_str,
            model_response=response,
            model_output=model_output,  # Store full model output (needed for PPO)
            prompt_ids=model_output.prompt_ids if model_output else [],
            response_ids=model_output.completion_ids if model_output else [],
            logprobs=model_output.logprobs if model_output else [],
            info={
                "raw_action": action_str,
                "inner_reward": inner_reward,
                "reward_details": reward_details,
            }
        )
        self._trajectory.steps.append(new_step)
        
        logger.info(f"Solver: {action_str}")
        
        return Action(action=action_str)
    
    def _log_response(self, response: str, thought: str, action: str):
        """Log model response for debugging."""
        output = []
        output.append("\n" + "=" * 100)
        output.append(f"LLM RESPONSE FOR STEP {len(self._trajectory.steps) + 1}")
        output.append("=" * 100)
        output.append("\n[FULL RAW RESPONSE]")
        output.append(response)
        output.append("\n" + "-" * 100)
        output.append("[PARSED OUTPUT]")
        output.append(f"Thought: {thought}")
        output.append(f"Action: {action}")
        output.append("=" * 100)
        logger.info("\n".join(output))
    
    def obs_preprocessor(self, obs: dict) -> dict:
        """
        Preprocess observation, extracting relevant fields.
        """
        from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
        
        processed = {
            "goal_object": obs.get("goal_object", []),
            "last_action_error": obs.get("last_action_error", ""),
            "open_pages_urls": obs.get("open_pages_urls", []),
            "target_info": obs.get("target_info", {}),
            "step_count": obs.get("step_count", len(self._trajectory.steps)),
        }
        
        # Process axtree if available
        if "axtree_object" in obs:
            processed["axtree_txt"] = flatten_axtree_to_str(
                obs["axtree_object"], 
                filter_visible_only=False, 
                extra_properties=obs.get("extra_element_properties", {})
            )
        elif "axtree_txt" in obs:
            processed["axtree_txt"] = obs["axtree_txt"]
        
        # Extract URL
        if "open_pages_urls" in obs and obs["open_pages_urls"]:
            active_idx = obs.get("active_page_index", 0)
            # 确保 active_idx 是整数
            if hasattr(active_idx, 'item'):
                active_idx = active_idx.item()
            else:
                active_idx = int(active_idx)
            if active_idx < len(obs["open_pages_urls"]):
                processed["url"] = obs["open_pages_urls"][active_idx]
        
        return processed
    
    def action_processor(self, action: str) -> str:
        """
        Process action string into executable Python code.
        """
        action = action.strip()
        return self.action_set.to_python_code(action)
    
    def _extract_target_id_from_action(self, action_str: str) -> Optional[str]:
        """Extract element ID from action string."""
        if action_str and (match := re.search(r"['\"](\d+)['\"]", action_str)):
            return match.group(1)
        return None
    
    def _verify_target_consistency(self, action_id: str, target_desc: str, axtree_txt: str) -> float:
        """Verify if action targets the correct element based on description."""
        if not action_id or not target_desc:
            return 0.0
            
        pattern = re.compile(fr"\[{re.escape(action_id)}\]\s*(.+)")
        match = pattern.search(axtree_txt)
        
        if not match:
            return 0.0
            
        element_line = match.group(1).lower()
        target_keywords = [w.lower() for w in target_desc.split() if len(w) > 2]
        
        if not target_keywords:
            return 0.0

        matches = sum(1 for word in target_keywords if word in element_line)
        match_rate = matches / len(target_keywords) if target_keywords else 0
        
        if match_rate > 0.6:
            return 1.0
        elif match_rate > 0.3:
            return 0.5
            
        return 0.0
    
    def _calculate_inner_reward(
        self, 
        raw_action: str, 
        target_info: Dict, 
        step_count: int, 
        axtree_txt: str
    ) -> Tuple[float, Dict]:
        """
        Calculate inner reward (consistency & efficiency).
        
        Returns:
            Tuple of (total_reward, details_dict)
        """
        # Efficiency Penalty
        r_efficiency = -self.efficiency_penalty * step_count
        
        # Consistency Reward
        r_consistency = 0.0
        target_element_desc = target_info.get("target_element", "")
        action_id = self._extract_target_id_from_action(raw_action)
        
        if target_element_desc and action_id:
            match_score = self._verify_target_consistency(action_id, target_element_desc, axtree_txt)
            r_consistency = self.consistency_weight * match_score
        
        total = r_consistency + r_efficiency
        return total, {
            "r_consistency": r_consistency,
            "r_efficiency": r_efficiency,
            "action_id": action_id,
            "target_match_score": r_consistency / (self.consistency_weight + 1e-6)
        }
    
    # === Backward compatibility method (deprecated) ===
    def get_action(self, obs: dict, oracle_action: tuple[str, str] = None, **kwargs) -> tuple[str, dict]:
        """
        Legacy method for backward compatibility.
        
        DEPRECATED: Use update_from_env() and update_from_model() instead.
        This method is kept for testing and gradual migration.
        """
        logger.warning("get_action() is deprecated. Use update_from_env/update_from_model pattern.")
        
        # Simulate env update
        self.update_from_env(obs, reward=0.0, done=False, info={})
        
        if oracle_action is not None:
            action, thought = oracle_action
            raw_output = json.dumps({"thought": thought, "action": action})
            result = self.update_from_model(raw_output)
            return result.action, {"thought": thought, "raw_action": action}
        
        # For non-oracle case, return chat_completions for external LLM call
        # The actual LLM call should be handled by AgentExecutionEngine
        return "", {"messages": self.chat_completions, "needs_model_response": True}