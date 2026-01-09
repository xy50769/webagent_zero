import logging
import re
import json
from typing import Tuple, Dict, Optional

from .base_agent import AgentFactory, BaseAgent
from .prompt_builders.solver_prompt_builder import SolverPromptBuilder
from .trajectory_data import BrowserGymAgentStepData
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str,flatten_dom_to_str, prune_html
from openai import OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def extract_action_and_thought(raw_string):
    """Extract Thought and Action from raw string"""
    thought = ""
    action = ""
    
    try:
        raw_string = raw_string.strip()
        if raw_string.startswith('{'):
            try:
                data = json.loads(raw_string)
                thought = data.get("thought", "")
                action = data.get("action", "")
                if thought or action:
                    return thought, action
            except json.JSONDecodeError:
                pass
        
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
        
        t_match = re.search(r'Thought:\s*(.*?)(?=Action:|$)', raw_string, re.DOTALL | re.IGNORECASE)
        if t_match:
            thought = t_match.group(1).replace('\\"', '"').strip()

        a_match = re.search(r'Action:\s*(.*)', raw_string, re.DOTALL | re.IGNORECASE)
        if a_match:
            action = a_match.group(1).replace('\\"', '"').strip()
        elif not action and re.search(r'(click|type|scroll|goto|go_back)\(', raw_string):
            action = raw_string.strip()

    except Exception as e:
        logger.warning(f"Error parsing string: {e}")
        
    return thought, action

@AgentFactory.register
class SolverAgent(BaseAgent):
    """
    [RLLM Solver Agent]
    Assemble the consistency reward and efficiency penalty into a single agent.
    """

    def __init__(
            self,
            model_id: str | None = None,
            base_url: str | None = None,
            api_key: str | None = None,
            temperature: float = 1.0,
            char_limit: int = 16000,
            demo_mode: str = 'off',
            max_retries: int = 3,
            debug_mode: str = 'off',
            consistency_weight: float = 1.0,
            efficiency_penalty: float = 0.05,
            **kwargs
    ):
        super().__init__(model_id=model_id, temperature=temperature, char_limit=char_limit, demo_mode=demo_mode, **kwargs)
        
        self.model_id = model_id
        self.temperature = temperature
        self.char_limit = char_limit
        self.max_retries = max_retries
        self.demo_mode = demo_mode
        self.debug_mode = debug_mode
        self.consistency_weight = consistency_weight
        self.efficiency_penalty = efficiency_penalty

        self.client = OpenAI(base_url=base_url, api_key=api_key)

        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "infeas", "nav"],
            strict=False,
            multiaction=False,
            demo_mode=self.demo_mode
        )

        self.prompt_builder = SolverPromptBuilder(self.action_set)
        self.history: list[BrowserGymAgentStepData] = []

    def reset(self):
        self.history.clear()

    def _record_prompt(self, messages: list[dict], step_count: int):
        if self.debug_mode != "off":
            output = []
            output.append("\n" + "=" * 100)
            output.append(f"PROMPT FOR STEP {step_count}")
            output.append("=" * 100)
            
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                output.append(f"\n[{role.upper()}]")
                output.append(content)
            
            output.append("=" * 100)
            logger.info("\n".join(output))

    def _record_response(self, response_text: str, thought: str, action: str, step_count: int):
        if self.debug_mode != "off":
            output = []
            output.append("\n" + "=" * 100)
            output.append(f"LLM RESPONSE FOR STEP {step_count}")
            output.append("=" * 100)
            
            output.append("\n[FULL RAW RESPONSE]")
            output.append(response_text)
            
            output.append("\n" + "-" * 100)
            output.append("[PARSED OUTPUT]")
            output.append(f"Thought: {thought}")
            output.append(f"Action: {action}")
            output.append("=" * 100)
            
            logger.info("\n".join(output))

    def obs_preprocessor(self, obs: dict) -> dict:
        """
        Preprocess observation, extracting target information for reward calculation
        """
        
        return {
            "chat_messages": obs["chat_messages"],
            "screenshot": obs["screenshot"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "open_pages_urls": obs["open_pages_urls"],
            "open_pages_titles": obs["open_pages_titles"],
            "active_page_index": obs["active_page_index"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"], filter_visible_only=False, extra_properties=obs["extra_element_properties"]),
            "axtree_visible_only_txt": flatten_axtree_to_str(obs["axtree_object"], filter_visible_only=True, extra_properties=obs["extra_element_properties"]),
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
            "extra_element_properties": obs["extra_element_properties"],
            # New
            "target_info": obs.get("target_info", {}),  # Proposer target for Consistency Reward
            "step_count": obs.get("step_count", len(self.history)), # Current step count for Efficiency Penalty
        }
    
    def action_processor(self, action: str) -> str:
        action = action.strip()
        logger.info(f"[DEBUG] Raw action input to action_processor: {repr(action[:200])}")
        result = self.action_set.to_python_code(action)
        logger.info(f"[DEBUG] Parsed action output (first 200 chars): {repr(result[:200])}")
        return result

    def _extract_target_id_from_action(self, action_str: str) -> Optional[str]:
        if action_str and (match := re.search(r"['\"](\d+)['\"]", action_str)):
            return match.group(1)
        return None

    def _verify_target_consistency(self, action_id: str, target_desc: str, axtree_txt: str) -> float:
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

    def _calculate_inner_reward(self, raw_action: str, target_info: Dict, step_count: int, axtree_txt: str) -> Tuple[float, Dict]:
        """
        计算 Agent 内部的 Dense Reward (Consistency & Efficiency)
        注意：Outcome Reward (成败) 通常由环境在 Episode 结束时给出，这里只计算过程奖励。
        """
        # 1. Efficiency Penalty
        r_efficiency = -self.efficiency_penalty * step_count
        
        # 2. Consistency Reward
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

    def get_action(self, obs: dict, oracle_action: tuple[str, str] = None, **kwargs) -> tuple[str, dict]:
        
        current_step = BrowserGymAgentStepData(
            action=None,
            thought=None,
            axtree=obs["axtree_txt"],
            last_action_error=obs.get("last_action_error"),
            misc={}
        )

        action = "" 
        thought = ""
        raw_output = ""

        if oracle_action is None:
            for attempt in range(self.max_retries):
                try:
                    messages = self.prompt_builder.build_messages(
                        goal=obs["goal_object"][0]["text"],
                        current_step=current_step,
                        history=self.history,
                        char_limit=self.char_limit
                    )['prompt']

                    self._record_prompt(messages, len(self.history) + 1)

                    response = self.client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        temperature=self.temperature, 
                        max_tokens=2048
                    )
                    
                    raw_output = response.choices[0].message.content
                    thought, action = extract_action_and_thought(raw_output)
                    
                    self._record_response(raw_output, thought, action, len(self.history) + 1)
                    
                    if action and action.strip():
                        if response.usage:
                            current_step.misc["model_usage"] = response.usage.model_dump()
                        break 
                    else:
                        logger.warning(f"Attempt {attempt + 1}: Failed to extract action. Retrying...")
                
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} Inference Error: {e}")
                    if attempt == self.max_retries - 1:
                        thought = f"Error after {self.max_retries} retries: {e}"
                        action = "report_infeasible('Model failed to generate a valid action after multiple retries.')"
        
        else:
            action, thought = oracle_action
            raw_output = json.dumps({"thought": thought, "action": action})

        logger.info(f"Solver: {action}")

        # === BrowserGym env.step() will handle action parsing ===
        # No need to call action_processor here

        # === Calculate Inner Reward ===
        inner_reward, reward_details = self._calculate_inner_reward(
            raw_action=action,
            target_info=obs.get("target_info", {}),
            step_count=obs.get("step_count", 0),
            axtree_txt=obs["axtree_txt"]
        )

        # Update Step Data
        current_step.action = action 
        current_step.thought = thought
        current_step.misc.update({
            "thought": thought, 
            "raw_action": action,
            "raw_output": raw_output,
            "inner_reward": inner_reward,
            "reward_details": reward_details,
            "target_info_snapshot": obs.get("target_info", {})
        })
        
        self.history.append(current_step)

        return action, current_step.misc