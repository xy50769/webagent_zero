import logging
import re
import json
from .base_agent import AgentFactory, BaseAgent
from .prompt_builders.solver_prompt_builder import SolverPromptBuilder
from .prompt_builders import flatten_messages # 引用我们在 __init__.py 里写的辅助函数
from .server.llm_engine import LLMEngine
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str, prune_html, flatten_dom_to_str
from .trajectory_data import BrowserGymAgentStepData

logger = logging.getLogger(__name__)

def extract_action_and_thought(raw_string):
    """
    Extract thought and action from model output.
    Supports JSON format and text format (Thought: ... Action: ...).
    Returns: (thought, action)
    """
    thought = None
    action = None
    
    try:
        try:
            json_match = re.search(r'\{.*\}', raw_string, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                thought = data.get("thought", "")
                action = data.get("action", "")
                if action:
                    return thought, action
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse error, attempting to fix: {e}")
            try:
                json_match = re.search(r'\{.*', raw_string, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    json_str = _fix_json_string(json_str)
                    data = json.loads(json_str)
                    thought = data.get("thought", "")
                    action = data.get("action", "")
                    if action:
                        logger.info("Successfully fixed and parsed malformed JSON")
                        return thought, action
            except Exception:
                pass
        
        thought_match = re.search(r'"thought"\s*:\s*"(.*?)"(?=\s*[,}])', raw_string, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1)
            thought = thought.replace('\\\\', '\\').replace('\\"', '"')
            
        action_patterns = [
            r'"action"\s*:\s*"(.*?)"(?=\s*[,}])',
            r'"action"\s*:\s*"([^"]*?)(?:"|,|})',
            r'"action"\s*:\s*"([^"]+)',
            r"'action'\s*:\s*'([^']*?)(?:'|,|})",
            r"'action'\s*:\s*'([^']+)",
        ]
        
        for pattern in action_patterns:
            action_match = re.search(pattern, raw_string, re.DOTALL)
            if action_match:
                action = action_match.group(1)
                action = action.replace('\\\\', '\\').replace('\\"', '"')
                action = action.rstrip('\\').rstrip()
                
                if action and not action.endswith(')'):
                    func_match = re.match(r'(\w+)\s*\(\s*["\']?([^"\')\]]*)["\']?', action)
                    if func_match:
                        func_name = func_match.group(1)
                        param = func_match.group(2)
                        if param and len(param) > 0:
                            logger.warning(f"Detected potentially truncated action: {action}")
                            action = f"{func_name}('{param}')"
                            logger.info(f"Auto-completed action to: {action}")
                
                if action:
                    break

        if not action:
            t_match = re.search(r'Thought:\s*(.*?)(?=Action:|$)', raw_string, re.DOTALL | re.IGNORECASE)
            if t_match:
                thought = t_match.group(1).strip()
            
            a_match = re.search(r'Action:\s*(.*?)(?=\n\n|$)', raw_string, re.DOTALL | re.IGNORECASE)
            if a_match:
                action = a_match.group(1).strip()
                action = action.strip('"').strip("'")

    except Exception as e:
        logger.warning(f"Error parsing string: {e}")
        return None, None
        
    return thought, action


def _fix_json_string(json_str: str) -> str:
    """Fix common JSON format errors: unclosed quotes, brackets, trailing commas."""
    fixed = json_str
    
    action_match = re.search(r'"action"\s*:\s*"([^"]*?)(?:"|,|}|$)', fixed, re.DOTALL)
    if action_match:
        action_start = action_match.start(1)
        action_end = action_match.end(1)
        if action_end < len(fixed) and fixed[action_end] not in ['"', ',', '}']:
            next_delimiter = min(
                (fixed.find(',', action_end) if fixed.find(',', action_end) != -1 else len(fixed)),
                (fixed.find('}', action_end) if fixed.find('}', action_end) != -1 else len(fixed))
            )
            fixed = fixed[:next_delimiter] + '"' + fixed[next_delimiter:]
    
    open_braces = fixed.count('{')
    close_braces = fixed.count('}')
    if open_braces > close_braces:
        fixed += '}' * (open_braces - close_braces)
    
    fixed = re.sub(r',\s*}', '}', fixed)
    fixed = fixed.rstrip().rstrip(',').rstrip()
    
    return fixed

@AgentFactory.register
class SolverAgent(BaseAgent):
    """Solver Agent for executing task instructions using LLM."""

    def __init__(
            self,
            llm_engine: LLMEngine,
            temperature: float = 0.01,
            char_limit: int = 16000,
            demo_mode: str = 'off',
            action_timeout: int = 2000,
            **kwargs
    ):
        """Initialize SolverAgent with LLM engine and action set."""
        super().__init__(**kwargs)
        
        self.llm = llm_engine
        self.temperature = temperature
        self.char_limit = char_limit
        self.action_timeout = action_timeout
        
        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "infeas", "nav"],
            strict=False,
            multiaction=False,
            demo_mode=demo_mode
        )

        self.prompt_builder = SolverPromptBuilder(self.action_set)
        self.history: list[BrowserGymAgentStepData] = []

    def reset(self):
        self.history = []

    def obs_preprocessor(self, obs: dict) -> dict:
        """Preprocess observation to extract fields needed for prompt construction."""
        return {
            "goal_object": obs.get("goal_object", [{"text": "Follow instructions."}]),
            "axtree_txt": flatten_axtree_to_str(
                obs["axtree_object"], 
                filter_visible_only=False, 
                extra_properties=obs.get("extra_element_properties", {})
            ),
            "last_action_error": obs.get("last_action_error", ""),
            "open_pages_urls": obs.get("open_pages_urls", []),
            "extra_element_properties": obs.get("extra_element_properties", {}),
        }
    
    def action_processor(self, action: str) -> str:
        """Convert semantic action string to executable Python code."""
        return self.action_set.to_python_code(action)

    
    def get_action(self, obs: dict, oracle_action: tuple[str, str] = None, **kwargs) -> tuple[str, dict]:
        """
        Generate action based on observation and history.
        Returns: (raw_action, extras_dict) where extras contains parsed_action, thought, etc.
        """
        current_step = BrowserGymAgentStepData(
            action=None,
            thought=None,
            axtree=obs["axtree_txt"],
            last_action_error=obs.get("last_action_error"),
            misc={}
        )

        response_text = ""
        action = ""
        thought = ""

        if oracle_action is None:
            messages_dict = self.prompt_builder.build_messages(
                goal=obs["goal_object"][0]["text"],
                current_step=current_step,
                history=self.history,
                char_limit=self.char_limit
            )
            
            raw_messages = messages_dict['prompt']
            flat_msgs = flatten_messages(raw_messages)
            
            system_msg = ""
            user_msg = ""
            
            for m in flat_msgs:
                if m['role'] == 'system':
                    system_msg = m['content']
                elif m['role'] == 'user':
                    user_msg = m['content']

            response_text = self.llm.generate(
                system_msg=system_msg,
                user_msg=user_msg,
                mode="base",
                temperature=self.temperature
            )
            
            thought, action = extract_action_and_thought(response_text)
            current_step.misc["model_usage"] = {"completion_tokens": len(response_text)//4}
        else:
            action, thought = oracle_action
            response_text = json.dumps({"thought": thought, "action": action})
            
        logger.info(f"Solver Output: Thought: {thought} Action: {action}")

        parsed_action = self.action_processor(action) if action else ""

        current_step.action = action
        current_step.thought = thought
        current_step.misc.update({
            "thought": thought, 
            "raw_action": action,
            "parsed_action": parsed_action,
            "raw_output": response_text
        })
        
        self.history.append(current_step)

        return action, {
            "raw_action": action,
            "parsed_action": parsed_action,
            "thought": thought,
            "raw_output": response_text
        }