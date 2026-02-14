from . import BasePromptBuilder, flatten_messages
from ..trajectory_data import BrowserGymAgentStepData, BrowserGymAgentTrajectoryData
from browsergym.core.action.base import AbstractActionSet
from dataclasses import dataclass
from textwrap import dedent
import json

class SolverPromptBuilder(BasePromptBuilder):

    def __init__(self, action_set: AbstractActionSet):
        self.action_set = action_set
        self.action_set_description = action_set.describe(with_long_description=True, with_examples=True)


    def build_messages(self, obs: dict):
        messages = []
        if "message" in obs:
            messages.append({"text": obs["message"]})
        return messages

    def format_thought_and_action(self, thought: str, action: str) -> str:
        d = {}
        if thought:
            d['thought'] = thought
        if action:
            d['action'] = action
        return json.dumps(d)
    
    def trim_axtree(self, axtree: str, num_chars_overflow: int) -> str:
        trim_str = "...trimmed due to context size limit"
        return axtree[:-num_chars_overflow - len(trim_str)] + trim_str
            
    
    def build_trajectory_messages(self, trajectory_data: BrowserGymAgentTrajectoryData, char_limit: int=-1) -> list[dict]:
        messages = []
        for i, step in enumerate(trajectory_data.steps):
            messages.append(self.build_messages(trajectory_data.goal, step, trajectory_data.steps[:i], char_limit))
        return messages
    

    def build_messages(self, goal: str, current_step: BrowserGymAgentStepData, history: list[BrowserGymAgentStepData], char_limit: int=-1) -> dict:
        past_thoughts = [step.thought for step in history]
        # Use raw_action (concise LLM output) instead of parsed_action (full executable code)
        past_actions = [step.misc.get('raw_action', step.action) if step.misc else step.action for step in history]
        
        axtree = current_step.axtree
        last_action_error = current_step.last_action_error
        completion_thought = current_step.thought
        completion_action = current_step.misc.get('raw_action', current_step.action) if current_step.misc else current_step.action

        add_completion = completion_thought or completion_action
        
        messages = self._build_messages(
            goal,
            past_thoughts,
            past_actions,
            axtree,
            last_action_error,
            completion_thought,
            completion_action
        )
        curr_char_count = self.count_message_chars(messages['prompt'] + (messages['completion'] if add_completion else []))
        if char_limit > 0 and curr_char_count > char_limit:
            past_thoughts, past_actions = self.trim_past_thoughts_and_actions(past_thoughts, past_actions, max_allowed=8)
            messages = self._build_messages(
                goal,
                past_thoughts,
                past_actions,
                axtree,
                last_action_error,
                completion_thought,
                completion_action
            )
            
            curr_char_count = self.count_message_chars(messages['prompt'] + (messages['completion'] if add_completion else []))
            remaining_overflow = curr_char_count - char_limit
            if remaining_overflow > 0:
                axtree = self.trim_axtree(axtree, remaining_overflow)
                messages = self._build_messages(    
                    goal,
                    past_thoughts,
                    past_actions,
                    axtree,
                    last_action_error,
                    completion_thought,
                    completion_action
                )

        return {k : flatten_messages(v) for k, v in messages.items() if v}
        
    
    def count_message_chars(self, messages: list[dict]) -> int:
        return sum([len(m['text']) for message in messages for m in message['content']])
    
    def trim_past_thoughts_and_actions(self, past_thoughts: list[str | None], past_actions: list[str], max_allowed: int=3) -> tuple[list[str | None], list[str]]:
        if len(past_thoughts) > max_allowed:
            past_thoughts = past_thoughts[-max_allowed:]
            past_actions = past_actions[-max_allowed:]
        return past_thoughts, past_actions
    

    def _build_messages(
        self,
        goal: str,
        thoughts: list[str | None],
        actions: list[str | None],
        axtree: str,
        last_action_error: str | None = None,
        completion_thought: str | None = None,
        completion_action: str | None = None
    ):
        system_messages = {"role": "system", "content": [self.system_message()]}
        user_messages = {
            "role": "user",
            "content": [
                self.goal_message(goal),
                self.axtree_message(axtree),
                self.action_space_message(self.action_set),
                self.action_history_messages(thoughts, actions),
            ]
        }
        if last_action_error:
            user_messages["content"].append(self.last_action_error_message(last_action_error))
        
        user_messages["content"].append(self.next_action_request_message())
        
        output = { "prompt": [system_messages, user_messages] }
        
        if completion_thought or completion_action:
            assistant_messages = {
                "role": "assistant",
                "content": [self.completion_message(completion_thought, completion_action)]
            }
            output["completion"] = [assistant_messages]
        
        return output


    def system_message(self):
        return  {
                "type": "text",
                "text": dedent("""\
                    # Instructions
                    You are a UI Assistant helping users perform tasks using a web browser. 
                    Review the current page state and determine the best next action.
                    
                    ## Action Format (CRITICAL)
                    Use element's `bid` from square brackets in the accessibility tree.
                    - CORRECT: click('42') where 42 is the bid from [42]
                    - WRONG: click('Submit') - do NOT use text labels
                    """
                )
        }

    def goal_message(self, goal: str):
        return  {
                "type": "text",
                "text": (
                    "# Goal\n"
                    f"{goal}"
                )
        }
        
        
    def action_space_message(self, action_set: AbstractActionSet):
        return  {
                "type": "text",
                "text": (
                    "# Action Space\n"
                    "Available actions: click(bid), fill(bid, text), scroll(x, y), goto(url)\n\n"
                    "Output format: {\"thought\": \"reasoning\", \"action\": \"click('42')\"}\n"
                )
        }
        
    def cot_examples(self) -> list[dict]:
        return [
            {"thought": "I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.", "action": "click('12')"},
            {"thought": "I found the information requested by the user, I will send it to the chat.", "action": "send_msg_to_user('The price for a 15 inch laptop is 1499 USD.')"},
            {"thought": "I have finished navigating to the Products page. I will inform the user that I have completed the task.", "action": "send_msg_to_user('I have finished navigating to the Products page.')"},
        ]
        

    def axtree_message(self, axtree: str):
        return  {
                "type": "text",
                "text": (
                    "# Current page Accessibility Tree\n"
                    f"{axtree}"
                )
        }
        
    def last_action_error_message(self, last_action_error: str):
        return  {
                "type": "text",
                "text": (
                    "# Error message from last action\n"
                    f"{last_action_error}"
                )
        }
        
    def action_history_messages(self, thoughts: list[str | None], actions: list[str]):
        newline = "\n"
        return  {
                "type": "text",
                "text": (
                    "# History of past actions\n"
                    f"{newline.join(self.format_thought_and_action(thought, action) for thought, action in zip(thoughts, actions))}"
                )
        }
        
    
    def next_action_request_message(self):
        return  {
                "type": "text",
                "text": (
                    "# Next Action\n"
                    "Think step by step about the best next action. "
                    "Output a single JSON with thought and action keys.\n"
                )
        }
        
    
    def completion_message(self, completion_thought: str, completion_action: str):
        return  {
                "type": "text",
                "text": f"{self.format_thought_and_action(completion_thought, completion_action)}"
        }


