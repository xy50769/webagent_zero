from .solver_prompt_builder import SolverPromptBuilder
from ..trajectory_data import BrowserGymAgentStepData
from textwrap import dedent

class ExplorerPromptBuilder(SolverPromptBuilder):
    """
    Explorer Prompt Builder
    Supports:
    1. Candidate Filtering (Mask): Guide model to avoid visited actions
    2. Frontier Guidance: Guide model to focus on unexplored areas
    """
    
    def system_message(self):
        """
        Override parent's system_message to add explicit History usage rules for exploration.
        """
        return {
            "type": "text",
            "text": dedent("""\
                # Instructions
                You are a UI Assistant specialized in exploration, your goal is to help the user explore and discover new states in a web browser.
                Review the instructions from the user, the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.
                
                ## Action Format Requirements
                **CRITICAL**: When interacting with elements, you MUST use the element's bid (browsergym id) which is shown in square brackets in the accessibility tree.
                - CORRECT: click('42') where 42 is the bid from [42] in the tree
                - WRONG: click('Page 2') or click('Next') - do NOT use text labels
                
                ## History Usage Rules
                You are given a history of past actions and their outcomes. You must follow these rules:
                
                1. **Avoid Repetition**: Do NOT repeat actions that have already been taken, unless they may lead to a different state (e.g., clicking on a dynamic element, submitting a form with different data, or navigating to a page that may have changed).
                
                2. **Prefer Novel Actions**: Prioritize actions that have NOT appeared in the history. Novel actions are more likely to lead to undiscovered states and provide new information.
                
                3. **Use History for Inference**: Analyze the history to infer:
                   - Which elements or links have already been explored
                   - Which pages have been visited
                   - Which interaction patterns have been tried
                   - Which areas of the website remain unexplored
                
                4. **Maximize Information Gain**: Your goal is exploration. Each action should aim to discover new states, reveal new elements, or expose new functionality. Avoid actions that are likely to result in already-seen states.
                """
            )
        }
    
    def construct_explorer_prompt_messages(
        self, 
        goal: str, 
        obs: dict, 
        history: list, 
        visited_actions: list = None,
        frontier_info: dict = None,
        unvisited_elements: list = None,  # 新增：未访问的元素
        visited_element_bids: list = None,  # 新增：已访问的元素 bid
        exploration_stats: dict = None  # 新增：探索统计
    ) -> list[dict]:
        """
        Build prompt with Exploration Mask and Frontier Info
        """
        # 1. Get base prompt (Goal, AxTree, History)
        current_step = BrowserGymAgentStepData(
            axtree=obs.get("axtree_txt", ""),
            last_action_error=obs.get("last_action_error", "")
        )
        
        messages_dict = self.build_messages(goal, current_step, history)
        messages = messages_dict['prompt']
        
        # 2. Inject Frontier Guidance
        if frontier_info:
            frontier_msg = (
                f"\n\n### Exploration Guidance (Frontier)\n"
                f"There are unexplored areas of the website (Frontier Nodes) that need attention.\n"
                f"Target Node ID: {frontier_info.get('node_id', 'Unknown')}\n"
                f"Target URL: {frontier_info.get('url', 'Unknown')}\n"
                f"Hint: Try to perform actions that might lead towards this area or similar unexplored states."
            )
            
            # Append to last user message
            if messages and messages[-1]['role'] == 'user':
                last_content = messages[-1]['content']
                if isinstance(last_content, list):
                    # If content is a list, append as new text item
                    messages[-1]['content'].append({
                        "type": "text",
                        "text": frontier_msg
                    })
                elif isinstance(last_content, str):
                    # If content is a string, append directly
                    messages[-1]['content'] += frontier_msg

        # 3. Inject Element-level Exploration Guidance（元素级别探索引导）
        if unvisited_elements and len(unvisited_elements) > 0:
            element_guidance = "\n\n### Unvisited Elements (PRIORITIZE THESE)\n"
            element_guidance += "The following elements on this page have NOT been interacted with yet:\n\n"
            
            for i, elem in enumerate(unvisited_elements[:]):
                element_guidance += f"  - bid='{elem['bid']}' [{elem['role']}]: \"{elem['text'][:60]}\"\n"
            
            
            # 添加探索统计
            if exploration_stats:
                element_guidance += (
                    f"\nCurrent Page Coverage: {exploration_stats['visited']}/{exploration_stats['total']} "
                    f"({exploration_stats['coverage']*100:.1f}%)\n"
                )
            
            element_guidance += (
                "\n**EXPLORATION STRATEGY**: Please select the element from the unvisited elements above that you think is most likely to reach the new state and interact with it.\n"
            )
            
            if messages and messages[-1]['role'] == 'user':
                last_content = messages[-1]['content']
                if isinstance(last_content, list):
                    messages[-1]['content'].append({
                        "type": "text",
                        "text": element_guidance
                    })
                elif isinstance(last_content, str):
                    messages[-1]['content'] += element_guidance
        
        # 4. Inject Exploration Mask (Visited Elements Warning)
        if visited_element_bids and len(visited_element_bids) > 0:
            mask_msg = (
                f"\n\n### Already Explored Elements (Avoid Unless Necessary)\n"
                f"These {len(visited_element_bids)} elements have been interacted with: "
                f"{visited_element_bids[:20]}"
            )
            
            if len(visited_element_bids) > 20:
                mask_msg += f" ... and {len(visited_element_bids) - 20} more."
            
            mask_msg += "\nPrefer unvisited elements to maximize exploration efficiency.\n"
            
            if messages and messages[-1]['role'] == 'user':
                last_content = messages[-1]['content']
                if isinstance(last_content, list):
                    messages[-1]['content'].append({
                        "type": "text",
                        "text": mask_msg
                    })
                elif isinstance(last_content, str):
                    messages[-1]['content'] += mask_msg
        
        # 5. Legacy action-level mask (keep for compatibility)
        if visited_actions:
            mask_msg = (
                f"\n\n### Exploration History\n"
                f"Repeated actions on this page: {visited_actions[-10:]} (showing last 10)\n"
            )
            
            if messages and messages[-1]['role'] == 'user':
                last_content = messages[-1]['content']
                if isinstance(last_content, list):
                    messages[-1]['content'].append({
                        "type": "text",
                        "text": mask_msg
                    })
                elif isinstance(last_content, str):
                    messages[-1]['content'] += mask_msg
            
        return messages

    def cot_examples(self) -> list[dict]:
        return [
            {
                "thought": "The goal is to explore new states. I see a 'Reviews' link (bid 42) that I haven't visited yet. This looks like a promising Frontier.", 
                "action": "click('42')"
            },
            {
                "thought": "I am on the product page. I've already clicked 'Description'. Now I will try adding the item to the cart (bid 1310) to see if it triggers a popup or a new state (Novelty).", 
                "action": "click('1310')"
            },
            {
                "thought": "I have explored all visible links. I will scroll down to reveal more potential interaction elements.", 
                "action": "scroll(0, 1000)"
            },
        ]