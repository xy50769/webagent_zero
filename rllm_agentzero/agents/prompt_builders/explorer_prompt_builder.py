from .solver_prompt_builder import SolverPromptBuilder
from ..trajectory_data import BrowserGymAgentStepData

class ExplorerPromptBuilder(SolverPromptBuilder):
    """
    Explorer Prompt Builder
    Supports:
    1. Candidate Filtering (Mask): Guide model to avoid visited actions
    2. Frontier Guidance: Guide model to focus on unexplored areas
    """
    
    def construct_explorer_prompt_messages(
        self, 
        goal: str, 
        obs: dict, 
        history: list, 
        visited_actions: list = None,
        frontier_info: dict = None
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

        # 3. Inject Exploration Mask (Candidate Filtering)
        if visited_actions:
            mask_msg = (
                f"\n\n### Exploration Mask (Avoid These)\n"
                f"You have already visited/explored the following actions on this page ({len(visited_actions)} times).\n"
                f"To maximize information gain, DO NOT repeat them unless necessary for navigation.\n"
                f"Visited Actions: {visited_actions[-10:]} (showing last 10)"
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
                "thought": "I am on the product page. I've already clicked 'Description'. Now I will try adding the item to the cart to see if it triggers a popup or a new state (Novelty).", 
                "action": "click('add-to-cart-btn')"
            },
            {
                "thought": "I have explored all visible links. I will scroll down to reveal more potential interaction elements.", 
                "action": "scroll(0, 1000)"
            },
        ]