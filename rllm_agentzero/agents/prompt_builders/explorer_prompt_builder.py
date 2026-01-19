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
        Override parent's system_message to add exploration-specific instructions.
        """
        return {
            "type": "text",
            "text": dedent("""\
                # Instructions
                You are a **Web Explorer Agent**. Your goal is to map the structure of the website.
                
                ## Action Format (CRITICAL)
                Use element's `bid` from square brackets `[...]` in the accessibility tree.
                - CORRECT: `click('42')` (where 42 is the bid)
                - WRONG: `click('Add to Cart')` (Do NOT use text labels)

                ## Exploration Strategy
                1. **Prioritize content over navigation**: Click product items, forms, buttons rather than nav links
                2. **Avoid repetition**: Do NOT repeat actions already taken
                3. **Maximize discovery**: Find new states (popups, forms, checkout flows)
               """)
        }
    
    def construct_explorer_prompt_messages(
        self, 
        goal: str, 
        obs: dict, 
        history: list, 
        frontier_info: dict = None,
        unvisited_elements: list = None,  
        visited_elements: list = None
    ) -> list[dict]:
        """
        Build prompt with Exploration Mask and Frontier Info
        """
        # 1. Get base prompt (Goal, AxTree, History)
        # Truncate AXTree to fit within max_prompt_length (3000 chars ~ 750 tokens)
        axtree_txt = obs.get("axtree_txt", "")
        max_axtree_chars = 3000  # ~750 tokens, safe margin for other content
        if len(axtree_txt) > max_axtree_chars:
             axtree_txt = axtree_txt[:max_axtree_chars] + "\n...[AXTree Truncated]..."
        
        current_step = BrowserGymAgentStepData(
            axtree=axtree_txt,
            last_action_error=obs.get("last_action_error", "")
        )
        
        # Pass char_limit to enable history truncation if prompt is too long
        # max_prompt_length is 5500 in config, using 5000 as safety buffer
        messages_dict = self.build_messages(goal, current_step, history, char_limit=5000)
        messages = messages_dict['prompt']
        
        # 2. Inject Frontier Guidance (compact)
        if frontier_info:
            frontier_msg = (
                f"\n\n### Frontier Node\n"
                f"Target: {frontier_info.get('node_id', 'Unknown')} - {frontier_info.get('url', 'Unknown')}"
            )
            messages[-1]['content'] += frontier_msg

        # 3. Inject Element-level Exploration Guidance
        if unvisited_elements and len(unvisited_elements) > 0:
            limit = 10
            element_guidance = "\n\n### Unvisited Elements (PRIORITIZE)\n"
            
            for elem in unvisited_elements[:limit]:
                element_guidance += f"- bid='{elem['bid']}' [{elem['role']}]: {elem['text'][:50]}\n"
            
            if len(unvisited_elements) > limit:
                element_guidance += f"... (+{len(unvisited_elements) - limit} more)\n"

            messages[-1]['content'] += element_guidance
        
        # 4. Compact Visited Elements Warning
        if visited_elements and len(visited_elements) > 0:
            mask_msg = f"\n\n### Already Explored ({len(visited_elements)} elements) - avoid unless necessary\n"
            messages[-1]['content'] += mask_msg
        
        return messages

    def cot_examples(self) -> list[dict]:
        return [
            {"thought": "Click product to explore details", "action": "click('42')"},
            {"thought": "Add to cart to see checkout flow", "action": "click('1310')"},
        ]
