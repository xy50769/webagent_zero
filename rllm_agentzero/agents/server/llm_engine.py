import torch
import logging
from typing import Optional, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class LLMEngine:
    """
    LLM inference engine (singleton pattern).
    Loads base model, manages adapter switching, and formats prompts.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMEngine, cls).__new__(cls)
        return cls._instance
    
    def __init__(self,
                 base_model_name: str, 
                 adapter_model_path: str = None, 
                 use_4bit: bool = True,
                 max_new_tokens: int = 8192
                 ):
        """Initialize LLM engine with base model and optional adapter."""
        if hasattr(self, 'initialized') and self.initialized:
            return
        
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proposer_adapter_loaded = False 

        logger.info(f"Loading tokenizer: {base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ) if use_4bit else None

        logger.info(f"Loading base model: {base_model_name} on {self.device}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if not use_4bit else None
        )
        
        self.model = self.base_model

        if adapter_model_path:
            logger.info(f"Loading adapter: {adapter_model_path}...")
            self.model = PeftModel.from_pretrained(self.base_model, adapter_model_path, adapter_name="proposer")
            self.proposer_adapter_loaded = True
            self.model.set_adapter("proposer") 

        self.initialized = True 

    def construct_prompt(self, system_msg: str, user_msg: str) -> str:
        """Format system and user messages using chat template."""
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True 
        )
        return text
    
    def generate(self, system_msg: str, user_msg: str, mode: str = "base", temperature: float = 0.01) -> str:
        """Generate text using base model or adapter based on mode."""
        prompt_text = self.construct_prompt(system_msg, user_msg)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        try:
            with torch.no_grad():
                if mode == "base" and self.proposer_adapter_loaded:
                    with self.model.disable_adapter():
                        outputs = self.model.generate(
                            **inputs, 
                            max_new_tokens=self.max_new_tokens,
                            temperature=temperature,
                            do_sample=(temperature > 0),
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                else:
                    if mode == "proposer" and self.proposer_adapter_loaded:
                        self.model.set_adapter("proposer")
                    
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens=self.max_new_tokens,
                        temperature=temperature,
                        do_sample=(temperature > 0),
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return decoded.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""