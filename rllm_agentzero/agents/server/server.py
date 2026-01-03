import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_engine import LLMEngine 

app = FastAPI()

engine = LLMEngine(
    base_model_name="/root/autodl-fs/go-browse-model", 
    adapter_model_path="/root/autodl-fs/outputs_proposer_qwen2.5_7B_lora/final_checkpoint", 
    use_4bit=True
)


class GenerateRequest(BaseModel):
    system_msg: str
    user_msg: str
    mode: str = "base"
    temperature: float = 0.01

@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        response = engine.generate(
            system_msg=req.system_msg,
            user_msg=req.user_msg,
            mode=req.mode,
            temperature=req.temperature
        )
        return {"text": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006)