import logging
import requests
import time

logger = logging.getLogger(__name__)

class LLMEngine:
    """
    RLLM 远程推理客户端
    职责: 将生成请求发送给 AutoDL 上的 API Server
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, api_url: str = "http://127.0.0.1:6006/generate"):
        """Initialize LLMEngine with API URL."""
        self.api_url = api_url
        logger.info(f"Connected to Remote LLM Engine at {self.api_url}")

    def generate(self, system_msg: str, user_msg: str, mode: str = "base", temperature: float = 0.01) -> str:
        """
        send http to get response
        """
        payload = {
            "system_msg": system_msg,
            "user_msg": user_msg,
            "mode": mode,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            
            if response.status_code == 200:
                return response.json()["text"]
            else:
                logger.error(f"Remote LLM Error {response.status_code}: {response.text}")
                return ""
                
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to AutoDL server. Please check your SSH Tunnel.")
            return ""
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return ""

    def construct_prompt(self, system_msg: str, user_msg: str) -> str:
        return f"{system_msg}\n\n{user_msg}"
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("\n" + "="*50)
    print("Local LLMEngine Client Test")
    print("="*50)
    print("This test will connect to the LLM API server running on AutoDL.")
    print("Make sure you have started the server.py on AutoDL first.")
    print("ssh -CNg -L 6006:127.0.0.1:6006 -p 28244 root@connect.bjb2.seetacloud.com")
    print("-" * 50)

    client = LLMEngine(api_url="http://127.0.0.1:6006/generate")

    test_system = "You are a helpful assistant."
    test_user = "Hello! Please introduce yourself briefly."
    print("\nTest Prompt:")
    print(f"   System: {test_system}")
    print(f"   User:   {test_user}")

    start_time = time.time()
    # response = client.generate(
    #     system_msg=test_system, 
    #     user_msg=test_user, 
    #     mode="base",   
    #     temperature=0.7
    # )

    response = client.generate(
        system_msg=test_system, 
        user_msg=test_user, 
        mode="proposer",   
        temperature=0.7
    )
    duration = time.time() - start_time

    print("\n" + "-"*50)
    if response:
        print(f"Test successful! (Duration: {duration:.2f}s)")
        print(f"Model Response:\n{response}")
    else:
        print("Failed to get response from the LLM API server.")
    print("="*50 + "\n")