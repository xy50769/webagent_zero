import os
import sys
import shutil
import logging
import gymnasium as gym

# === 路径配置 ===
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rllm_agentzero.core.graph import Graph
from rllm_agentzero.core.episode import run_episode

# 尝试导入 BaseAgent，兼容不同的项目结构
try:
    from rllm_agentzero.agents_old.base_agent import BaseAgent
except ImportError:
    try:
        from agents.base_agent import BaseAgent
    except ImportError:
        raise ImportError("Could not import BaseAgent. Please check your python path.")

# 引入 BrowserGym 相关组件
from browsergym.core.env import BrowserEnv
from browsergym.core.task import AbstractBrowserTask

# === 日志配置 ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestEpisode")

# ==========================================
# 🛠️ 1. 定义 Mock 组件 (替身)
# ==========================================

class MockProposerAgent(BaseAgent):
    """一个模拟 Agent，不调用 LLM，直接返回固定动作"""
    
    def reset(self):
        logger.info("MockAgent reset called.")

    def obs_preprocessor(self, obs: dict) -> dict:
        """简单的预处理，确保 observation 中包含 RLLM 需要的字段"""
        # 模拟真实 Agent 将 AxTree 展平为字符串的过程
        if "axtree_txt" not in obs:
            obs["axtree_txt"] = "[Mock AxTree] Button: Submit, Link: Home..."
        return obs

    def get_action(self, obs: dict, oracle_action=None, **kwargs) -> tuple[str, dict]:
        # 模拟思考过程
        thought = "I am testing the integration loop."
        # 返回一个安全的 Python 动作 (打印语句)
        action = "print('🤖 Mock Agent Action Executed!')"
        
        return action, {
            "thought": thought,
            "parsed_action": action,
            "model_usage": {"input_tokens": 10, "output_tokens": 10}
        }

class MockEvaluator:
    """一个总是给出正面评价的评测器"""
    def evaluate(self, trajectory):
        logger.info("MockEvaluator: Evaluating trajectory... Result: Success! 👍")
        trajectory.success = True
        trajectory.reward = 1.0
        trajectory.misc["evaluation_info"] = {"status": "success (mock)", "score": 100}

class GenericTestTask(AbstractBrowserTask):
    """一个最简的 BrowserGym 任务定义"""
    def setup(self, page):
        return "Test Goal: Run Loop", {}
    
    def validate(self, page, chat_messages):
        return 0.0, False, "", {}
    
    def teardown(self):
        pass

# 注册测试环境 (防止重复注册)
if "browsergym/test" not in gym.envs.registry:
    gym.register(
        id="browsergym/test",
        entry_point="browsergym.core.env:BrowserEnv",
        kwargs={"task_entrypoint": GenericTestTask}
    )

# ==========================================
# 🏃 2. 执行测试逻辑
# ==========================================

def test_episode_loop():
    print("\n🎬 Starting Episode Integration Test...")
    
    # 准备测试目录
    test_dir = "./test_episode_data"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    env = None
    try:
        # 1. 初始化 Graph
        graph = Graph(root_url="https://www.example.com", exp_dir=test_dir)
        
        # 2. 手动造一个起点 Node
        obs_mock = {"url": "https://www.example.com", "axtree_txt": "Example Domain Mock"}
        node, _ = graph.add_state(obs_mock, parent=None, prefixes=[], hint="Start Here")
        
        # 3. 初始化环境
        # headless=True 在 CI/CD 或服务器上更好，headless=False 方便本地调试
        env = gym.make("browsergym/test", headless=True) 
        
        # 4. 初始化 Mock 对象
        agent = MockProposerAgent()
        evaluator = MockEvaluator()
        
        # 5. 【核心】运行 Episode
        print("\n🚀 Calling run_episode()...")
        traj = run_episode(
            goal="Test the whole loop",
            node=node,
            env=env,
            agent=agent,
            evaluator=evaluator,
            graph=graph,
            max_steps=3 # 只跑3步
        )
        
        print("\n✅ run_episode() returned successfully!")
        
        # 6. 结果验证
        assert len(traj.steps) == 3, f"Expected 3 steps, got {len(traj.steps)}"
        assert traj.steps[0].action == "print('🤖 Mock Agent Action Executed!')", "Action content mismatch"
        assert "axtree_txt" in traj.steps[0].observation, "Observation 'axtree_txt' missing in trajectory"
        assert traj.success is True, "Evaluator failed to mark trajectory as success"
        
        print("   - Trajectory steps check passed.")
        print("   - Observation recording check passed.")
        print("   - Evaluation result check passed.")

    except Exception as e:
        print(f"\n❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 资源清理
        if env:
            env.close()
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        print("\n🧹 Cleanup done.")

if __name__ == "__main__":
    test_episode_loop()