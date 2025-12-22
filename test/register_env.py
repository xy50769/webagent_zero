import gymnasium as gym
from browsergym.core.task import AbstractBrowserTask
from browsergym.core.env import BrowserEnv

class GenericTestTask(AbstractBrowserTask):
    def setup(self, page):
        self.goal = "Test Trace Replay"
        page.goto("https://www.google.com")
        return self.goal, {}

    def validate(self, page, chat_messages):
        return 0, False, "", {}

    def teardown(self):
        pass

def register_test_env():
    if "Test" not in gym.envs.registry:
        gym.register(
            id="Test",
            entry_point="browsergym.core.env:BrowserEnv",
            kwargs={
                "task_entrypoint": GenericTestTask,
                "viewport": {"width": 1280, "height": 720},
            },
        )

# register on import for convenience (keeps backward compat)
register_test_env()

if __name__ == "__main__":
    print("Environment 'Test' registered successfully!")