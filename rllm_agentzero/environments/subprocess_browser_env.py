"""
Subprocess-based Browser Environment Wrapper

Runs BrowserGym in a separate subprocess to avoid Playwright Sync API 
conflicts with asyncio event loops.

Compatible with different BrowserGym versions (no browser_args).
"""
import multiprocessing as mp
import logging

logger = logging.getLogger(__name__)


class SubprocessBrowserEnv:
    """
    Browser environment that runs BrowserGym in a subprocess.
    
    Uses mp.Process + mp.Pipe to isolate Playwright from asyncio,
    similar to rLLM's BrowserGymEnv but without browser_args for compatibility.
    """
    
    def __init__(self, env_id="browsergym/openended", task=None, **env_kwargs):
        self.parent_conn, self.child_conn = mp.Pipe()
        self.env_id = env_id
        self.task = task
        self.env_kwargs = env_kwargs
        self.timeout = env_kwargs.get("timeout", None)  # in seconds
        
        # Extract task_kwargs for gym.make
        self.task_kwargs = None
        if task:
            self.task_kwargs = {
                "start_url": task.get("url", "https://www.google.com"),
                "goal": task.get("goal", "Explore the website"),
            }
        
        # Start worker process
        self.process = mp.Process(
            target=self._worker, 
            args=(self.child_conn, self.env_id, self.task_kwargs, self.env_kwargs)
        )
        self.process.start()
        logger.info(f"SubprocessBrowserEnv: Worker started (pid={self.process.pid})")

    def _worker(self, conn, env_id, task_kwargs, env_kwargs):
        """Worker function running in subprocess."""
        import gymnasium as gym
        
        # Import browsergym modules to register environments
        try:
            import browsergym.core  # noqa: F401 - registers browsergym namespace
        except ImportError:
            pass
        try:
            import browsergym.webarena  # noqa: F401
        except ImportError:
            pass
        try:
            import browsergym.miniwob  # noqa: F401
        except ImportError:
            pass
        
        # Filter out unsupported kwargs
        safe_kwargs = {}
        for key in ["headless", "timeout"]:
            if key in env_kwargs:
                safe_kwargs[key] = env_kwargs[key]
        
        # Create environment
        if task_kwargs:
            env = gym.make(
                env_id,
                task_kwargs=task_kwargs,
                **safe_kwargs,
            )
        else:
            env = gym.make(env_id, **safe_kwargs)
        
        try:
            while True:
                cmd, data = conn.recv()
                if cmd == "reset":
                    obs = env.reset()
                    conn.send(obs)
                elif cmd == "step":
                    action = data
                    obs, reward, terminated, truncated, extra_info = env.step(action)
                    conn.send((obs, reward, terminated or truncated, extra_info))
                elif cmd == "close":
                    env.close()
                    conn.close()
                    break
        except EOFError:
            env.close()

    def reset(self):
        """Reset the environment."""
        self.parent_conn.send(("reset", None))
        if self.timeout is not None:
            if not self.parent_conn.poll(self.timeout):
                raise TimeoutError(f"Timeout after {self.timeout} seconds")
        return self.parent_conn.recv()

    def step(self, action):
        """Execute action in environment."""
        self.parent_conn.send(("step", action))
        if self.timeout is not None:
            if not self.parent_conn.poll(self.timeout):
                raise TimeoutError(f"Timeout after {self.timeout} seconds")
        return self.parent_conn.recv()

    def close(self):
        """Close the environment."""
        try:
            self.parent_conn.send(("close", None))
            self.process.join(60 * 2)
            if self.process.is_alive():
                logger.warning("Process still alive, terminating...")
                self.process.terminate()
                self.process.join()
        except:
            pass

    @staticmethod
    def is_multithread_safe() -> bool:
        return True
