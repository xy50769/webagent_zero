"""
Data Preparation Script

Prepares and registers datasets for rLLM training.
"""
import json
import logging
import os
from typing import Dict, List, Optional

from rllm.data.dataset import DatasetRegistry

logger = logging.getLogger(__name__)


def prepare_webarena_data(
    train_urls: List[str] = None,
    test_urls: List[str] = None,
    config_file: Optional[str] = None,
) -> tuple:
    """
    Prepare and register WebArena/OneStopMarket datasets.
    
    Args:
        train_urls: List of URLs for training
        test_urls: List of URLs for testing
        config_file: Optional path to JSON config file with URLs
        
    Returns:
        Tuple of (train_dataset, test_dataset)
        
    Example:
        train_dataset, test_dataset = prepare_webarena_data(
            train_urls=["http://example.com/page1", "http://example.com/page2"],
            test_urls=["http://example.com/page3"]
        )
    """
    # Load from config file if provided
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        train_urls = config.get("train_urls", [])
        test_urls = config.get("test_urls", [])
    
    # Default URLs if none provided
    if not train_urls:
        train_urls = [
            "http://3.148.75.200:7770/",  # OneStopMarket default
        ]
    
    if not test_urls:
        test_urls = train_urls[:1]  # Use first training URL for testing
    
    # Create training data
    train_data = [
        {
            "url": url,
            "goal": "Explore the website to discover new pages and interaction states",
            "env_id": "browsergym/openended",
            "headless": True,
        }
        for url in train_urls
    ]
    
    # Create test data
    test_data = [
        {
            "url": url,
            "goal": "Explore the website to discover new pages and interaction states",
            "env_id": "browsergym/openended",
            "headless": True,
        }
        for url in test_urls
    ]
    
    # Register datasets
    train_dataset = DatasetRegistry.register_dataset("webarena", train_data, "train")
    test_dataset = DatasetRegistry.register_dataset("webarena", test_data, "test")
    
    logger.info(f"Registered webarena dataset:")
    logger.info(f"  Train: {len(train_data)} examples")
    logger.info(f"  Test: {len(test_data)} examples")
    
    return train_dataset, test_dataset


def prepare_exploration_tasks(
    base_url: str,
    num_tasks: int = 10,
    dataset_name: str = "exploration"
) -> tuple:
    """
    Generate exploration tasks with different starting conditions.
    
    Args:
        base_url: Base URL for exploration
        num_tasks: Number of tasks to generate
        dataset_name: Name for the dataset
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Generate tasks with variations
    tasks = []
    for i in range(num_tasks):
        tasks.append({
            "url": base_url,
            "goal": f"Explore the website starting from the homepage. Focus on discovering navigation patterns and interactive elements. Task {i+1}.",
            "env_id": "browsergym/openended",
            "headless": True,
            "task_id": f"explore_{i}",
            "exp_dir": f"./exploration_output/task_{i}",
        })
    
    # Split into train/test (80/20)
    split_idx = int(len(tasks) * 0.8)
    train_data = tasks[:split_idx] if split_idx > 0 else tasks[:1]
    test_data = tasks[split_idx:] if split_idx < len(tasks) else tasks[-1:]
    
    # Register
    train_dataset = DatasetRegistry.register_dataset(dataset_name, train_data, "train")
    test_dataset = DatasetRegistry.register_dataset(dataset_name, test_data, "test")
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Prepare default WebArena data
    train, test = prepare_webarena_data()
    print(f"Train dataset: {len(train.get_data())} examples")
    print(f"Test dataset: {len(test.get_data())} examples")
    print(f"Sample: {train.get_data()[0]}")
