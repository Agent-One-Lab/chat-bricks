"""
Workload test for tokenize_conversations to identify performance bottlenecks.

This test loads ~1k samples from HuggingFace datasets and profiles tokenize_conversations
to identify where time is being spent.
"""
import cProfile
import pstats
import io
import time
from typing import List, Dict, Any
from transformers import AutoTokenizer
from chat_bricks.utils import tokenize_conversations
import pytest


def load_conversation_dataset(dataset_names: List[str] = None, num_samples: int = 1000):
    """
    Load conversation data from HuggingFace datasets.
    
    Args:
        dataset_names: List of dataset names to try (will try each until one works)
        num_samples: Number of samples to load (default: 1000)
    
    Returns:
        List of message lists in the format expected by tokenize_conversations
    """
    if dataset_names is None:
        dataset_names = [
            "HuggingFaceH4/ultrafeedback_binarized",
            "Open-Orca/OpenOrca",
            "HuggingFaceH4/oasst2_en",
        ]
    
    try:
        try:
            from datasets import load_dataset
        except ImportError:
            print("Warning: 'datasets' library not installed. Install with: pip install datasets")
            return generate_synthetic_conversations(num_samples)
        
        # Try each dataset until one works
        for dataset_name in dataset_names:
            try:
                print(f"Trying to load dataset: {dataset_name}")
                dataset = load_dataset(dataset_name, split="train", streaming=False)
        
                # Take first num_samples
                dataset_size = len(dataset)
                if dataset_size == 0:
                    print(f"  Dataset {dataset_name} is empty, trying next...")
                    continue
                    
                take_count = min(num_samples, dataset_size)
                dataset = dataset.select(range(take_count))
                
                messages_list = []
                for example in dataset:
                    # Try to extract messages from different possible formats
                    messages = None
                    
                    # Format 1: Direct messages field
                    if "messages" in example:
                        messages = example["messages"]
                    
                    # Format 2: UltraFeedback format (chosen/rejected with messages)
                    elif "chosen" in example and isinstance(example["chosen"], dict):
                        if "messages" in example["chosen"]:
                            messages = example["chosen"]["messages"]
                    
                    # Format 3: Prompt/response format
                    elif "prompt" in example and "response" in example:
                        messages = [
                            {"role": "user", "content": str(example["prompt"])},
                            {"role": "assistant", "content": str(example["response"])}
                        ]
                    
                    # Format 4: Instruction/input/output format
                    elif "instruction" in example:
                        user_content = str(example["instruction"])
                        if "input" in example and example["input"]:
                            user_content += "\n" + str(example["input"])
                        messages = [
                            {"role": "user", "content": user_content}
                        ]
                        if "output" in example:
                            messages.append({"role": "assistant", "content": str(example["output"])})
                    
                    # Format 5: Conversation format
                    elif "conversation" in example:
                        messages = example["conversation"]
                    
                    if messages and isinstance(messages, list) and len(messages) > 0:
                        # Validate message format
                        valid = True
                        for msg in messages:
                            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                                valid = False
                                break
                        
                        if valid:
                            messages_list.append(messages)
                
                if len(messages_list) > 0:
                    print(f"  Successfully loaded {len(messages_list)} valid conversations from {dataset_name}")
                    return messages_list[:num_samples]
                else:
                    print(f"  No valid conversations found in {dataset_name}, trying next...")
                    continue
                    
            except Exception as e:
                print(f"  Failed to load {dataset_name}: {e}")
                continue
        
        # If all datasets failed, use synthetic data
        print("All datasets failed, falling back to synthetic data...")
        return generate_synthetic_conversations(num_samples)
        
    except Exception as e:
        print(f"Unexpected error loading datasets: {e}")
        print("Falling back to synthetic data...")
        import traceback
        traceback.print_exc()
        return generate_synthetic_conversations(num_samples)


def generate_synthetic_conversations(num_samples: int = 1000) -> List[List[Dict[str, Any]]]:
    """
    Generate synthetic conversation data for testing.
    
    Args:
        num_samples: Number of conversations to generate
    
    Returns:
        List of message lists
    """
    import random
    
    templates = [
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in simple terms."},
            {"role": "assistant", "content": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to perform computations. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in multiple states simultaneously."},
            {"role": "user", "content": "How does this differ from classical computing?"},
            {"role": "assistant", "content": "Classical computers process information using bits that are definitively 0 or 1. Quantum computers can process many possibilities simultaneously, potentially solving certain problems much faster."}
        ],
        [
            {"role": "user", "content": "Write a Python function to calculate factorial."},
            {"role": "assistant", "content": "Here's a Python function to calculate factorial:\n\n```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)\n```"}
        ],
        [
            {"role": "user", "content": "What are the main causes of climate change?"},
            {"role": "assistant", "content": "The main causes of climate change include greenhouse gas emissions from burning fossil fuels, deforestation, industrial processes, and agricultural practices. These activities increase the concentration of greenhouse gases in the atmosphere, trapping heat and causing global temperatures to rise."}
        ],
        [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "How do I implement a binary search tree?"},
            {"role": "assistant", "content": "A binary search tree (BST) is a data structure where each node has at most two children, and for each node, all values in the left subtree are less than the node's value, and all values in the right subtree are greater."},
            {"role": "user", "content": "Can you show me the code?"},
            {"role": "assistant", "content": "Here's a basic BST implementation in Python:\n\n```python\nclass TreeNode:\n    def __init__(self, val):\n        self.val = val\n        self.left = None\n        self.right = None\n\nclass BST:\n    def __init__(self):\n        self.root = None\n    \n    def insert(self, val):\n        # Implementation here\n        pass\n```"}
        ]
    ]
    
    messages_list = []
    for i in range(num_samples):
        # Randomly select a template and optionally add variations
        base = random.choice(templates)
        messages = [msg.copy() for msg in base]
        
        # Add some random variations
        if random.random() < 0.3:
            messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
        
        messages_list.append(messages)
    
    print(f"Generated {len(messages_list)} synthetic conversations")
    return messages_list


def profile_tokenize_conversations(
    messages_list: List[List[Dict[str, Any]]],
    tokenizer,
    template: str,
    max_length: int = 2048,
    profile_output_file: str = None
) -> Dict[str, Any]:
    """
    Profile tokenize_conversations and return timing and profiling information.
    
    Args:
        messages_list: List of conversations to tokenize
        tokenizer: Tokenizer to use
        template: Template name
        max_length: Maximum sequence length
        profile_output_file: Optional file to save profiling results
    
    Returns:
        Dictionary with timing and profiling statistics
    """
    # Warm-up run
    print("Warming up...")
    _ = tokenize_conversations(
        messages_list[:10],
        tokenizer,
        template,
        max_length=max_length,
        add_generation_prompt=False
    )
    
    # Time the actual run
    print(f"Profiling tokenize_conversations on {len(messages_list)} conversations...")
    start_time = time.time()
    
    # Create profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run tokenization
    result = tokenize_conversations(
        messages_list,
        tokenizer,
        template,
        max_length=max_length,
        add_generation_prompt=False
    )
    
    profiler.disable()
    end_time = time.time()
    
    # Get profiling stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    
    profile_output = s.getvalue()
    
    # Save to file if requested
    if profile_output_file:
        with open(profile_output_file, 'w') as f:
            f.write(profile_output)
        print(f"Profile saved to {profile_output_file}")
    
    elapsed_time = end_time - start_time
    avg_time_per_conversation = elapsed_time / len(messages_list)
    
    # Extract key statistics
    stats = {
        'total_time': elapsed_time,
        'num_conversations': len(messages_list),
        'avg_time_per_conversation': avg_time_per_conversation,
        'conversations_per_second': len(messages_list) / elapsed_time,
        'profile_output': profile_output,
        'result_shape': {
            'input_ids': result['input_ids'].shape if 'input_ids' in result else None,
            'attention_mask': result['attention_mask'].shape if 'attention_mask' in result else None,
        }
    }
    
    return stats


@pytest.mark.slow
def test_workload_tokenize_conversations():
    """
    Main workload test for tokenize_conversations.
    
    This test:
    1. Loads ~1k conversation samples from HuggingFace datasets
    2. Profiles tokenize_conversations to identify bottlenecks
    3. Reports timing and profiling statistics
    """
    # Configuration
    num_samples = 100
    tokenizer_name = "Qwen/Qwen2.5-3B-Instruct"
    max_length = 2048
    
    print("=" * 80)
    print("Workload Test: tokenize_conversations Performance Analysis")
    print("=" * 80)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # Load dataset
    print(f"\nLoading {num_samples} conversation samples...")
    messages_list = load_conversation_dataset(num_samples=num_samples)
    
    if len(messages_list) == 0:
        pytest.skip("No valid conversations loaded from dataset")
    
    print(f"Loaded {len(messages_list)} conversations")
    print(f"Average messages per conversation: {sum(len(msgs) for msgs in messages_list) / len(messages_list):.2f}")
    
    # Profile tokenization
    print(f"\n{'=' * 80}")
    print("Profiling tokenize_conversations...")
    print(f"{'=' * 80}\n")
    
    stats = profile_tokenize_conversations(
        messages_list,
        tokenizer,
        template=tokenizer_name,
        max_length=max_length,
        profile_output_file="tokenize_conversations_profile.txt"
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("Performance Results")
    print("=" * 80)
    print(f"Total conversations processed: {stats['num_conversations']}")
    print(f"Total time: {stats['total_time']:.2f} seconds")
    print(f"Average time per conversation: {stats['avg_time_per_conversation']*1000:.2f} ms")
    print(f"Throughput: {stats['conversations_per_second']:.2f} conversations/second")
    print(f"\nOutput shapes:")
    print(f"  input_ids: {stats['result_shape']['input_ids']}")
    print(f"  attention_mask: {stats['result_shape']['attention_mask']}")
    
    print("\n" + "=" * 80)
    print("Top Functions by Cumulative Time (cProfile)")
    print("=" * 80)
    print(stats['profile_output'])
    
    # Assertions to ensure the test actually ran
    assert stats['num_conversations'] > 0, "No conversations were processed"
    assert stats['total_time'] > 0, "Processing took no time (unexpected)"
    assert 'input_ids' in stats['result_shape'] and stats['result_shape']['input_ids'] is not None, "No input_ids in result"


if __name__ == "__main__":
    # Allow running directly without pytest
    import sys
    # Set up basic logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    try:
        test_workload_tokenize_conversations()
    except Exception as e:
        print(f"\nError running test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
