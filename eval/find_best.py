import json
import os
import argparse
import re
from collections import defaultdict

def calculate_average_accuracy(file_path):
    """Compute the average accuracy for a single JSONL file."""
    total_correct = 0
    total_responses = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'answers_correctness' in data:
                    correct_list = data['answers_correctness']
                    total_correct += sum(correct_list)
                    total_responses += len(correct_list)
    except FileNotFoundError:
        print(f"WARNING: File not found:{file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Warning: JSON decoding error:{file_path}")
        return None

    if total_responses > 0:
        return total_correct / total_responses
    else:
        return 0.0

def find_all_checkpoint_accuracies_sorted_and_best(output_root, task_name):
    """Traverse the output directory, calculate the average accuracy of all checkpoints, print them sorted by step, and find the best checkpoint."""
    checkpoint_data = {}
    best_accuracy = -1.0
    best_checkpoint_step = None

    for root, dirs, files in os.walk(output_root):
        for file in files:
            if file.endswith(".jsonl") and "_k5_" in file and root.endswith(f"/{task_name}"):
                file_path = os.path.join(root, file)
                accuracy = calculate_average_accuracy(file_path)
                if accuracy is not None:
                    # Extract the checkpoint name and step from the file path
                    parts = root.split(os.sep)
                    checkpoint_dir = parts[-2]
                    match = re.search(r"checkpoint-(\d+)", checkpoint_dir)
                    if match:
                        step = int(match.group(1))
                        if step not in checkpoint_data:
                            checkpoint_data[step] = {'total_correct': 0, 'total_responses': 0}
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                data = json.loads(line.strip())
                                if 'answers_correctness' in data:
                                    correct_list = data['answers_correctness']
                                    checkpoint_data[step]['total_correct'] += sum(correct_list)
                                    checkpoint_data[step]['total_responses'] += len(correct_list)

    # Calculate the average accuracy of each checkpoint
    checkpoint_accuracies = {}
    for step, counts in checkpoint_data.items():
        if counts['total_responses'] > 0:
            accuracy = counts['total_correct'] / counts['total_responses']
            checkpoint_accuracies[step] = accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_checkpoint_step = step
        else:
            checkpoint_accuracies[step] = None

    # Sort by step and print the average accuracy of all checkpoints
    print("Average accuracy of all checkpoints (sorted by step):")
    for step in sorted(checkpoint_accuracies.keys()):
        accuracy = checkpoint_accuracies[step]
        if accuracy is not None:
            print(f"Checkpoint: checkpoint-{step}, Average accuracy: {accuracy:.4f}")
        else:
            print(f"Checkpoint: checkpoint-{step}, No valid evaluation results were found.")

    # print best checkpoint
    print("\n-----------------------------------------\n")
    if best_checkpoint_step is not None:
        print(f"The checkpoint with the highest average accuracy is: checkpoint-{best_checkpoint_step}")
        print(f"Average accuracy: {best_accuracy:.4f}")
    else:
        print("No valid evaluation results were found.")

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate and print the average accuracy for all checkpoints, sorted by step, and identify the best one.")
    parser.add_argument("--output_root", type=str, required=True, help="The root directory containing the evaluation outputs.")
    parser.add_argument("--task_name", type=str, required=True, help="The name of the task (e.g., 'math').")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    find_all_checkpoint_accuracies_sorted_and_best(args.output_root, args.task_name)