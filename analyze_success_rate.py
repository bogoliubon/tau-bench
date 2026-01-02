#!/usr/bin/env python3
"""
Analyze success rates across tasks from multiple tau-bench trajectory runs.
Creates a histogram showing the distribution of success rates.
"""

import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse


def load_trajectories(file_path: str) -> List[Dict]:
    """Load trajectories from JSON or pickle file."""
    path = Path(file_path)
    
    if path.suffix == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    elif path.suffix == '.jsonl':
        trajectories = []
        with open(path, 'r') as f:
            for line in f:
                trajectories.append(json.loads(line))
        return trajectories
    elif path.suffix in ['.pkl', '.pickle']:
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def extract_task_results(trajectories: List[Dict]) -> Dict[str, List[bool]]:
    """
    Extract success/failure results for each task.
    
    Returns:
        Dict mapping task_id to list of success indicators (True/False)
    """
    task_results = defaultdict(list)
    
    for traj in trajectories:
        # Get task_id
        task_id = traj.get('task_id')
        if task_id is None:
            task_id = traj.get('id') or traj.get('task')
        
        # Try different possible key names for success indicator
        success = None
        
        # First check for reward (tau-bench format)
        if 'reward' in traj:
            reward = traj['reward']
            success = (reward == 1.0)  # Success if reward is 1.0
        elif 'success' in traj:
            success = traj['success']
        elif 'passed' in traj:
            success = traj['passed']
        elif 'correct' in traj:
            success = traj['correct']
        elif 'is_correct' in traj:
            success = traj['is_correct']
        elif 'result' in traj:
            result = traj['result']
            if isinstance(result, bool):
                success = result
            elif isinstance(result, str):
                success = result.lower() in ['success', 'passed', 'correct', 'true']
        
        if task_id is not None and success is not None:
            task_results[task_id].append(success)
    
    return task_results


def calculate_success_rates(task_results: Dict[str, List[bool]]) -> Dict[str, float]:
    """Calculate success rate for each task."""
    success_rates = {}
    for task_id, results in task_results.items():
        success_rates[task_id] = sum(results) / len(results) if results else 0.0
    return success_rates


def plot_success_rate_histogram(success_rates: Dict[str, float], 
                                output_path: str = '/mnt/user-data/outputs/success_rate_histogram.png',
                                bins: int = 20):
    """Create and save histogram of success rates."""
    rates = list(success_rates.values())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create histogram
    n, bins_edges, patches = ax.hist(rates, bins=bins, edgecolor='black', alpha=0.7)
    
    # Color code the bars
    for i, patch in enumerate(patches):
        rate = (bins_edges[i] + bins_edges[i+1]) / 2
        if rate >= 0.8:
            patch.set_facecolor('green')
        elif rate >= 0.5:
            patch.set_facecolor('yellow')
        else:
            patch.set_facecolor('red')
    
    ax.set_xlabel('Success Rate', fontsize=12)
    ax.set_ylabel('Number of Tasks', fontsize=12)
    ax.set_title(f'Distribution of Success Rates Across Tasks (n={len(success_rates)} tasks)', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics
    mean_rate = np.mean(rates)
    median_rate = np.median(rates)
    
    stats_text = f'Mean: {mean_rate:.2%}\nMedian: {median_rate:.2%}\n'
    stats_text += f'Tasks with 100% success: {sum(r == 1.0 for r in rates)}\n'
    stats_text += f'Tasks with 0% success: {sum(r == 0.0 for r in rates)}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to: {output_path}")
    
    return fig


def print_summary_statistics(success_rates: Dict[str, float]):
    """Print detailed summary statistics."""
    rates = list(success_rates.values())
    
    print("\n" + "="*60)
    print("SUCCESS RATE SUMMARY STATISTICS")
    print("="*60)
    print(f"Total number of tasks: {len(rates)}")
    print(f"Mean success rate: {np.mean(rates):.2%}")
    print(f"Median success rate: {np.median(rates):.2%}")
    print(f"Std deviation: {np.std(rates):.2%}")
    print(f"Min success rate: {np.min(rates):.2%}")
    print(f"Max success rate: {np.max(rates):.2%}")
    
    print("\nSuccess rate distribution:")
    print(f"  Tasks with 100% success: {sum(r == 1.0 for r in rates)}")
    print(f"  Tasks with ≥80% success: {sum(r >= 0.8 for r in rates)}")
    print(f"  Tasks with 50-80% success: {sum(0.5 <= r < 0.8 for r in rates)}")
    print(f"  Tasks with <50% success: {sum(r < 0.5 for r in rates)}")
    print(f"  Tasks with 0% success: {sum(r == 0.0 for r in rates)}")
    
    # List tasks by category
    print("\n" + "-"*60)
    print("TASKS BY SUCCESS RATE CATEGORY")
    print("-"*60)
    
    perfect_tasks = [task for task, rate in success_rates.items() if rate == 1.0]
    if perfect_tasks:
        print(f"\nPerfect tasks (100% success, n={len(perfect_tasks)}):")
        for task in sorted(perfect_tasks)[:10]:  # Show first 10
            print(f"  - {task}")
        if len(perfect_tasks) > 10:
            print(f"  ... and {len(perfect_tasks) - 10} more")
    
    failed_tasks = [task for task, rate in success_rates.items() if rate == 0.0]
    if failed_tasks:
        print(f"\nFailed tasks (0% success, n={len(failed_tasks)}):")
        for task in sorted(failed_tasks)[:10]:  # Show first 10
            print(f"  - {task}")
        if len(failed_tasks) > 10:
            print(f"  ... and {len(failed_tasks) - 10} more")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze tau-bench trajectory success rates')
    parser.add_argument('trajectory_input', nargs='+', 
                       help='Path to trajectory files or directory containing JSON files')
    parser.add_argument('--bins', type=int, default=20, help='Number of bins for histogram')
    parser.add_argument('--output', type=str, default='/mnt/user-data/outputs/success_rate_histogram.png',
                       help='Output path for histogram')
    
    args = parser.parse_args()
    
    # Collect all trajectory files
    trajectory_files = []
    for input_path in args.trajectory_input:
        path = Path(input_path)
        if path.is_dir():
            # Find all JSON files in directory
            json_files = list(path.glob('*.json')) + list(path.glob('*.jsonl'))
            trajectory_files.extend(json_files)
            print(f"Found {len(json_files)} JSON files in directory: {input_path}")
        elif path.is_file():
            trajectory_files.append(path)
        else:
            print(f"Warning: {input_path} not found, skipping...")
    
    if not trajectory_files:
        print("Error: No trajectory files found!")
        return
    
    print(f"\nTotal files to process: {len(trajectory_files)}")
    
    # Load all trajectories from all runs
    all_trajectories = []
    for file_path in trajectory_files:
        print(f"Loading trajectories from: {file_path}")
        trajectories = load_trajectories(str(file_path))
        all_trajectories.extend(trajectories)
        print(f"  Loaded {len(trajectories)} trajectories")
    
    print(f"\nTotal trajectories across all runs: {len(all_trajectories)}")
    
    # Extract results per task
    task_results = extract_task_results(all_trajectories)
    print(f"Found {len(task_results)} unique tasks")
    
    # Calculate success rates
    success_rates = calculate_success_rates(task_results)
    
    # Print statistics
    print_summary_statistics(success_rates)
    
    # Create histogram
    plot_success_rate_histogram(success_rates, output_path=args.output, bins=args.bins)
    
    # Save detailed results to CSV for further analysis
    csv_path = args.output.replace('.png', '_detailed.csv')
    with open(csv_path, 'w') as f:
        f.write("task_id,success_rate,num_runs,num_successes\n")
        for task_id in sorted(success_rates.keys()):
            rate = success_rates[task_id]
            results = task_results[task_id]
            f.write(f"{task_id},{rate:.4f},{len(results)},{sum(results)}\n")
    print(f"Detailed results saved to: {csv_path}")


if __name__ == '__main__':
    main()