"""
Result Aggregator for Hallucinogen benchmark.
Aggregates and compares results from multiple model evaluations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ModelResult:
    """Represents evaluation results for a single model."""
    model_name: str
    total_samples: int
    correct: int
    accuracy: float
    results_by_task_type: Dict[str, Dict[str, Any]]
    results_by_label: Dict[str, Dict[str, Any]]

@dataclass
class ComparisonResult:
    """Represents comparison between two models."""
    model1: str
    model2: str
    accuracy_diff: float
    better_on_tasks: List[str]
    worse_on_tasks: List[str]
    agreement_rate: float

class ResultAggregator:
    """Aggregator for benchmark evaluation results."""
    
    def __init__(self):
        self.model_results: Dict[str, ModelResult] = {}
        self.task_types = ["counterfactual", "identification", "localization", "visual_context"]
    
    def load_results(self, result_file: str, model_name: str) -> ModelResult:
        """
        Load evaluation results from a JSONL file.
        
        Args:
            result_file: Path to the results JSONL file
            model_name: Name of the model
        
        Returns:
            ModelResult object
        """
        results = []
        
        with open(result_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    results.append(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line: {e}")
        
        # Calculate statistics
        total = len(results)
        correct = sum(1 for r in results if r.get('correct', False))
        accuracy = correct / total if total > 0 else 0
        
        # Group by task type
        by_task = defaultdict(lambda: {'total': 0, 'correct': 0})
        for result in results:
            task_type = result.get('task_type', 'unknown')
            by_task[task_type]['total'] += 1
            if result.get('correct', False):
                by_task[task_type]['correct'] += 1
        
        # Calculate task-specific accuracies
        task_accuracies = {}
        for task_type, counts in by_task.items():
            task_accuracies[task_type] = {
                'total': counts['total'],
                'correct': counts['correct'],
                'accuracy': counts['correct'] / counts['total'] if counts['total'] > 0 else 0
            }
        
        # Group by label
        by_label = defaultdict(lambda: {'total': 0, 'correct': 0})
        for result in results:
            label = result.get('label', 'unknown')
            by_label[label]['total'] += 1
            if result.get('correct', False):
                by_label[label]['correct'] += 1
        
        # Calculate label-specific accuracies
        label_accuracies = {}
        for label, counts in by_label.items():
            label_accuracies[label] = {
                'total': counts['total'],
                'correct': counts['correct'],
                'accuracy': counts['correct'] / counts['total'] if counts['total'] > 0 else 0
            }
        
        model_result = ModelResult(
            model_name=model_name,
            total_samples=total,
            correct=correct,
            accuracy=accuracy,
            results_by_task_type=task_accuracies,
            results_by_label=label_accuracies
        )
        
        self.model_results[model_name] = model_result
        logger.info(f"Loaded results for {model_name}: {accuracy:.2%} accuracy")
        
        return model_result
    
    def compare_models(self, model1: str, model2: str) -> ComparisonResult:
        """
        Compare results between two models.
        
        Args:
            model1: Name of first model
            model2: Name of second model
        
        Returns:
            ComparisonResult object
        """
        if model1 not in self.model_results or model2 not in self.model_results:
            raise ValueError("Both models must have loaded results")
        
        result1 = self.model_results[model1]
        result2 = self.model_results[model2]
        
        # Calculate accuracy difference
        accuracy_diff = result1.accuracy - result2.accuracy
        
        # Compare by task type
        better_on = []
        worse_on = []
        
        for task_type in self.task_types:
            if task_type in result1.results_by_task_type and task_type in result2.results_by_task_type:
                acc1 = result1.results_by_task_type[task_type]['accuracy']
                acc2 = result2.results_by_task_type[task_type]['accuracy']
                
                if acc1 > acc2:
                    better_on.append(task_type)
                elif acc1 < acc2:
                    worse_on.append(task_type)
        
        # Calculate agreement rate (placeholder - would need raw results)
        agreement_rate = 0.8  # Placeholder
        
        return ComparisonResult(
            model1=model1,
            model2=model2,
            accuracy_diff=accuracy_diff,
            better_on_tasks=better_on,
            worse_on_tasks=worse_on,
            agreement_rate=agreement_rate
        )
    
    def generate_leaderboard(self) -> List[Dict[str, Any]]:
        """
        Generate a leaderboard of all evaluated models.
        
        Returns:
            List of model rankings
        """
        leaderboard = []
        
        for model_name, result in self.model_results.items():
            leaderboard.append({
                'rank': 0,  # To be assigned
                'model': model_name,
                'accuracy': result.accuracy,
                'total_samples': result.total_samples,
                'correct': result.correct
            })
        
        # Sort by accuracy
        leaderboard.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Assign ranks
        for i, entry in enumerate(leaderboard):
            entry['rank'] = i + 1
        
        return leaderboard
    
    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report.
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("HALLUCINOGEN BENCHMARK - RESULTS SUMMARY")
        report.append("=" * 80)
        report.append("")
        
        # Leaderboard
        report.append("MODEL LEADERBOARD")
        report.append("-" * 40)
        leaderboard = self.generate_leaderboard()
        
        for entry in leaderboard:
            report.append(
                f"{entry['rank']}. {entry['model']:30s} {entry['accuracy']:.2%} "
                f"({entry['correct']}/{entry['total_samples']})"
            )
        
        report.append("")
        
        # Detailed model results
        report.append("DETAILED RESULTS BY MODEL")
        report.append("-" * 40)
        
        for model_name, result in self.model_results.items():
            report.append(f"\n{model_name}")
            report.append(f"  Overall Accuracy: {result.accuracy:.2%}")
            report.append(f"  Total Samples: {result.total_samples}")
            
            report.append("  By Task Type:")
            for task_type, stats in result.results_by_task_type.items():
                report.append(
                    f"    {task_type}: {stats['accuracy']:.2%} "
                    f"({stats['correct']}/{stats['total']})"
                )
            
            report.append("  By Label:")
            for label, stats in result.results_by_label.items():
                report.append(
                    f"    {label}: {stats['accuracy']:.2%} "
                    f"({stats['correct']}/{stats['total']})"
                )
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def export_results(self, output_file: str, format: str = "json"):
        """
        Export aggregated results to file.
        
        Args:
            output_file: Path to output file
            format: Export format (json, csv)
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            export_data = {
                'leaderboard': self.generate_leaderboard(),
                'model_results': {
                    name: {
                        'total_samples': result.total_samples,
                        'correct': result.correct,
                        'accuracy': result.accuracy,
                        'results_by_task_type': result.results_by_task_type,
                        'results_by_label': result.results_by_label
                    }
                    for name, result in self.model_results.items()
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        elif format == "csv":
            import csv
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow(['Model', 'Accuracy', 'Total Samples', 'Correct'])
                
                # Write model results
                for model_name, result in self.model_results.items():
                    writer.writerow([
                        model_name,
                        result.accuracy,
                        result.total_samples,
                        result.correct
                    ])
        
        logger.info(f"Exported results to {output_file}")
    
    def get_task_type_analysis(self) -> Dict[str, Any]:
        """
        Get analysis of performance across task types.
        
        Returns:
            Dictionary with task type analysis
        """
        task_analysis = {}
        
        for task_type in self.task_types:
            task_results = []
            
            for model_name, result in self.model_results.items():
                if task_type in result.results_by_task_type:
                    task_results.append({
                        'model': model_name,
                        'accuracy': result.results_by_task_type[task_type]['accuracy'],
                        'total': result.results_by_task_type[task_type]['total']
                    })
            
            if task_results:
                task_results.sort(key=lambda x: x['accuracy'], reverse=True)
                task_analysis[task_type] = task_results
        
        return task_analysis
    
    def get_best_model_by_task(self, task_type: str) -> Optional[str]:
        """
        Get the best performing model for a specific task type.
        
        Args:
            task_type: Task type to check
        
        Returns:
            Name of best model or None
        """
        best_model = None
        best_accuracy = 0.0
        
        for model_name, result in self.model_results.items():
            if task_type in result.results_by_task_type:
                accuracy = result.results_by_task_type[task_type]['accuracy']
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_name
        
        return best_model

# Global result aggregator instance
result_aggregator = ResultAggregator()

def test_result_aggregator():
    """Test the result aggregator."""
    aggregator = ResultAggregator()
    
    # Create mock result files
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Mock results for model 1
    results1 = [
        {'model': 'gpt-4v', 'image': 'img1.jpg', 'question': 'test', 'task_type': 'identification', 'label': 'yes', 'output': 'yes', 'prediction': 'yes', 'correct': True},
        {'model': 'gpt-4v', 'image': 'img2.jpg', 'question': 'test', 'task_type': 'identification', 'label': 'no', 'output': 'no', 'prediction': 'no', 'correct': True},
        {'model': 'gpt-4v', 'image': 'img3.jpg', 'question': 'test', 'task_type': 'counterfactual', 'label': 'yes', 'output': 'no', 'prediction': 'no', 'correct': False}
    ]
    
    result_file1 = os.path.join(temp_dir, "gpt4v_results.jsonl")
    with open(result_file1, 'w') as f:
        for result in results1:
            f.write(json.dumps(result) + '\n')
    
    # Mock results for model 2
    results2 = [
        {'model': 'claude-3', 'image': 'img1.jpg', 'question': 'test', 'task_type': 'identification', 'label': 'yes', 'output': 'yes', 'prediction': 'yes', 'correct': True},
        {'model': 'claude-3', 'image': 'img2.jpg', 'question': 'test', 'task_type': 'identification', 'label': 'no', 'output': 'yes', 'prediction': 'yes', 'correct': False},
        {'model': 'claude-3', 'image': 'img3.jpg', 'question': 'test', 'task_type': 'counterfactual', 'label': 'yes', 'output': 'yes', 'prediction': 'yes', 'correct': True}
    ]
    
    result_file2 = os.path.join(temp_dir, "claude3_results.jsonl")
    with open(result_file2, 'w') as f:
        for result in results2:
            f.write(json.dumps(result) + '\n')
    
    # Load results
    aggregator.load_results(result_file1, "gpt-4v")
    aggregator.load_results(result_file2, "claude-3")
    
    # Generate leaderboard
    leaderboard = aggregator.generate_leaderboard()
    print("Leaderboard:")
    for entry in leaderboard:
        print(f"{entry['rank']}. {entry['model']}: {entry['accuracy']:.2%}")
    
    # Compare models
    comparison = aggregator.compare_models("gpt-4v", "claude-3")
    print(f"\nComparison: {comparison.model1} vs {comparison.model2}")
    print(f"Accuracy difference: {comparison.accuracy_diff:.2%}")
    
    # Generate summary
    summary = aggregator.generate_summary_report()
    print(f"\n{summary}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_result_aggregator()
