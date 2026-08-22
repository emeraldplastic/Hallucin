"""
Benchmark analyzer for Hallucinogen and MED-Hallucinogen.
Provides comprehensive analysis and visualization of benchmark results.
"""

import json
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import re

class BenchmarkAnalyzer:
    """Analyzer for Hallucinogen benchmark results."""
    
    def __init__(self, results_dir: str = "."):
        """Initialize the analyzer with results directory."""
        self.results_dir = Path(results_dir)
        self.results_data = []
        self.models = set()
        self.task_types = set()
        self.splits = set()
    
    def load_results(self, pattern: str = "*response*.jsonl") -> None:
        """
        Load benchmark results from JSONL files.
        
        Args:
            pattern: Glob pattern to match result files
        """
        for result_file in self.results_dir.glob(pattern):
            print(f"Loading {result_file}...")
            try:
                with open(result_file, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            data['source_file'] = str(result_file)
                            self.results_data.append(data)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
        
        print(f"Loaded {len(self.results_data)} results total")
        
        # Extract metadata
        for result in self.results_data:
            if 'model' in result:
                self.models.add(result['model'])
            if 'task_type' in result:
                self.task_types.add(result['task_type'])
            if 'split' in result:
                self.splits.add(result['split'])
    
    def compute_accuracy_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute accuracy metrics for each model and task type.
        
        Returns:
            Dictionary mapping model names to metric dictionaries
        """
        metrics = defaultdict(lambda: defaultdict(list))
        
        for result in self.results_data:
            model = result.get('model', 'unknown')
            task_type = result.get('task_type', 'unknown')
            label = result.get('label', '')
            
            # Extract predictions from output fields
            predictions = []
            for key in result.keys():
                if key.startswith('output_') and isinstance(result[key], str):
                    predictions.append(result[key])
            
            # Compute accuracy for each prediction
            for pred in predictions:
                # Simple accuracy: check if prediction contains expected answer
                if label.lower() in pred.lower():
                    accuracy = 1.0
                else:
                    accuracy = 0.0
                
                metrics[model][task_type].append(accuracy)
        
        # Aggregate metrics
        aggregated_metrics = {}
        for model, task_data in metrics.items():
            model_metrics = {}
            for task_type, accuracies in task_data.items():
                if accuracies:
                    model_metrics[task_type] = {
                        'mean': np.mean(accuracies),
                        'std': np.std(accuracies),
                        'count': len(accuracies)
                    }
            
            # Overall metrics
            all_accuracies = [acc for task_accs in task_data.values() for acc in task_accs]
            if all_accuracies:
                model_metrics['overall'] = {
                    'mean': np.mean(all_accuracies),
                    'std': np.std(all_accuracies),
                    'count': len(all_accuracies)
                }
            
            aggregated_metrics[model] = model_metrics
        
        return aggregated_metrics
    
    def compute_hallucination_rate(self) -> Dict[str, Dict[str, float]]:
        """
        Compute hallucination rates (false positives) for each model.
        
        Returns:
            Dictionary mapping model names to hallucination rate dictionaries
        """
        hallucination_rates = defaultdict(lambda: defaultdict(list))
        
        for result in self.results_data:
            model = result.get('model', 'unknown')
            task_type = result.get('task_type', 'unknown')
            label = result.get('label', '')
            
            # Check for hallucinations: model says "yes" when label is "no"
            for key in result.keys():
                if key.startswith('output_') and isinstance(result[key], str):
                    output = result[key].lower()
                    
                    # If ground truth is "no" but model indicates presence
                    if label.lower() == 'no':
                        if any(word in output for word in ['yes', 'present', 'visible', 'exists']):
                            hallucination_rates[model][task_type].append(1.0)  # Hallucination
                        else:
                            hallucination_rates[model][task_type].append(0.0)  # Correct rejection
        
        # Aggregate hallucination rates
        aggregated_rates = {}
        for model, task_data in hallucination_rates.items():
            model_rates = {}
            for task_type, rates in task_data.items():
                if rates:
                    model_rates[task_type] = {
                        'mean': np.mean(rates),
                        'std': np.std(rates),
                        'count': len(rates)
                    }
            
            # Overall rates
            all_rates = [rate for task_rates in task_data.values() for rate in task_rates]
            if all_rates:
                model_rates['overall'] = {
                    'mean': np.mean(all_rates),
                    'std': np.std(all_rates),
                    'count': len(all_rates)
                }
            
            aggregated_rates[model] = model_rates
        
        return aggregated_rates
    
    def analyze_response_patterns(self) -> Dict[str, Dict]:
        """
        Analyze response patterns and common phrases.
        
        Returns:
            Dictionary with pattern analysis results
        """
        patterns = defaultdict(lambda: defaultdict(int))
        response_lengths = defaultdict(list)
        
        for result in self.results_data:
            model = result.get('model', 'unknown')
            
            for key in result.keys():
                if key.startswith('output_') and isinstance(result[key], str):
                    output = result[key]
                    
                    # Track response length
                    response_lengths[model].append(len(output))
                    
                    # Extract common phrases (simple n-gram analysis)
                    words = output.lower().split()
                    for i in range(len(words) - 2):
                        trigram = ' '.join(words[i:i+3])
                        patterns[model][trigram] += 1
        
        # Get top patterns for each model
        top_patterns = {}
        for model, pattern_counts in patterns.items():
            sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
            top_patterns[model] = {
                'top_trigrams': sorted_patterns[:10],
                'avg_response_length': np.mean(response_lengths[model]) if response_lengths[model] else 0,
                'response_length_std': np.std(response_lengths[model]) if response_lengths[model] else 0
            }
        
        return top_patterns
    
    def generate_comparison_report(self) -> str:
        """
        Generate a comprehensive comparison report.
        
        Returns:
            Formatted report string
        """
        accuracy_metrics = self.compute_accuracy_metrics()
        hallucination_rates = self.compute_hallucination_rate()
        response_patterns = self.analyze_response_patterns()
        
        report = []
        report.append("=" * 80)
        report.append("HALLUCINOGEN BENCHMARK ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Total results analyzed: {len(self.results_data)}")
        report.append(f"Models evaluated: {len(self.models)}")
        report.append(f"Task types: {', '.join(sorted(self.task_types))}")
        report.append("")
        
        # Model comparison
        report.append("MODEL PERFORMANCE COMPARISON")
        report.append("-" * 80)
        
        for model in sorted(accuracy_metrics.keys()):
            report.append(f"\n{model.upper()}")
            report.append("-" * 40)
            
            model_acc = accuracy_metrics[model]
            model_halluc = hallucination_rates.get(model, {})
            model_patterns = response_patterns.get(model, {})
            
            # Overall metrics
            if 'overall' in model_acc:
                report.append(f"Overall Accuracy: {model_acc['overall']['mean']:.3f} ± {model_acc['overall']['std']:.3f}")
                report.append(f"Total Queries: {model_acc['overall']['count']}")
            
            if 'overall' in model_halluc:
                report.append(f"Hallucination Rate: {model_halluc['overall']['mean']:.3f} ± {model_halluc['overall']['std']:.3f}")
            
            # Task-specific metrics
            report.append("\nTask-Specific Performance:")
            for task_type in sorted(model_acc.keys()):
                if task_type == 'overall':
                    continue
                
                acc = model_acc[task_type]
                halluc = model_halluc.get(task_type, {})
                
                report.append(f"  {task_type}:")
                report.append(f"    Accuracy: {acc['mean']:.3f} ± {acc['std']:.3f} (n={acc['count']})")
                if halluc:
                    report.append(f"    Hallucination Rate: {halluc['mean']:.3f} ± {halluc['std']:.3f}")
            
            # Response patterns
            if model_patterns:
                report.append(f"\nResponse Statistics:")
                report.append(f"  Avg Response Length: {model_patterns['avg_response_length']:.1f} ± {model_patterns['response_length_std']:.1f}")
                report.append(f"  Top Phrases:")
                for phrase, count in model_patterns['top_trigrams'][:5]:
                    report.append(f"    - '{phrase}': {count} occurrences")
        
        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def plot_performance_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Create performance comparison plots.
        
        Args:
            save_path: Optional path to save the plot
        """
        accuracy_metrics = self.compute_accuracy_metrics()
        hallucination_rates = self.compute_hallucination_rate()
        
        if not accuracy_metrics:
            print("No accuracy metrics to plot")
            return
        
        # Prepare data for plotting
        models = list(accuracy_metrics.keys())
        task_types = set()
        for model_metrics in accuracy_metrics.values():
            task_types.update(model_metrics.keys())
        task_types.discard('overall')
        task_types = sorted(task_types)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy comparison bar chart
        if task_types:
            acc_data = []
            for model in models:
                for task_type in task_types:
                    if task_type in accuracy_metrics[model]:
                        acc_data.append({
                            'Model': model,
                            'Task': task_type,
                            'Accuracy': accuracy_metrics[model][task_type]['mean']
                        })
            
            if acc_data:
                df_acc = pd.DataFrame(acc_data)
                sns.barplot(data=df_acc, x='Task', y='Accuracy', hue='Model', ax=axes[0, 0])
                axes[0, 0].set_title('Accuracy by Task and Model')
                axes[0, 0].set_ylim(0, 1)
                axes[0, 0].legend(title='Model')
                axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Overall accuracy comparison
        overall_acc = [accuracy_metrics[model]['overall']['mean'] for model in models if 'overall' in accuracy_metrics[model]]
        overall_models = [model for model in models if 'overall' in accuracy_metrics[model]]
        
        if overall_acc:
            axes[0, 1].bar(overall_models, overall_acc, color='skyblue')
            axes[0, 1].set_title('Overall Accuracy Comparison')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Hallucination rate comparison
        if hallucination_rates:
            halluc_data = []
            for model in models:
                if model in hallucination_rates and 'overall' in hallucination_rates[model]:
                    halluc_data.append({
                        'Model': model,
                        'Hallucination Rate': hallucination_rates[model]['overall']['mean']
                    })
            
            if halluc_data:
                df_halluc = pd.DataFrame(halluc_data)
                sns.barplot(data=df_halluc, x='Model', y='Hallucination Rate', ax=axes[1, 0], color='coral')
                axes[1, 0].set_title('Hallucination Rate Comparison')
                axes[1, 0].set_ylim(0, 1)
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Accuracy vs Hallucination scatter plot
        if overall_acc and hallucination_rates:
            x_data = []
            y_data = []
            labels = []
            
            for model in models:
                if model in accuracy_metrics and 'overall' in accuracy_metrics[model]:
                    if model in hallucination_rates and 'overall' in hallucination_rates[model]:
                        x_data.append(accuracy_metrics[model]['overall']['mean'])
                        y_data.append(hallucination_rates[model]['overall']['mean'])
                        labels.append(model)
            
            if x_data:
                axes[1, 1].scatter(x_data, y_data, s=100, alpha=0.7)
                axes[1, 1].set_xlabel('Accuracy')
                axes[1, 1].set_ylabel('Hallucination Rate')
                axes[1, 1].set_title('Accuracy vs Hallucination Rate')
                axes[1, 1].set_xlim(0, 1)
                axes[1, 1].set_ylim(0, 1)
                
                # Add labels
                for i, label in enumerate(labels):
                    axes[1, 1].annotate(label, (x_data[i], y_data[i]), 
                                        xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_results_csv(self, output_path: str) -> None:
        """
        Export results to CSV format.
        
        Args:
            output_path: Path to save CSV file
        """
        # Flatten results for CSV export
        flat_results = []
        
        for result in self.results_data:
            flat_result = {
                'source_file': result.get('source_file', ''),
                'image': result.get('image', ''),
                'text': result.get('text', ''),
                'label': result.get('label', ''),
                'model': result.get('model', 'unknown'),
                'task_type': result.get('task_type', 'unknown')
            }
            
            # Add all output fields
            for key in result.keys():
                if key.startswith('output_') and isinstance(result[key], str):
                    flat_result[key] = result[key]
            
            flat_results.append(flat_result)
        
        df = pd.DataFrame(flat_results)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(flat_results)} results to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Hallucinogen benchmark results")
    parser.add_argument("--results_dir", type=str, default=".", help="Directory containing result files")
    parser.add_argument("--pattern", type=str, default="*response*.jsonl", help="Glob pattern for result files")
    parser.add_argument("--report", type=str, help="Path to save text report")
    parser.add_argument("--plot", type=str, help="Path to save comparison plot")
    parser.add_argument("--csv", type=str, help="Path to export results as CSV")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = BenchmarkAnalyzer(args.results_dir)
    
    # Load results
    analyzer.load_results(args.pattern)
    
    if not analyzer.results_data:
        print("No results loaded. Check your pattern and directory.")
        return
    
    # Generate and save report
    report = analyzer.generate_comparison_report()
    print(report)
    
    if args.report:
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {args.report}")
    
    # Generate plots
    if args.plot:
        analyzer.plot_performance_comparison(args.plot)
    
    # Export CSV
    if args.csv:
        analyzer.export_results_csv(args.csv)

if __name__ == "__main__":
    main()
