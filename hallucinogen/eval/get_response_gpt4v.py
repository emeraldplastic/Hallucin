"""
Evaluation script for GPT-4V (GPT-4 Vision) on Hallucinogen benchmark.
Tests object hallucination detection capabilities of GPT-4V.
"""

import os
import json
import argparse
import tqdm
import time
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Model configuration
MODEL_PATH = "gpt-4-vision-preview"
API_KEY_ENV = "OPENAI_API_KEY"

# Prompt templates for different task types
COUNTERFACTUAL_PROMPTS = [
    "How would the scene change if the <obj> were invisible? Provide a concise answer",
    "What might replace the <obj> if it vanished from the image? Provide a concise answer",
]

IDENTIFICATION_PROMPTS = [
    'Is the <obj> present in the image?',
    'Determine whether a <obj> is visible in the image.',
]

LOCALIZATION_PROMPTS = [
    'Describe where is <obj> in the image.',
    'Examine the presence and location of <obj> in the image.',
]

VISUAL_CONTEXT_PROMPTS = [
    'Interpret the object <obj> with respect to its context within the image. Provide a concise answer',
    'Analyze the neighboring elements of <obj> in the image. Provide a concise answer',
]

class GPT4VEvaluator:
    """Evaluator for GPT-4V on Hallucinogen benchmark."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the evaluator with OpenAI API key."""
        self.api_key = api_key or os.environ.get(API_KEY_ENV)
        if not self.api_key:
            raise ValueError(f"OpenAI API key not found. Set {API_KEY_ENV} environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.results = []
        self.errors = []
    
    def generate_output(
        self,
        image_path: str,
        prompt: str,
        label: Optional[str] = None,
        max_tokens: int = 256
    ) -> Tuple[str, Optional[str]]:
        """
        Generate response from GPT-4V for a given image and prompt.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt to send with the image
            label: Ground truth label (for evaluation)
            max_tokens: Maximum tokens in response
        
        Returns:
            Tuple of (response, error_message)
        """
        try:
            # Read and encode image
            import base64
            
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create API request
            response = self.client.chat.completions.create(
                model=MODEL_PATH,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=0
            )
            
            output = response.choices[0].message.content
            return output, None
            
        except Exception as e:
            error_msg = f"Error processing {image_path}: {str(e)}"
            return None, error_msg
    
    def evaluate_file(
        self,
        data_file: str,
        image_dir: str,
        split_num: int,
        prompt_type: str = "counterfactual"
    ) -> List[Dict]:
        """
        Evaluate GPT-4V on a data file from Hallucinogen benchmark.
        
        Args:
            data_file: Path to JSON data file
            image_dir: Directory containing images
            split_num: Split number for output file naming
            prompt_type: Type of prompts to use (counterfactual, identification, localization, visual_context)
        
        Returns:
            List of evaluation results
        """
        # Select prompt template based on type
        if prompt_type == "counterfactual":
            prompts = COUNTERFACTUAL_PROMPTS
        elif prompt_type == "identification":
            prompts = IDENTIFICATION_PROMPTS
        elif prompt_type == "localization":
            prompts = LOCALIZATION_PROMPTS
        elif prompt_type == "visual_context":
            prompts = VISUAL_CONTEXT_PROMPTS
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        # Load data
        content = []
        with open(data_file) as file:
            for line in file:
                content.append(json.loads(line))
        
        results = []
        
        # Process each data point
        for data in tqdm.tqdm(content, desc=f"Evaluating {prompt_type}"):
            image_path = os.path.join(image_dir, data["image"])
            
            # Extract object name from text
            obj = ' '.join(data["text"].split(' ')[3:-3])
            
            # Generate responses for each prompt variant
            for i, prompt_template in enumerate(prompts):
                prompt = prompt_template.replace('<obj>', obj)
                
                response, error = self.generate_output(
                    image_path,
                    prompt,
                    data.get("label")
                )
                
                if error:
                    self.errors.append({
                        "image": data["image"],
                        "prompt": prompt,
                        "error": error
                    })
                    response = "ERROR: " + error
                
                # Store result
                result_data = data.copy()
                result_data[f'query_{i+1}'] = prompt
                result_data[f'output_{i+1}'] = response
                
                results.append(result_data)
        
        # Save results
        output_filename = f'gpt4v_{prompt_type}_response_{split_num}_' + os.path.basename(data_file)
        with open(output_filename, 'w') as file:
            for data in results:
                json.dump(data, file)
                file.write('\n')
        
        print(f"Saved results to {output_filename}")
        print(f"Processed {len(results)} queries")
        print(f"Errors: {len(self.errors)}")
        
        return results
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """
        Analyze evaluation results and compute metrics.
        
        Args:
            results: List of evaluation results
        
        Returns:
            Dictionary with analysis metrics
        """
        analysis = {
            "total_queries": len(results),
            "successful_queries": 0,
            "failed_queries": 0,
            "response_lengths": [],
            "error_types": {}
        }
        
        for result in results:
            # Check for errors
            for key in result.keys():
                if key.startswith('output_'):
                    output = result[key]
                    if isinstance(output, str) and output.startswith("ERROR:"):
                        analysis["failed_queries"] += 1
                        error_type = output.split(":")[1].strip() if ":" in output else "unknown"
                        analysis["error_types"][error_type] = analysis["error_types"].get(error_type, 0) + 1
                    else:
                        analysis["successful_queries"] += 1
                        analysis["response_lengths"].append(len(output))
        
        # Compute statistics
        if analysis["response_lengths"]:
            analysis["avg_response_length"] = np.mean(analysis["response_lengths"])
            analysis["median_response_length"] = np.median(analysis["response_lengths"])
            analysis["max_response_length"] = max(analysis["response_lengths"])
            analysis["min_response_length"] = min(analysis["response_lengths"])
        
        analysis["success_rate"] = analysis["successful_queries"] / analysis["total_queries"] if analysis["total_queries"] > 0 else 0
        
        return analysis
    
    def plot_analysis(self, results: List[Dict], save_path: Optional[str] = None):
        """
        Create visualization plots for the analysis.
        
        Args:
            results: List of evaluation results
            save_path: Optional path to save the plot
        """
        # Extract response lengths
        response_lengths = []
        for result in results:
            for key in result.keys():
                if key.startswith('output_') and isinstance(result[key], str) and not result[key].startswith("ERROR:"):
                    response_lengths.append(len(result[key]))
        
        if not response_lengths:
            print("No valid responses to plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Response length distribution
        axes[0, 0].hist(response_lengths, bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('Response Length')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Response Lengths')
        
        # Response length box plot
        axes[0, 1].boxplot(response_lengths)
        axes[0, 1].set_ylabel('Response Length')
        axes[0, 1].set_title('Response Length Box Plot')
        
        # Success rate pie chart
        successful = sum(1 for r in results for k in r.keys() if k.startswith('output_') and isinstance(r[k], str) and not r[k].startswith("ERROR:"))
        failed = sum(1 for r in results for k in r.keys() if k.startswith('output_') and isinstance(r[k], str) and r[k].startswith("ERROR:"))
        
        axes[1, 0].pie([successful, failed], labels=['Successful', 'Failed'], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        axes[1, 0].set_title('Query Success Rate')
        
        # Error types bar chart
        error_types = {}
        for result in results:
            for key in result.keys():
                if key.startswith('output_') and isinstance(result[key], str) and result[key].startswith("ERROR:"):
                    error_type = result[key].split(":")[1].strip() if ":" in result[key] else "unknown"
                    error_types[error_type] = error_types.get(error_type, 0) + 1
        
        if error_types:
            axes[1, 1].bar(error_types.keys(), error_types.values(), color='coral')
            axes[1, 1].set_xlabel('Error Type')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Error Types Distribution')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No errors', ha='center', va='center')
            axes[1, 1].set_title('Error Types Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-4V on Hallucinogen benchmark")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--split_num", type=int, required=True, help="Split number for output")
    parser.add_argument("--prompt_type", type=str, default="counterfactual", 
                       choices=["counterfactual", "identification", "localization", "visual_context"],
                       help="Type of prompts to use")
    parser.add_argument("--api_key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--plot", action="store_true", help="Generate analysis plots")
    parser.add_argument("--plot_path", type=str, help="Path to save analysis plot")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = GPT4VEvaluator(api_key=args.api_key)
    
    # Run evaluation
    print(f"Starting evaluation with {args.prompt_type} prompts...")
    results = evaluator.evaluate_file(
        args.file_path,
        args.image_dir,
        args.split_num,
        args.prompt_type
    )
    
    # Analyze results
    analysis = evaluator.analyze_results(results)
    print("\n=== Analysis Results ===")
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    # Generate plots if requested
    if args.plot:
        evaluator.plot_analysis(results, args.plot_path)

if __name__ == '__main__':
    main()
