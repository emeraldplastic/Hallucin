"""
Gemini Pro Vision evaluation script for Hallucinogen benchmark.
Evaluates Google's Gemini Pro Vision model on object hallucination tasks.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import logging

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt templates for different task types
PROMPT_TEMPLATES = {
    "counterfactual": """
    You are given an image and a question about it. The question asks about a counterfactual scenario.
    Answer the question with "yes" or "no" based on the image content.
    
    Question: {question}
    
    Answer:""",
    
    "identification": """
    You are given an image and asked to identify if an object is present.
    Answer "yes" if the object is clearly visible in the image, "no" otherwise.
    
    Question: {question}
    
    Answer:""",
    
    "localization": """
    You are given an image and asked about the location of an object.
    Answer "yes" if the object is at the specified location, "no" otherwise.
    
    Question: {question}
    
    Answer:""",
    
    "visual_context": """
    You are given an image and a question requiring visual context understanding.
    Answer "yes" or "no" based on the image content.
    
    Question: {question}
    
    Answer:"""
}

class GeminiEvaluator:
    """Evaluator for Gemini Pro Vision on Hallucinogen benchmark."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro-vision"):
        """Initialize the Gemini evaluator."""
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
    
    def evaluate_single(
        self,
        image_path: str,
        question: str,
        task_type: str,
        label: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single image-question pair.
        
        Args:
            image_path: Path to the image file
            question: Question about the image
            task_type: Type of task (counterfactual, identification, localization, visual_context)
            label: Ground truth label (yes/no)
        
        Returns:
            Dictionary with evaluation results
        """
        # Load image
        try:
            import PIL.Image
            image = PIL.Image.open(image_path)
        except ImportError:
            raise ImportError("PIL/Pillow not installed. Install with: pip install Pillow")
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return {
                "error": str(e),
                "image": image_path,
                "question": question,
                "label": label
            }
        
        # Get prompt template
        template = PROMPT_TEMPLATES.get(task_type, PROMPT_TEMPLATES["identification"])
        prompt = template.format(question=question)
        
        # Generate response
        try:
            response = self.model.generate_content([prompt, image])
            output_text = response.text.strip().lower()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "error": str(e),
                "image": image_path,
                "question": question,
                "label": label
            }
        
        # Parse output
        prediction = "yes" if "yes" in output_text else "no"
        correct = prediction == label.lower()
        
        return {
            "model": self.model_name,
            "image": image_path,
            "question": question,
            "task_type": task_type,
            "label": label,
            "output": output_text,
            "prediction": prediction,
            "correct": correct
        }
    
    def evaluate_dataset(
        self,
        data_file: str,
        image_dir: str,
        output_file: str,
        task_type: str = None
    ):
        """
        Evaluate the entire dataset.
        
        Args:
            data_file: Path to the JSONL data file
            image_dir: Directory containing images
            output_file: Path to save results
            task_type: Optional task type filter
        """
        results = []
        
        with open(data_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                
                # Filter by task type if specified
                if task_type and data.get('task_type') != task_type:
                    continue
                
                # Construct image path
                image_name = data.get('image', '')
                image_path = os.path.join(image_dir, image_name)
                
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                # Evaluate
                result = self.evaluate_single(
                    image_path=image_path,
                    question=data.get('text', ''),
                    task_type=data.get('task_type', 'identification'),
                    label=data.get('label', 'no')
                )
                
                results.append(result)
                
                # Log progress
                if len(results) % 10 == 0:
                    logger.info(f"Processed {len(results)} samples")
        
        # Save results
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        logger.info(f"Saved {len(results)} results to {output_file}")
        
        # Print summary
        correct = sum(1 for r in results if r.get('correct', False))
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        
        logger.info(f"Accuracy: {correct}/{total} ({accuracy:.2%})")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemini Pro Vision on Hallucinogen")
    parser.add_argument("--api_key", type=str, required=True, help="Google API key")
    parser.add_argument("--data_file", type=str, required=True, help="Path to data JSONL file")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save results")
    parser.add_argument("--task_type", type=str, choices=["counterfactual", "identification", "localization", "visual_context"], help="Task type to evaluate")
    parser.add_argument("--model", type=str, default="gemini-pro-vision", help="Model name")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = GeminiEvaluator(api_key=args.api_key, model_name=args.model)
    
    # Run evaluation
    evaluator.evaluate_dataset(
        data_file=args.data_file,
        image_dir=args.image_dir,
        output_file=args.output_file,
        task_type=args.task_type
    )

if __name__ == "__main__":
    main()
