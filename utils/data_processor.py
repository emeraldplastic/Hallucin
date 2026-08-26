"""
Data Processor for Hallucinogen benchmark.
Handles data loading, preprocessing, and augmentation for evaluation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from collections import defaultdict
import random

logger = logging.getLogger(__name__)

@dataclass
class DataSample:
    """Represents a single data sample."""
    image: str
    text: str
    label: str
    task_type: str
    split: str
    metadata: Dict[str, Any]

@dataclass
class DatasetStats:
    """Statistics about a dataset."""
    total_samples: int
    samples_by_task_type: Dict[str, int]
    samples_by_split: Dict[str, int]
    label_distribution: Dict[str, int]
    average_text_length: float

class DataProcessor:
    """Processor for Hallucinogen benchmark data."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.samples: List[DataSample] = []
        self.task_types = ["counterfactual", "identification", "localization", "visual_context"]
        self.splits = ["train", "val", "test"]
    
    def load_data(self, split: Optional[str] = None) -> List[DataSample]:
        """
        Load data from JSONL files.
        
        Args:
            split: Optional split to load (train, val, test)
        
        Returns:
            List of DataSample objects
        """
        self.samples = []
        
        # Determine which files to load
        if split:
            splits_to_load = [split]
        else:
            splits_to_load = self.splits
        
        for split_name in splits_to_load:
            data_file = self.data_dir / f"{split_name}.jsonl"
            
            if not data_file.exists():
                logger.warning(f"Data file not found: {data_file}")
                continue
            
            with open(data_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        
                        sample = DataSample(
                            image=data.get('image', ''),
                            text=data.get('text', ''),
                            label=data.get('label', 'no'),
                            task_type=data.get('task_type', 'identification'),
                            split=split_name,
                            metadata=data.get('metadata', {})
                        )
                        
                        self.samples.append(sample)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing line: {e}")
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(splits_to_load)} split(s)")
        
        return self.samples
    
    def filter_by_task_type(self, task_type: str) -> List[DataSample]:
        """
        Filter samples by task type.
        
        Args:
            task_type: Task type to filter by
        
        Returns:
            Filtered list of DataSample objects
        """
        return [s for s in self.samples if s.task_type == task_type]
    
    def filter_by_split(self, split: str) -> List[DataSample]:
        """
        Filter samples by split.
        
        Args:
            split: Split to filter by
        
        Returns:
            Filtered list of DataSample objects
        """
        return [s for s in self.samples if s.split == split]
    
    def filter_by_label(self, label: str) -> List[DataSample]:
        """
        Filter samples by label.
        
        Args:
            label: Label to filter by
        
        Returns:
            Filtered list of DataSample objects
        """
        return [s for s in self.samples if s.label == label]
    
    def get_statistics(self) -> DatasetStats:
        """
        Calculate dataset statistics.
        
        Returns:
            DatasetStats object
        """
        if not self.samples:
            return DatasetStats(
                total_samples=0,
                samples_by_task_type={},
                samples_by_split={},
                label_distribution={},
                average_text_length=0.0
            )
        
        # Count by task type
        by_task = defaultdict(int)
        for sample in self.samples:
            by_task[sample.task_type] += 1
        
        # Count by split
        by_split = defaultdict(int)
        for sample in self.samples:
            by_split[sample.split] += 1
        
        # Count by label
        by_label = defaultdict(int)
        for sample in self.samples:
            by_label[sample.label] += 1
        
        # Calculate average text length
        avg_length = sum(len(s.text) for s in self.samples) / len(self.samples)
        
        return DatasetStats(
            total_samples=len(self.samples),
            samples_by_task_type=dict(by_task),
            samples_by_split=dict(by_split),
            label_distribution=dict(by_label),
            average_text_length=avg_length
        )
    
    def create_balanced_subset(
        self,
        task_type: str,
        samples_per_label: int = 100,
        random_seed: int = 42
    ) -> List[DataSample]:
        """
        Create a balanced subset of samples.
        
        Args:
            task_type: Task type to sample from
            samples_per_label: Number of samples per label
            random_seed: Random seed for reproducibility
        
        Returns:
            Balanced list of DataSample objects
        """
        random.seed(random_seed)
        
        # Filter by task type
        task_samples = self.filter_by_task_type(task_type)
        
        # Separate by label
        yes_samples = [s for s in task_samples if s.label == 'yes']
        no_samples = [s for s in task_samples if s.label == 'no']
        
        # Sample from each label
        sampled_yes = random.sample(yes_samples, min(samples_per_label, len(yes_samples)))
        sampled_no = random.sample(no_samples, min(samples_per_label, len(no_samples)))
        
        balanced = sampled_yes + sampled_no
        random.shuffle(balanced)
        
        logger.info(f"Created balanced subset: {len(balanced)} samples ({len(sampled_yes)} yes, {len(sampled_no)} no)")
        
        return balanced
    
    def augment_text(
        self,
        sample: DataSample,
        augmentation_type: str = "paraphrase"
    ) -> DataSample:
        """
        Augment text for a sample.
        
        Args:
            sample: DataSample to augment
            augmentation_type: Type of augmentation (paraphrase, synonym, etc.)
        
        Returns:
            Augmented DataSample
        """
        # In production, this would use NLP models for actual augmentation
        # For now, we'll do simple rule-based augmentation
        
        if augmentation_type == "paraphrase":
            # Simple paraphrasing by adding/removing words
            words = sample.text.split()
            if len(words) > 5:
                # Remove a random word
                idx = random.randint(0, len(words) - 1)
                words.pop(idx)
                augmented_text = ' '.join(words)
            else:
                augmented_text = sample.text
        
        elif augmentation_type == "case_change":
            # Change case
            augmented_text = sample.text.swapcase()
        
        else:
            augmented_text = sample.text
        
        # Create new sample with augmented text
        augmented_sample = DataSample(
            image=sample.image,
            text=augmented_text,
            label=sample.label,
            task_type=sample.task_type,
            split=sample.split,
            metadata={**sample.metadata, 'augmented': True, 'augmentation_type': augmentation_type}
        )
        
        return augmented_sample
    
    def split_train_val_test(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[List[DataSample], List[DataSample], List[DataSample]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            random_seed: Random seed for reproducibility
        
        Returns:
            Tuple of (train, val, test) lists
        """
        random.seed(random_seed)
        
        # Shuffle samples
        shuffled = self.samples.copy()
        random.shuffle(shuffled)
        
        # Calculate split points
        total = len(shuffled)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]
        
        # Update split labels
        for sample in train:
            sample.split = "train"
        for sample in val:
            sample.split = "val"
        for sample in test:
            sample.split = "test"
        
        logger.info(f"Split data: {len(train)} train, {len(val)} val, {len(test)} test")
        
        return train, val, test
    
    def save_data(self, samples: List[DataSample], output_file: str):
        """
        Save samples to a JSONL file.
        
        Args:
            samples: List of DataSample objects to save
            output_file: Path to output file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for sample in samples:
                data = {
                    'image': sample.image,
                    'text': sample.text,
                    'label': sample.label,
                    'task_type': sample.task_type,
                    'split': sample.split,
                    'metadata': sample.metadata
                }
                f.write(json.dumps(data) + '\n')
        
        logger.info(f"Saved {len(samples)} samples to {output_file}")
    
    def merge_datasets(self, data_files: List[str]) -> List[DataSample]:
        """
        Merge multiple datasets.
        
        Args:
            data_files: List of data file paths to merge
        
        Returns:
            Merged list of DataSample objects
        """
        merged_samples = []
        
        for data_file in data_files:
            data_path = Path(data_file)
            if not data_path.exists():
                logger.warning(f"Data file not found: {data_file}")
                continue
            
            with open(data_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        
                        sample = DataSample(
                            image=data.get('image', ''),
                            text=data.get('text', ''),
                            label=data.get('label', 'no'),
                            task_type=data.get('task_type', 'identification'),
                            split=data.get('split', 'unknown'),
                            metadata=data.get('metadata', {})
                        )
                        
                        merged_samples.append(sample)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing line: {e}")
        
        logger.info(f"Merged {len(merged_samples)} samples from {len(data_files)} files")
        
        return merged_samples
    
    def validate_data(self, samples: List[DataSample]) -> Dict[str, Any]:
        """
        Validate data samples.
        
        Args:
            samples: List of DataSample objects to validate
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        for sample in samples:
            # Check required fields
            if not sample.image:
                issues.append(f"Missing image for sample")
            
            if not sample.text:
                issues.append(f"Missing text for sample")
            
            if sample.label not in ['yes', 'no']:
                issues.append(f"Invalid label: {sample.label}")
            
            if sample.task_type not in self.task_types:
                issues.append(f"Invalid task type: {sample.task_type}")
        
        # Check for duplicates
        text_set = set()
        duplicates = 0
        for sample in samples:
            if sample.text in text_set:
                duplicates += 1
            text_set.add(sample.text)
        
        return {
            'total_samples': len(samples),
            'issues_found': len(issues),
            'issue_details': issues[:10],  # First 10 issues
            'duplicate_texts': duplicates
        }

# Global data processor instance
data_processor = None

def get_data_processor(data_dir: str) -> DataProcessor:
    """Get or create a data processor instance."""
    global data_processor
    if data_processor is None:
        data_processor = DataProcessor(data_dir)
    return data_processor

def test_data_processor():
    """Test the data processor."""
    # Create a temporary data directory with sample data
    import tempfile
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir)
    
    # Create sample data
    sample_data = [
        {
            'image': 'image1.jpg',
            'text': 'Is there a cat in the image?',
            'label': 'yes',
            'task_type': 'identification',
            'split': 'train',
            'metadata': {}
        },
        {
            'image': 'image2.jpg',
            'text': 'Is there a dog in the image?',
            'label': 'no',
            'task_type': 'identification',
            'split': 'train',
            'metadata': {}
        }
    ]
    
    # Write sample data
    data_file = data_dir / "train.jsonl"
    with open(data_file, 'w') as f:
        for data in sample_data:
            f.write(json.dumps(data) + '\n')
    
    # Test processor
    processor = DataProcessor(str(data_dir))
    samples = processor.load_data()
    
    print(f"Loaded {len(samples)} samples")
    
    stats = processor.get_statistics()
    print(f"Statistics: {stats}")
    
    # Validate
    validation = processor.validate_data(samples)
    print(f"Validation: {validation}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_data_processor()
