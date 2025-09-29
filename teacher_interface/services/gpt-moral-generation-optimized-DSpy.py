#!/usr/bin/env python3
"""
Optimized Moral Generation with DSPy Feedback Learning
This script provides moral generation with DSPy optimization based on teacher feedback.
"""

import dspy
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure DSPy
lm = dspy.OpenAI(model="gpt-4o", max_tokens=1000, temperature=0.7)
dspy.settings.configure(lm=lm)

class MoralExtractorSignature(dspy.Signature):
    """Signature for extracting moral lessons from story content."""
    story: str = dspy.InputField(desc="The complete story text")
    moral: str = dspy.OutputField(desc="A clear, age-appropriate moral lesson that can be taught to children")
    segments: str = dspy.OutputField(desc="JSON array of story segments with SUMMARY, REASONING, START, END fields")

class MoralExtractor(dspy.Module):
    """DSPy module for extracting moral lessons from stories."""
    
    def __init__(self):
        super().__init__()
        self.moral_extractor = dspy.Predict(MoralExtractorSignature)
    
    def forward(self, story: str) -> Dict[str, Any]:
        """Extract moral lesson and segments from story."""
        try:
            result = self.moral_extractor(story=story)
            
            # Parse segments
            segments = result.segments
            if isinstance(segments, str):
                try:
                    import re
                    json_match = re.search(r'\[.*\]', segments, re.DOTALL)
                    if json_match:
                        segments = json.loads(json_match.group())
                    else:
                        segments = self._parse_segments_from_text(segments)
                except:
                    segments = self._parse_segments_from_text(segments)
            
            # Add name field to each segment and limit to 7 segments
            if isinstance(segments, list):
                for i, segment in enumerate(segments[:7], 1):
                    segment['name'] = f"segment_{i}"
            
            return {
                "moral": result.moral,
                "segments": segments[:7] if isinstance(segments, list) else segments
            }
        except Exception as e:
            print(f"Error in moral extraction: {e}")
            return None
    
    def _parse_segments_from_text(self, text: str) -> List[Dict]:
        """Parse segments from text output if JSON parsing fails."""
        segments = []
        lines = text.split('\n')
        current_segment = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('SUMMARY:'):
                current_segment['SUMMARY'] = line.replace('SUMMARY:', '').strip()
            elif line.startswith('REASONING:'):
                current_segment['REASONING'] = line.replace('REASONING:', '').strip()
            elif line.startswith('START:'):
                current_segment['START'] = int(line.replace('START:', '').strip())
            elif line.startswith('END:'):
                current_segment['END'] = int(line.replace('END:', '').strip())
                if len(current_segment) == 4:
                    segments.append(current_segment)
                    current_segment = {}
        
        return segments

class MoralFeedbackCollector:
    """Collects and stores moral feedback for DSPy optimization."""
    
    def __init__(self, feedback_file: str = "moral_feedback_dataset.json"):
        self.feedback_file = feedback_file
        self.load_feedback_data()
    
    def load_feedback_data(self):
        """Load existing feedback data."""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    self.feedback_data = json.load(f)
            except:
                self.feedback_data = {}
        else:
            self.feedback_data = {}
    
    def save_feedback_data(self):
        """Save feedback data to file."""
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_data, f, indent=2, ensure_ascii=False)
    
    def record_positive_feedback(self, story_title: str, inputs: Dict, outputs: Dict):
        """Record positive feedback for a moral generation."""
        if story_title not in self.feedback_data:
            self.feedback_data[story_title] = {
                "positive_examples": [],
                "negative_examples": [],
                "last_updated": datetime.now().isoformat()
            }
        
        self.feedback_data[story_title]["positive_examples"].append({
            "inputs": inputs,
            "outputs": outputs,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 20 positive examples
        if len(self.feedback_data[story_title]["positive_examples"]) > 20:
            self.feedback_data[story_title]["positive_examples"] = self.feedback_data[story_title]["positive_examples"][-20:]
        
        self.save_feedback_data()
        print(f"Positive feedback recorded for {story_title}!")
    
    def record_negative_feedback(self, story_title: str, inputs: Dict, outputs: Dict, reason: str = ""):
        """Record negative feedback for a moral generation."""
        if story_title not in self.feedback_data:
            self.feedback_data[story_title] = {
                "positive_examples": [],
                "negative_examples": [],
                "last_updated": datetime.now().isoformat()
            }
        
        self.feedback_data[story_title]["negative_examples"].append({
            "inputs": inputs,
            "outputs": outputs,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 10 negative examples
        if len(self.feedback_data[story_title]["negative_examples"]) > 10:
            self.feedback_data[story_title]["negative_examples"] = self.feedback_data[story_title]["negative_examples"][-10:]
        
        self.save_feedback_data()
        print(f"Negative feedback recorded for {story_title}!")
    
    def get_feedback_examples(self, story_title: str) -> List[Dict]:
        """Get feedback examples for a specific story."""
        if story_title not in self.feedback_data:
            return []
        
        examples = []
        for example in self.feedback_data[story_title]["positive_examples"]:
            examples.append({
                "inputs": example["inputs"],
                "outputs": example["outputs"],
                "is_positive": True
            })
        
        for example in self.feedback_data[story_title]["negative_examples"]:
            examples.append({
                "inputs": example["inputs"],
                "outputs": example["outputs"],
                "is_positive": False,
                "reason": example.get("reason", "")
            })
        
        return examples

class OptimizedMoralGenerator:
    """DSPy-based moral generator with feedback learning."""
    
    def __init__(self, feedback_file: str = "moral_feedback_dataset.json"):
        self.moral_extractor = MoralExtractor()
        self.feedback_collector = MoralFeedbackCollector(feedback_file)
        self.optimizer = None
        self.optimized_generator = None
    
    def generate_moral(self, story: str, story_title: str = "") -> Dict[str, Any]:
        """Generate moral lesson with optional optimization."""
        try:
            # Check if we have enough feedback examples for optimization
            feedback_examples = self.feedback_collector.get_feedback_examples(story_title)
            positive_examples = [ex for ex in feedback_examples if ex["is_positive"]]
            
            if len(positive_examples) >= 3:
                # Use optimized generator
                if not self.optimized_generator:
                    self._create_optimized_generator(positive_examples)
                
                if self.optimized_generator:
                    result = self.optimized_generator(story=story)
                    return self._process_result(result)
            
            # Use basic generator
            return self.moral_extractor.forward(story)
            
        except Exception as e:
            print(f"Error generating moral: {e}")
            return None
    
    def _create_optimized_generator(self, examples: List[Dict]):
        """Create optimized generator using BootstrapFewShot."""
        try:
            # Create training examples
            trainset = []
            for example in examples:
                trainset.append(dspy.Example(
                    story=example["inputs"]["story"],
                    moral=example["outputs"]["moral"],
                    segments=json.dumps(example["outputs"]["segments"])
                ).with_inputs("story"))
            
            # Create optimizer
            self.optimizer = dspy.BootstrapFewShot(metric=None, max_bootstrapped_demos=4)
            
            # Optimize the generator
            self.optimized_generator = self.optimizer.compile(
                self.moral_extractor,
                trainset=trainset
            )
            
            print(f"Created optimized moral generator with {len(examples)} examples!")
            
        except Exception as e:
            print(f"Error creating optimized generator: {e}")
            self.optimized_generator = None
    
    def _process_result(self, result) -> Dict[str, Any]:
        """Process DSPy result into expected format."""
        try:
            segments = result.segments
            if isinstance(segments, str):
                try:
                    segments = json.loads(segments)
                except:
                    segments = self.moral_extractor._parse_segments_from_text(segments)
            
            # Add name field to each segment
            if isinstance(segments, list):
                for i, segment in enumerate(segments[:7], 1):
                    segment['name'] = f"segment_{i}"
            
            return {
                "moral": result.moral,
                "segments": segments[:7] if isinstance(segments, list) else segments
            }
        except Exception as e:
            print(f"Error processing result: {e}")
            return None
    
    def record_feedback(self, story_title: str, story: str, moral: str, segments: List[Dict], is_positive: bool, reason: str = ""):
        """Record feedback for moral generation."""
        inputs = {"story": story}
        outputs = {"moral": moral, "segments": segments}
        
        if is_positive:
            self.feedback_collector.record_positive_feedback(story_title, inputs, outputs)
        else:
            self.feedback_collector.record_negative_feedback(story_title, inputs, outputs, reason)

def main():
    """Main function for testing."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gpt-moral-generation-optimized-DSpy.py <story_content> [story_title]")
        sys.exit(1)
    
    story_content = sys.argv[1]
    story_title = sys.argv[2] if len(sys.argv) > 2 else "test_story"
    
    generator = OptimizedMoralGenerator()
    
    # Generate moral
    result = generator.generate_moral(story_content, story_title)
    
    if result:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("Failed to generate moral lesson")

if __name__ == "__main__":
    main()
