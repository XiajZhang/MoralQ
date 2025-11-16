"""
Generate objective-based questions using DSPy framework with teacher control and feedback learning.
"""

import dspy
import json
import os
import argparse
from typing import List, Dict, Any
from pathlib import Path

# Configure DSPy with OpenAI
dspy.configure(lm=dspy.LM("openai/gpt-4.1-2025-04-14"))

class ObjectiveQuestionSignature(dspy.Signature):
    """Generate questions aligned with a teacher-defined objective."""
    
    story = dspy.InputField(desc="Full story content, marked with page numbers.")
    segments = dspy.InputField(desc="Summarized segments of the story and reasoning.")
    objective = dspy.InputField(desc="Teacher-specified objective for the questions, e.g., empathy, kindness, curiosity, moral, comprehension.")
    story_title = dspy.InputField(desc="Title of the story.")
    
    questions = dspy.OutputField(desc="""A JSON array of 5-10 questions with the following fields:
    {
        "question": "...",
        "type": "comprehension/reflection/application/extension",
        "difficulty": "easy/medium/hard",
        "explanation": "...",
        "page_number": "integer representing the page number where this question should be discussed"
    }""")
    
    learning_objectives = dspy.OutputField(desc="A list of 3-4 learning objectives related to the story and given objective.")

class ObjectiveQuestionGenerator(dspy.Module):
    """Generate questions based on teacher-defined objectives using DSPy."""
    
    def __init__(self):
        super().__init__()
        self.generate_questions = dspy.Predict(ObjectiveQuestionSignature)
    
    def forward(self, story: str, segments: List[Dict], objective: str, story_title: str):
        """Generate questions based on the given objective."""
        
        # Format segments for the prompt with page information
        formatted_segments = "\n".join([
            f"Segment {i+1} (Pages {segment.get('START', '?')}-{segment.get('END', '?')}): {segment.get('SUMMARY', '')}"
            f"\nReasoning: {segment.get('REASONING', '')}"
            for i, segment in enumerate(segments)
        ])
        
        # Generate questions using DSPy
        result = self.generate_questions(
            story=story,
            segments=formatted_segments,
            objective=objective,
            story_title=story_title
        )
        
        return result

class FeedbackCollector:
    """Collect and manage teacher feedback for learning optimization."""
    
    def __init__(self, feedback_file: str = None):
        storage_dir = Path(__file__).resolve().parent.parent / "storage"
        os.makedirs(storage_dir, exist_ok=True)
        if feedback_file is None:
            feedback_file = storage_dir / "teacher_feedback_records.json"
        self.feedback_file = str(feedback_file)
        self.question_evaluations_file = str(storage_dir / "question_evaluations.json")
        self.feedback_data = self.load_feedback()
    
    def load_feedback(self) -> List[Dict]:
        """Load existing feedback data from teacher_feedback_records.json and question_evaluations.json."""
        feedback_records = []
        
        # Load teacher feedback records
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Parse hierarchical structure: school -> teacher -> records
                    for school_id, school_data in data.items():
                        if isinstance(school_data, dict):
                            for teacher_id, teacher_records in school_data.items():
                                if isinstance(teacher_records, list):
                                    for record in teacher_records:
                                        # Convert to DSPy training format
                                        dspy_record = self._convert_to_dspy_format(record)
                                        if dspy_record:
                                            feedback_records.append(dspy_record)
                
                print(f"[INFO] Loaded {len(feedback_records)} feedback records from {self.feedback_file}")
            except Exception as e:
                print(f"Error loading feedback data from {self.feedback_file}: {e}")
        
        # Load question evaluations
        question_evaluations = self._load_question_evaluations()
        if question_evaluations:
            print(f"[INFO] Loaded question evaluations: {len(question_evaluations)} records")
        
        return feedback_records
    
    def _load_question_evaluations(self) -> List[Dict]:
        """Load question evaluations from question_evaluations.json."""
        if os.path.exists(self.question_evaluations_file):
            try:
                with open(self.question_evaluations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("evaluations", [])
            except Exception as e:
                print(f"Error loading question evaluations: {e}")
        return []
    
    def _convert_to_dspy_format(self, record: Dict) -> Dict:
        """Convert teacher_feedback_records.json format to DSPy training format."""
        
        # Get story_title and objective
        story_title = record.get("story_title", "")
        objective = record.get("objective", "")
        
        # Extract inputs and outputs for DSPy
        inputs = {
            "story_title": story_title,
            "objective": objective,
        }
        
        outputs = {
            "questions": [],  # Will be populated from question_evaluations
            "evaluator_scores": record.get("evaluator_scores", {}),
        }
        
        # Check if this is a positive feedback record
        feedback = record.get("teacher_feedback", {})
        feedback_type = feedback.get("feedback", "")
        
        # Try to get story and segments from question_evaluations.json
        # by matching story_title
        question_evaluations = self._load_question_evaluations()
        
        # Find evaluations for this story
        story_evaluations = [
            eval_record for eval_record in question_evaluations
            if eval_record.get("storybook_id") == story_title or eval_record.get("story_title") == story_title
        ]
        
        # Extract questions from evaluations
        generated_questions = []
        story_content = ""
        segments_data = []
        
        if story_evaluations:
            # Get story content from the first evaluation
            story_content = story_evaluations[0].get("story_context", "")
            
            # Extract questions
            generated_questions = [
                eval_record.get("question", "") 
                for eval_record in story_evaluations 
                if eval_record.get("question")
            ]
            
            # For segments, we'll need to create a basic structure
            # This is a limitation - segments aren't stored in question_evaluations
            segments_data = [
                {
                    "name": f"Segment {i+1}",
                    "START": i,
                    "END": i+1,
                    "summary": f"Segment summary {i+1}",
                    "reasoning": "Generated from story"
                }
                for i in range(min(len(generated_questions), 3))
            ]
        
        # Add to inputs and outputs
        if story_content:
            inputs["story"] = story_content
        if segments_data:
            inputs["segments"] = segments_data
        
        if generated_questions:
            outputs["questions"] = generated_questions
        
        if feedback_type == "positive":
            return {
                "inputs": inputs,
                "outputs": outputs,
                "feedback_type": "positive",
                "iteration_id": record.get("iteration_id", ""),
                "course_of_action": record.get("course_of_action", {})
            }
        
        return None
    
    def save_feedback(self):
        """Save feedback data to file."""
        try:
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving feedback data: {e}")
    
    def add_positive_feedback(self, inputs: Dict, outputs: Dict, question_feedbacks: Dict = None, overall_feedback: str = ""):
        """Add positive feedback (liked question set) with individual question feedback."""
        objective = inputs.get("objective", "moral")
        story_title = inputs.get("story_title", "Unknown")
        
        # Count existing entries for this objective+story combination to determine iteration
        existing_entries = [entry for entry in self.feedback_data 
                          if entry.get("inputs", {}).get("objective") == objective and entry.get("inputs", {}).get("story_title") == story_title]
        iteration_number = len(existing_entries) + 1
        
        # Prepare individual question feedback
        individual_questions = {}
        if question_feedbacks:
            for i, question in enumerate(outputs.get("questions", [])):
                if i in question_feedbacks:
                    feedback_info = question_feedbacks[i]
                    individual_questions[str(i)] = {
                        "feedback": feedback_info["feedback"],
                        "reasoning": feedback_info["reasoning"]
                    }
        
        # Create DSPy-compatible structure
        feedback_entry = {
            "inputs": {
                "story": inputs.get("story", ""),
                "segments": inputs.get("segments", []),
                "objective": objective,
                "story_title": story_title
            },
            "outputs": {
                "questions": outputs.get("questions", []),
                "learning_objectives": outputs.get("learning_objectives", [])
            },
            "feedback": {
                "overall": "positive",
                "overall_reasoning": overall_feedback or "Teacher approved this question set",
                "individual_questions": individual_questions
            },
            "metadata": {
                "iteration": f"Set {iteration_number}",
                "regenerations": iteration_number - 1,
                "timestamp": self._get_timestamp()
            }
        }
        
        self.feedback_data.append(feedback_entry)
        self.save_feedback()
        print(f"Positive feedback recorded for {objective} - {story_title} - Set {iteration_number}!")
        print(f"Stored {len(individual_questions)} individual question feedbacks")
    
    def add_negative_feedback(self, inputs: Dict, outputs: Dict, question_feedbacks: Dict, overall_feedback: str = ""):
        """Add negative feedback (disliked question set) with individual question feedback."""
        objective = inputs.get("objective", "moral")
        story_title = inputs.get("story_title", "Unknown")
        
        # Count existing entries for this objective+story combination to determine iteration
        existing_entries = [entry for entry in self.feedback_data 
                          if entry.get("inputs", {}).get("objective") == objective and entry.get("inputs", {}).get("story_title") == story_title]
        iteration_number = len(existing_entries) + 1
        
        # Prepare individual question feedback
        individual_questions = {}
        for i, question in enumerate(outputs.get("questions", [])):
            if i in question_feedbacks:
                feedback_info = question_feedbacks[i]
                individual_questions[str(i)] = {
                    "feedback": feedback_info["feedback"],
                    "reasoning": feedback_info["reasoning"]
                }
        
        # Create DSPy-compatible structure
        feedback_entry = {
            "inputs": {
                "story": inputs.get("story", ""),
                "segments": inputs.get("segments", []),
                "objective": objective,
                "story_title": story_title
            },
            "outputs": {
                "questions": outputs.get("questions", []),
                "learning_objectives": outputs.get("learning_objectives", [])
            },
            "feedback": {
                "overall": "negative",
                "overall_reasoning": overall_feedback or "Teacher provided negative feedback and requested regeneration",
                "individual_questions": individual_questions
            },
            "metadata": {
                "iteration": f"Set {iteration_number}",
                "regenerations": iteration_number - 1,
                "timestamp": self._get_timestamp()
            }
        }
        
        self.feedback_data.append(feedback_entry)
        self.save_feedback()
        print(f"Negative feedback recorded for {objective} - {story_title} - Set {iteration_number}!")
        print(f"Stored {len(individual_questions)} individual question feedbacks")
    
    def get_positive_examples(self) -> List[Dict]:
        """Get all positive feedback examples for training."""
        return [entry for entry in self.feedback_data if entry.get("feedback_type") == "positive"]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import datetime
        return datetime.datetime.now().isoformat()

class ObjectiveQuestionGeneratorDSPy:
    """Main class for objective-based question generation with DSPy and feedback learning."""
    
    def __init__(self):
        self.generator = ObjectiveQuestionGenerator()
        self.feedback_collector = FeedbackCollector()
        self.optimizer = None  # Will be initialized when we have enough feedback
    
    def generate_objective_questions(self, story: str, segments: List[Dict], objective: str, story_title: str) -> Dict[str, Any]:
        """
        Generate questions based on teacher-defined objective.
        
        Args:
            story: The full story text
            segments: List of story segments with summaries and reasoning
            objective: Teacher-specified objective (e.g., "empathy", "moral", "comprehension")
            story_title: Title of the story
            
        Returns:
            Dict containing questions and learning objectives
        """
        try:
            # Generate questions using DSPy
            result = self.generator.forward(story, segments, objective, story_title)
            
            # Parse the structured output
            questions_data = self._parse_questions_output(result.questions)
            learning_objectives = self._parse_learning_objectives(result.learning_objectives)
            
            return {
                "questions": questions_data,
                "learning_objectives": learning_objectives,
                "objective": objective,
                "story_title": story_title
            }
            
        except Exception as e:
            print(f"Error generating objective questions: {e}")
            return None
    
    def _parse_questions_output(self, questions_text: str) -> List[Dict]:
        """Parse questions from DSPy output."""
        try:
            # Try to extract JSON from the output
            questions_data = self._extract_json_from_text(questions_text)
            
            if isinstance(questions_data, list):
                # Add name field to each question
                for i, question in enumerate(questions_data):
                    if isinstance(question, dict):
                        question["name"] = f"question_{i+1}"
                return questions_data
            else:
                print("Questions output is not a list")
                return []
                
        except Exception as e:
            print(f"Error parsing questions: {e}")
            return []
    
    def _parse_learning_objectives(self, objectives_text: str) -> List[str]:
        """Parse learning objectives from DSPy output."""
        try:
            # Try to extract list from the output
            objectives_data = self._extract_json_from_text(objectives_text)
            
            if isinstance(objectives_data, list):
                return objectives_data
            else:
                # Try to split by lines or other delimiters
                objectives = [obj.strip() for obj in objectives_text.split('\n') if obj.strip()]
                return objectives[:4]  # Limit to 4 objectives
                
        except Exception as e:
            print(f"Error parsing learning objectives: {e}")
            return []
    
    def _extract_json_from_text(self, text: str) -> Any:
        """Extract JSON from text that might contain other content."""
        import re
        
        # Try to find JSON array or object
        json_patterns = [
            r'\[.*\]',  # Array pattern
            r'\{.*\}',  # Object pattern
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # If no JSON found, return the original text
        return text
    
    def record_positive_feedback(self, inputs: Dict, outputs: Dict, question_feedbacks: Dict = None, overall_feedback: str = ""):
        """Record positive feedback from teacher."""
        self.feedback_collector.add_positive_feedback(inputs, outputs, question_feedbacks, overall_feedback)
        self._check_for_optimization()
    
    def record_negative_feedback(self, inputs: Dict, outputs: Dict, question_feedbacks: Dict, overall_feedback: str = ""):
        """Record negative feedback from teacher."""
        self.feedback_collector.add_negative_feedback(inputs, outputs, question_feedbacks, overall_feedback)
        self._check_for_optimization()
    
    def _check_for_optimization(self):
        """Check if we have enough feedback to optimize the model."""
        positive_examples = self.feedback_collector.get_positive_examples()
        total_examples = len(self.feedback_collector.feedback_data)
        
        print(f"Feedback Status: {len(positive_examples)} positive examples, {total_examples} total examples")
        
        if len(positive_examples) >= 3 and self.optimizer is None:
            print(f" OPTIMIZER ACTIVATED! Enough positive examples ({len(positive_examples)}) for optimization!")
            self._initialize_optimizer()
        elif self.optimizer is not None:
            print(f"Optimizer already active with {len(positive_examples)} positive examples")
        else:
            print(f" Need {3 - len(positive_examples)} more positive examples for optimization")
    
    def _initialize_optimizer(self):
        """Initialize DSPy optimizer with feedback data."""
        try:
            from dspy.teleprompt import BootstrapFewShot
            
            # Prepare training examples
            trainset = []
            for example in self.feedback_collector.get_positive_examples():
                trainset.append(dspy.Example(
                    story=example["inputs"]["story"],
                    segments=example["inputs"]["segments"],
                    objective=example["inputs"]["objective"],
                    story_title=example["inputs"]["story_title"],
                    questions=json.dumps(example["outputs"]["questions"]),
                    learning_objectives=json.dumps(example["outputs"]["learning_objectives"])
                ).with_inputs("story", "segments", "objective", "story_title"))
            
            # Initialize optimizer
            self.optimizer = BootstrapFewShot(metric=self._question_quality_metric)
            
            # Optimize the generator
            print(" Optimizing question generator with feedback data...")
            print(f"Training with {len(trainset)} examples")
            self.generator = self.optimizer.compile(self.generator, trainset=trainset)
            print("Optimization complete! Model now uses learned patterns.")
            
        except Exception as e:
            print(f"Error initializing optimizer: {e}")
    
    def _question_quality_metric(self, example, prediction, trace=None):
        """Simple metric for question quality (can be enhanced)."""
        try:
            questions = json.loads(prediction.questions)
            if isinstance(questions, list) and len(questions) >= 5:
                return 1.0  # Good quality
            return 0.0  # Poor quality
        except:
            return 0.0
    
    def load_story_content(self, story_dir: str) -> str:
        """Load story content from JSON files, maintaining page order."""
        story_content = []
        json_files = [f for f in os.listdir(story_dir) if f.endswith('.json')]
        
        # Sort files by page number (format: storyname_00.json, storyname_01.json, etc.)
        json_files.sort(key=lambda f: int(f.split('.')[0].split('_')[-1]))
        
        for filename in json_files:
            with open(os.path.join(story_dir, filename), 'r') as f:
                page_content = json.load(f)
                text = page_content.get('text', '')
                if isinstance(text, list):
                    text = ' '.join(text)
                story_content.append(text)
        
        return ' '.join(f"[Page {i+1}] {text}" for i, text in enumerate(story_content))

def main():
    """Main function to process storybooks with objective-based question generation."""
    
    parser = argparse.ArgumentParser(description='Generate objective-based questions using DSPy')
    parser.add_argument('--story', type=str, help='Story content to process')
    parser.add_argument('--file', type=str, help='Path to story file')
    parser.add_argument('--objective', type=str, default='moral', help='Objective for question generation')
    parser.add_argument('--batch', action='store_true', help='Process all storybooks in batch mode')
    parser.add_argument('--feedback', type=str, choices=['positive', 'negative'], help='Record feedback')
    parser.add_argument('--feedback-reason', type=str, help='Reason for negative feedback')
    
    args = parser.parse_args()
    
    generator = ObjectiveQuestionGeneratorDSPy()
    
    # If story content is provided directly
    if args.story:
        # For direct story input, we need segments - let's create a simple one
        segments = [{"SUMMARY": "Story content", "REASONING": "Full story for question generation"}]
        result = generator.generate_objective_questions(
            args.story, segments, args.objective, "Custom Story"
        )
        if result:
            print(json.dumps(result, indent=2))
        return
    
    # If file path is provided
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                story_content = f.read()
            segments = [{"SUMMARY": "Story content", "REASONING": "Full story for question generation"}]
            result = generator.generate_objective_questions(
                story_content, segments, args.objective, "Custom Story"
            )
            if result:
                print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error reading file {args.file}: {e}")
        return
    
    # Batch mode (original behavior)
    if args.batch:
        # Configuration paths - read from environment
        asset_path = "/Users/mariyamohiuddin/Desktop/interactive-storybook-assets/qna_json/"
        moral_path = "/Users/mariyamohiuddin/Desktop/Outputs/"
        output_path = "/Users/mariyamohiuddin/Desktop/ObjectiveQuestions/"
        
        # Create output directory if it doesn't exist
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Get list of moral files
        moral_files = [f for f in os.listdir(moral_path) if f.endswith('_moral_segments.json')]
        
        # Process each moral file
        for moral_file in moral_files:
            story_name = moral_file.replace('_moral_segments.json', '')
            print(f"\nProcessing story: {story_name}")
            
            # Load story content
            story_dir = os.path.join(asset_path, story_name)
            if not os.path.exists(story_dir):
                print(f"Story directory not found: {story_dir}")
                continue
                
            story_content = generator.load_story_content(story_dir)
            
            # Load existing moral and segments
            moral_file_path = os.path.join(moral_path, moral_file)
            if not os.path.exists(moral_file_path):
                print(f"Moral file not found: {moral_file_path}")
                continue
                
            with open(moral_file_path, 'r', encoding='utf-8') as f:
                moral_data = json.load(f)
                moral = moral_data.get('moral', '')
                segments = moral_data.get('segments', [])
            
            # Generate objective questions
            questions_file = os.path.join(output_path, f"{story_name}_objective_questions.json")
            if os.path.exists(questions_file):
                print(f"Questions file already exists: {questions_file}")
                continue
                
            try:
                questions_result = generator.generate_objective_questions(
                    story=story_content,
                    segments=segments,
                    objective=args.objective,
                    story_title=story_name
                )
                
                if questions_result:
                    with open(questions_file, 'w', encoding='utf-8') as f:
                        json.dump(questions_result, f, indent=4, ensure_ascii=False)
                    print(f"Saved objective questions to: {questions_file}")
                else:
                    print(f"Failed to generate objective questions for story: {story_name}")
                    
            except Exception as e:
                print(f"Error generating objective questions for {story_name}: {str(e)}")
                continue

if __name__ == "__main__":
    main()
