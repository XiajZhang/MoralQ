"""
Dynamic Evaluator Creation System
==================================

Based on teacher feedback, this system can:
✅ Reinforce existing evaluator weights (positive feedback)
⚙️ Adjust existing evaluator weights (known issue)
🆕 Create new evaluators dynamically (new concept)

Dynamic evaluators are created using DSPy signatures and integrated into the grading pipeline.
"""

import dspy
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure DSPy with OpenAI
dspy.configure(lm=dspy.LM("openai/gpt-4.1-2025-04-14"))


class OptimizedQuestionSignature(dspy.Signature):
    """Optimized question generation signature that learns from teacher feedback."""
    
    story = dspy.InputField(desc="Full story content, marked with page numbers")
    segments = dspy.InputField(desc="Summarized segments of the story with page ranges")
    objective = dspy.InputField(desc="Teacher-specified learning objective")
    story_title = dspy.InputField(desc="Title of the story")
    
    questions = dspy.OutputField(desc="""A JSON array of 5-10 questions with fields:
    {
        "question": "...",
        "type": "comprehension/reflection/application/extension",
        "difficulty": "easy/medium/hard",
        "explanation": "...",
        "page_number": integer
    }""")
    
    learning_objectives = dspy.OutputField(desc="List of 3-4 learning objectives")


class OptimizedQuestionGenerator(dspy.Module):
    """Generate optimized questions using DSPy BootstrapFewShot optimizer."""
    
    def __init__(self):
        super().__init__()
        self.generate_questions = dspy.Predict(OptimizedQuestionSignature)
        self.optimizer = None
        
    def forward(self, story: str, segments: List[Dict], objective: str, story_title: str):
        """Generate questions with optional optimization."""
        
        # Format segments
        formatted_segments = "\n".join([
            f"Segment {i+1} (Pages {seg.get('start', '?')}-{seg.get('end', '?')}): {seg.get('summary', '')}"
            for i, seg in enumerate(segments)
        ])
        
        # Generate questions
        result = self.generate_questions(
            story=story,
            segments=formatted_segments,
            objective=objective,
            story_title=story_title
        )
        
        return result
    
    def _question_quality_metric(self, example, prediction, trace=None):
        """Metric for question quality evaluation."""
        try:
            questions = json.loads(prediction.questions)
            if isinstance(questions, list) and len(questions) >= 5:
                return 1.0
            return 0.0
        except:
            return 0.0
    
    def initialize_optimizer(self, training_examples: List[dspy.Example]):
        """Initialize BootstrapFewShot optimizer with training examples."""
        try:
            from dspy.teleprompt import BootstrapFewShot
            
            if not training_examples:
                print("⚠️  No training examples provided")
                return
            
            print(f"📊 Initializing optimizer with {len(training_examples)} examples")
            
            # Initialize optimizer
            self.optimizer = BootstrapFewShot(metric=self._question_quality_metric)
            
            # Optimize the generator
            print("🔧 Compiling optimized generator...")
            self.generate_questions = self.optimizer.compile(
                self.generate_questions,
                trainset=training_examples
            )
            
            print("✅ Optimization complete!")
            
        except Exception as e:
            print(f"Error initializing optimizer: {e}")


def create_training_examples(feedback_data: List[Dict[str, Any]]) -> List[dspy.Example]:
    """Convert feedback data into DSPy training examples."""
    examples = []
    
    for item in feedback_data:
        inputs = item.get("inputs", {})
        outputs = item.get("outputs", {})
        
        example = dspy.Example(
            story=inputs.get("story", ""),
            segments=inputs.get("segments", []),
            objective=inputs.get("objective", ""),
            story_title=inputs.get("story_title", ""),
            questions=json.dumps(outputs.get("questions", [])),
            learning_objectives=json.dumps(outputs.get("learning_objectives", []))
        ).with_inputs("story", "segments", "objective", "story_title")
        
        examples.append(example)
    
    return examples


class DynamicEvaluatorCreator:
    """
    Creates new evaluator agents dynamically based on teacher feedback.
    
    When teacher feedback identifies a NEW concept not covered by existing evaluators,
    this class creates a new DSPy evaluator agent on the fly.
    """
    
    def __init__(self):
        self.created_evaluators = {}
        self.evaluator_registry_file = "dynamic_evaluator_registry.json"
    
    def create_evaluator_signature(self, evaluator_name: str, evaluator_description: str) -> type:
        """
        Dynamically create a new DSPy Signature class for an evaluator.
        
        Args:
            evaluator_name: Name of the new evaluator (e.g., "emotional_resonance")
            evaluator_description: Description of what this evaluator checks
            
        Returns:
            A new DSPy Signature class for the evaluator
        """
        signature_class_name = f"{evaluator_name.capitalize()}EvaluatorSignature"
        
        # Create the signature using exec (dynamic class creation)
        signature_code = f'''
class {signature_class_name}(dspy.Signature):
    """Evaluator for: {evaluator_description}"""
    
    question = dspy.InputField(desc="The question to evaluate")
    story_context = dspy.InputField(desc="The story context")
    moral_or_objective = dspy.InputField(desc="The moral or learning objective")
    
    score = dspy.OutputField(desc="Score (1-5) indicating quality for this evaluator")
    rationale = dspy.OutputField(desc="Brief explanation of the score")
'''
        
        namespace = {'dspy': dspy}
        exec(signature_code, namespace)
        
        signature_class = namespace[signature_class_name]
        
        print(f"✅ Created new evaluator signature: {signature_class_name}")
        return signature_class
    
    def create_evaluator_agent(self, signature_class: type) -> dspy.Predict:
        """
        Create a DSPy Predict agent from a signature.
        
        Args:
            signature_class: The DSPy Signature class
            
        Returns:
            A DSPy Predict agent
        """
        agent = dspy.Predict(signature_class)
        return agent
    
    def create_new_evaluator(self, evaluator_name: str, evaluator_description: str) -> Optional[dspy.Predict]:
        """
        Create a complete evaluator agent with prompt engineering.
        
        Args:
            evaluator_name: Name of the new evaluator
            evaluator_description: Description of what it checks
            
        Returns:
            A DSPy Predict agent that can evaluate questions
        """
        if evaluator_name in self.created_evaluators:
            print(f"✅ Evaluator '{evaluator_name}' already exists")
            return self.created_evaluators[evaluator_name]
        
        # Create the signature
        signature_class = self.create_evaluator_signature(evaluator_name, evaluator_description)
        
        # Create the agent
        agent = self.create_evaluator_agent(signature_class)
        
        # Store in registry
        self.created_evaluators[evaluator_name] = agent
        
        # Save to file
        self._save_evaluator_registry()
        
        print(f"✅ Created complete evaluator agent: {evaluator_name}")
        return agent
    
    def _save_evaluator_registry(self):
        """Save the registry of created evaluators."""
        registry = {
            evaluator_name: str(type(agent).__name__)
            for evaluator_name, agent in self.created_evaluators.items()
        }
        
        with open(self.evaluator_registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def load_evaluator_registry(self) -> Dict[str, str]:
        """Load existing evaluator registry."""
        if os.path.exists(self.evaluator_registry_file):
            with open(self.evaluator_registry_file, 'r') as f:
                return json.load(f)
        return {}


class FeedbackBasedOptimizer:
    """
    Main optimizer that implements the three feedback cases:
    1. Positive → Reinforce
    2. Known issue → Adjust existing evaluator weights
    3. New issue → Create new evaluator dynamically
    """
    
    def __init__(self, feedback_file: str = "teacher_feedback_records.json"):
        self.feedback_file = feedback_file
        self.evaluator_creator = DynamicEvaluatorCreator()
        self.evaluator_registry = self.evaluator_creator.load_evaluator_registry()
    
    def process_feedback_and_optimize(self, story_context: str, questions: List[Dict], 
                                     feedback_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process teacher feedback and implement the appropriate action.
        
        Args:
            story_context: The story content
            questions: Generated questions
            feedback_record: The teacher feedback record
            
        Returns:
            Result containing action taken and optimization details
        """
        action_taken = feedback_record.get("action_taken", {})
        action_type = action_taken.get("type", "no_action")
        
        print(f"\n{'='*80}")
        print(f"Processing Feedback: {action_type}")
        print(f"{'='*80}")
        
        if action_type == "reinforce_existing":
            return self._handle_reinforce_case(feedback_record)
        
        elif action_type == "adjust_evaluator":
            return self._handle_adjust_case(feedback_record)
        
        elif action_type == "add_new_evaluator":
            return self._handle_create_new_evaluator_case(feedback_record)
        
        else:
            return {
                "action": "no_action",
                "message": "No optimization applied",
                "details": {}
            }
    
    def _handle_reinforce_case(self, feedback_record: Dict[str, Any]) -> Dict[str, Any]:
        """Handle positive feedback - reinforce existing weights."""
        return {
            "action": "reinforce",
            "message": "Positive feedback - maintaining current evaluator weights",
            "details": {
                "feedback": "positive",
                "evaluators": ["relevance", "clarity", "depth", "engagement", "appropriateness"]
            }
        }
    
    def _handle_adjust_case(self, feedback_record: Dict[str, Any]) -> Dict[str, Any]:
        """Handle known issue - adjust existing evaluator weights."""
        action = feedback_record.get("action_taken", {})
        affected = action.get("affected", [])
        delta = action.get("delta", {})
        
        return {
            "action": "adjust",
            "message": f"Adjusted weights for: {affected}",
            "details": {
                "affected_evaluators": affected,
                "weight_changes": delta
            }
        }
    
    def _handle_create_new_evaluator_case(self, feedback_record: Dict[str, Any]) -> Dict[str, Any]:
        """Handle new issue - create a new evaluator dynamically."""
        interpretation = feedback_record.get("interpretation", {})
        evaluator_name = interpretation.get("new_evaluator_name", "").replace(" ", "_").lower()
        
        if not evaluator_name or evaluator_name == "none":
            return {
                "action": "no_action",
                "message": "No evaluator name provided",
                "details": {}
            }
        
        # Create a description for the new evaluator
        description = f"Evaluates questions for {evaluator_name.replace('_', ' ')} based on teacher feedback"
        
        # Create the evaluator
        evaluator_agent = self.evaluator_creator.create_new_evaluator(evaluator_name, description)
        
        return {
            "action": "create_new_evaluator",
            "message": f"Created new evaluator: {evaluator_name}",
            "details": {
                "evaluator_name": evaluator_name,
                "evaluator_description": description,
                "status": "active"
            }
        }


def optimize_question_generator(feedback_file: str = "teacher_feedback_records.json"):
    """Optimize question generator using BootstrapFewShot."""
    
    # Load feedback data
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
    else:
        print(f"⚠️  Feedback file not found: {feedback_file}")
        return None
    
    # Create training examples
    training_examples = create_training_examples(feedback_data)
    
    if not training_examples:
        print("⚠️  No training examples available")
        return None
    
    # Initialize generator and optimizer
    generator = OptimizedQuestionGenerator()
    generator.initialize_optimizer(training_examples)
    
    return generator


if __name__ == "__main__":
    # Test optimizer
    generator = optimize_question_generator()
    
    if generator:
        print("✅ Optimized question generator ready!")
    else:
        print("⚠️  Could not create optimized generator")

