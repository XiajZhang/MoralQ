"""
Phase E: Full Pedagogical Loop Validation Test Suite

Purpose: Confirm that regenerated questions align better with pedagogical rubrics and teacher intentions.

This is a COMPREHENSIVE test that integrates all functionality from previous test phases:

FROM PHASE A (Generation Quality):
  ✓ QuestionGeneratorModule.generate() - Question generation
  ✓ Moral and segment extraction (if needed)
  
FROM PHASE B (Evaluator Consistency):
  ✓ ContextQEvaluationPipeline.suitability_program() - Question evaluation
  ✓ All suitability evaluators (recall, open-ended, distancing, wh, completion)
  ✓ Evaluation metrics calculation

FROM PHASE D (Optimization):
  ✓ BootstrapFewShot optimizer compilation
  ✓ Optimizer training example creation
  ✓ Rubric-based quality metric
  ✓ Optimized generator usage vs base generator

FROM TEACHER FEEDBACK SYSTEM:
  ✓ TeacherFeedbackSystem.process_feedback() - Feedback processing
  ✓ Weight adjustment (reinforce/adjust)
  ✓ Dynamic evaluator creation
  ✓ FeedbackOrchestrator interpretation

Test Cases:
E1: Initial generation (BASE) → "too factual" feedback → Regenerate (OPTIMIZED) → Verify improvement
E2: "Too complex" feedback → Create complexity evaluator → Regenerate (OPTIMIZED) → Verify simpler vocabulary
E3: Positive feedback (5x) → Verify weight stabilization

Evaluation metric: Rubric average improvement + optimizer usage verification + weight stability
"""

import os
import sys
import json
import copy
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'services'))

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(env_path)

# Import required modules
from teacher_feedback_system import TeacherFeedbackSystem
from question_generator import QuestionGeneratorModule
from contextq_evaluators import ContextQEvaluationPipeline

# Base directory for test data
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_results', 'phase_e_test_data')
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# Test story data
TEST_STORY_CONTENT = """
Grumpy Monkey was having a bad day. Nothing was going right. His friend Norman tried to help.
Norman said, "Sometimes I feel grumpy too. It's okay to feel that way."
Grumpy Monkey learned that it's normal to have bad days, and friends can help you feel better.
"""

TEST_SEGMENTS = [
    {
        "name": "Bad Day",
        "start": 1,
        "end": 2,
        "summary": "Grumpy Monkey is having a bad day",
        "reasoning": "Sets up the emotional state"
    },
    {
        "name": "Friend Helps",
        "start": 3,
        "end": 4,
        "summary": "Norman helps Grumpy Monkey understand his feelings",
        "reasoning": "Shows friendship and emotional support"
    }
]


class PhaseETester:
    """Test suite for Phase E: Full Pedagogical Loop Validation"""
    
    def __init__(self):
        self.results = []
        self.timestamp = datetime.now().isoformat()
        self.story_title = "Grumpy Monkey"
        self.objective = "Emotional regulation"
    
    def get_test_feedback_file(self, test_id: str) -> str:
        """Get path to test-specific feedback file"""
        return os.path.join(TEST_DATA_DIR, f"{test_id}_feedback_records.json")
    
    def get_test_eval_file(self, test_id: str) -> str:
        """Get path to test-specific evaluation file"""
        eval_file = os.path.join(TEST_DATA_DIR, f"{test_id}_question_evaluations.json")
        # Initialize file if it doesn't exist
        if not os.path.exists(eval_file):
            with open(eval_file, 'w') as f:
                json.dump({"evaluations": []}, f)
        return eval_file
    
    def generate_questions(self, story_content: str, segments: List[Dict], feedback_context: str = "", use_optimizer: bool = False, feedback_file: str = None, eval_file: str = None) -> Dict[str, Any]:
        """
        Generate questions using the question generator
        
        Args:
            story_content: Story text
            segments: Story segments
            feedback_context: Avoidance instructions from feedback
            use_optimizer: Whether to use optimized generator (True) or base generator (False)
            feedback_file: Path to feedback file for optimizer
            eval_file: Path to evaluation file for optimizer
        
        Returns:
            Dict with questions and optimizer status
        """
        try:
            # Set up backend paths for optimizer
            backend_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'backend')
            backend_feedback_file = os.path.join(backend_dir, 'teacher_feedback_records.json')
            backend_eval_file = os.path.join(backend_dir, 'question_evaluations.json')
            
            # Backup existing files
            import shutil
            feedback_backup = None
            eval_backup = None
            if os.path.exists(backend_feedback_file):
                feedback_backup = backend_feedback_file + '.bak'
                shutil.copy(backend_feedback_file, feedback_backup)
            if os.path.exists(backend_eval_file):
                eval_backup = backend_eval_file + '.bak'
                shutil.copy(backend_eval_file, eval_backup)
            
            # Copy test files if provided and they exist
            if feedback_file and os.path.exists(feedback_file):
                shutil.copy(feedback_file, backend_feedback_file)
            if eval_file and os.path.exists(eval_file):
                shutil.copy(eval_file, backend_eval_file)
            
            # Initialize generator based on use_optimizer flag
            if use_optimizer:
                print("[INFO] Initializing generator WITH optimizer...")
                try:
                    generator = QuestionGeneratorModule(optimize_with_feedback=True)
                    # Verify optimizer actually compiled by checking if it has the compiled attribute
                    # The optimized flag might be True even if compilation failed
                    has_compiled_optimizer = (
                        generator.optimized and 
                        generator.optimizer is not None and
                        hasattr(generator.question_generator, '_compiled') and
                        generator.question_generator._compiled
                    )
                    
                    if has_compiled_optimizer:
                        optimizer_used = True
                        print("[INFO] ✓ Optimized generator compiled and ready to use")
                    else:
                        optimizer_used = False
                        print("[WARN] Optimizer flag set but compilation may have failed - using base generator")
                        # Use base generator if optimization didn't work
                        generator = QuestionGeneratorModule(optimize_with_feedback=False)
                except Exception as e:
                    print(f"[WARN] Error during optimizer initialization: {e}")
                    optimizer_used = False
                    generator = QuestionGeneratorModule(optimize_with_feedback=False)
            else:
                print("[INFO] Initializing generator WITHOUT optimizer (baseline)...")
                generator = QuestionGeneratorModule(optimize_with_feedback=False)
                optimizer_used = False
            
            # Generate questions
            result = generator.generate(
                story=story_content,
                segments=segments,
                objective=self.objective,
                story_title=self.story_title,
                moral_text="It's okay to have bad days and friends can help",
                avoidance_instructions=feedback_context
            )
            
            # Add optimizer status to result
            if result:
                result["optimizer_used"] = optimizer_used
                result["generator_type"] = "optimized" if optimizer_used else "base"
            
            # Restore backups
            if feedback_backup and os.path.exists(feedback_backup):
                shutil.move(feedback_backup, backend_feedback_file)
            if eval_backup and os.path.exists(eval_backup):
                shutil.move(eval_backup, backend_eval_file)
            
            return result
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_questions(self, questions: List[Dict], set_number: str, eval_file: str) -> List[Dict[str, Any]]:
        """Evaluate questions using ContextQ evaluator"""
        try:
            evaluator = ContextQEvaluationPipeline(
                storage_file=eval_file,
                feedback_records_file=None,
                dynamic_evaluators=None
            )
            
            evaluations = []
            for q_obj in questions:
                question_text = q_obj.get("question", "") if isinstance(q_obj, dict) else str(q_obj)
                if not question_text:
                    continue
                
                # Evaluate question using suitability_program
                eval_result = evaluator.suitability_program(
                    question=question_text,
                    story_context=TEST_STORY_CONTENT
                )
                
                # Add metadata fields expected by the test
                eval_result["storybook_id"] = self.story_title
                eval_result["objective"] = self.objective
                eval_result["set_number"] = set_number
                
                # The evaluator returns suitability_score for the question's classified type
                # Map it to the appropriate evaluator based on question_type
                question_type = eval_result.get("question_type", "").lower().replace("-", "_")
                suitability_score = eval_result.get("suitability_score", 0.0)
                
                # Map the score to the appropriate evaluator
                scores = {
                    "recall_suitability": 0.0,
                    "open_ended_suitability": 0.0,
                    "distancing_suitability": 0.0,
                    "wh_suitability": 0.0,
                    "completion_suitability": 0.0
                }
                
                # Set the score for the evaluated type
                if "recall" in question_type:
                    scores["recall_suitability"] = suitability_score
                elif "open_ended" in question_type or "open ended" in question_type:
                    scores["open_ended_suitability"] = suitability_score
                elif "distancing" in question_type:
                    scores["distancing_suitability"] = suitability_score
                elif "wh" in question_type or question_type.startswith("wh"):
                    scores["wh_suitability"] = suitability_score
                elif "completion" in question_type:
                    scores["completion_suitability"] = suitability_score
                
                evaluations.append({
                    "question": question_text,
                    "suitability_score": suitability_score,
                    "decision": eval_result.get("decision", "unknown"),
                    "question_type": eval_result.get("question_type", ""),
                    "evaluation_reasoning": eval_result.get("evaluation_reasoning", ""),
                    "scores": scores
                })
            
            return evaluations
            
        except Exception as e:
            print(f"Error evaluating questions: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def calculate_metrics(self, questions: List[Dict], evaluations: List[Dict]) -> Dict[str, float]:
        """Calculate metrics for question set"""
        if not evaluations:
            return {}
        
        # Average suitability scores
        avg_suitability = sum(e.get("suitability_score", 0.0) for e in evaluations) / len(evaluations)
        
        # Average specific evaluator scores
        avg_recall = sum(e.get("scores", {}).get("recall_suitability", 0.0) for e in evaluations) / len(evaluations)
        avg_open_ended = sum(e.get("scores", {}).get("open_ended_suitability", 0.0) for e in evaluations) / len(evaluations)
        avg_distancing = sum(e.get("scores", {}).get("distancing_suitability", 0.0) for e in evaluations) / len(evaluations)
        
        # Pass rate
        passing = sum(1 for e in evaluations if e.get("decision") == "pass")
        pass_rate = passing / len(evaluations) if evaluations else 0.0
        
        # Vocabulary complexity (simple heuristic: average word length)
        all_words = []
        for q in questions:
            question_text = q.get("question", "") if isinstance(q, dict) else str(q)
            words = question_text.split()
            all_words.extend(words)
        
        avg_word_length = sum(len(word) for word in all_words) / len(all_words) if all_words else 0.0
        
        return {
            "avg_suitability_score": avg_suitability,
            "avg_recall_suitability": avg_recall,
            "avg_open_ended_suitability": avg_open_ended,
            "avg_distancing_suitability": avg_distancing,
            "pass_rate": pass_rate,
            "avg_word_length": avg_word_length,
            "total_questions": len(questions),
            "passing_questions": passing
        }
    
    def _save_evaluations(self, evaluations: List[Dict], set_number: str, eval_file: str):
        """Save evaluations to evaluation file"""
        try:
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
            
            for eval_item in evaluations:
                eval_data["evaluations"].append({
                    **eval_item,
                    "set_metadata": {
                        "storybook_id": self.story_title,
                        "objective": self.objective,
                        "set_number": set_number
                    }
                })
            
            with open(eval_file, 'w') as f:
                json.dump(eval_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving evaluations: {e}")
    
    def run_test_e1(self) -> Dict[str, Any]:
        """Test E1: Initial generation + "too factual" feedback → Verify improvement"""
        print(f"\n{'='*80}")
        print(f"Running Test E1: Initial Generation + Factual Feedback")
        print(f"{'='*80}")
        
        test_id = "E1"
        feedback_file = self.get_test_feedback_file(test_id)
        eval_file = self.get_test_eval_file(test_id)
        
        # Initialize with fresh files
        if os.path.exists(feedback_file):
            os.remove(feedback_file)
        if os.path.exists(eval_file):
            os.remove(eval_file)
        with open(eval_file, 'w') as f:
            json.dump({"evaluations": []}, f)
        
        # Generate initial questions WITHOUT optimizer (baseline)
        print("\n[E1.1] Generating initial questions (baseline, no optimizer)...")
        initial_result = self.generate_questions(
            TEST_STORY_CONTENT, 
            TEST_SEGMENTS, 
            use_optimizer=False,
            feedback_file=None,
            eval_file=None
        )
        
        if not initial_result or not initial_result.get("questions"):
            return {"test_id": test_id, "status": "failed", "error": "Failed to generate initial questions"}
        
        initial_questions = initial_result.get("questions", [])
        print(f"Generated {len(initial_questions)} initial questions")
        
        # Evaluate initial questions
        print("\n[E1.2] Evaluating initial questions...")
        initial_evaluations = self.evaluate_questions(initial_questions, "set_001", eval_file)
        initial_metrics = self.calculate_metrics(initial_questions, initial_evaluations)
        
        print(f"Initial metrics:")
        print(f"  Average suitability: {initial_metrics.get('avg_suitability_score', 0):.3f}")
        print(f"  Average distancing: {initial_metrics.get('avg_distancing_suitability', 0):.3f}")
        print(f"  Average open-ended: {initial_metrics.get('avg_open_ended_suitability', 0):.3f}")
        print(f"  Pass rate: {initial_metrics.get('pass_rate', 0):.2%}")
        
        # Save initial evaluations
        self._save_evaluations(initial_evaluations, "set_001", eval_file)
        
        # Provide "too factual" feedback
        print("\n[E1.3] Providing 'too factual' feedback...")
        feedback_system = TeacherFeedbackSystem(storage_file=feedback_file)
        
        feedback_result = feedback_system.process_feedback(
            story_title=self.story_title,
            objective=self.objective,
            teacher_feedback="These questions are too factual. We need questions that encourage deeper thinking and personal connection, not just recalling story details.",
            feedback_type="negative",
            question_evaluations=[{
                "scores": {
                    "recall_suitability": 0.9,  # High recall (factual)
                    "distancing_suitability": 0.3,  # Low distancing (not personal)
                    "open_ended_suitability": 0.4  # Low open-ended (too factual)
                }
            }],
            story_context=TEST_STORY_CONTENT,
            generated_questions=[q.get("question", "") if isinstance(q, dict) else str(q) for q in initial_questions]
        )
        
        # Regenerate questions WITH optimizer (should use feedback to improve)
        print("\n[E1.4] Regenerating questions WITH optimizer (using feedback)...")
        avoidance_instructions = "Avoid factual questions that only test recall. Focus on personal connection and deeper thinking."
        regenerated_result = self.generate_questions(
            TEST_STORY_CONTENT, 
            TEST_SEGMENTS,
            feedback_context=avoidance_instructions,
            use_optimizer=True,  # USE OPTIMIZED GENERATOR
            feedback_file=feedback_file,
            eval_file=eval_file
        )
        
        # Verify optimizer was used
        optimizer_was_used = regenerated_result.get("optimizer_used", False) if regenerated_result else False
        print(f"[E1.4a] Optimizer used: {optimizer_was_used}")
        if not optimizer_was_used:
            print("[WARN] Optimizer was not used - results may not reflect optimization improvements")
        
        if not regenerated_result or not regenerated_result.get("questions"):
            return {"test_id": test_id, "status": "failed", "error": "Failed to generate regenerated questions"}
        
        regenerated_questions = regenerated_result.get("questions", [])
        print(f"Generated {len(regenerated_questions)} regenerated questions")
        
        # Evaluate regenerated questions
        print("\n[E1.5] Evaluating regenerated questions...")
        regenerated_evaluations = self.evaluate_questions(regenerated_questions, "set_002", eval_file)
        regenerated_metrics = self.calculate_metrics(regenerated_questions, regenerated_evaluations)
        
        print(f"Regenerated metrics:")
        print(f"  Average suitability: {regenerated_metrics.get('avg_suitability_score', 0):.3f}")
        print(f"  Average distancing: {regenerated_metrics.get('avg_distancing_suitability', 0):.3f}")
        print(f"  Average open-ended: {regenerated_metrics.get('avg_open_ended_suitability', 0):.3f}")
        print(f"  Pass rate: {regenerated_metrics.get('pass_rate', 0):.2%}")
        
        # Save regenerated evaluations
        self._save_evaluations(regenerated_evaluations, "set_002", eval_file)
        
        # Compare metrics
        distancing_improvement = regenerated_metrics.get('avg_distancing_suitability', 0) - initial_metrics.get('avg_distancing_suitability', 0)
        open_ended_improvement = regenerated_metrics.get('avg_open_ended_suitability', 0) - initial_metrics.get('avg_open_ended_suitability', 0)
        suitability_improvement = regenerated_metrics.get('avg_suitability_score', 0) - initial_metrics.get('avg_suitability_score', 0)
        
        print(f"\n[E1.6] Improvement Analysis:")
        print(f"  Distancing improvement: {distancing_improvement:+.3f}")
        print(f"  Open-ended improvement: {open_ended_improvement:+.3f}")
        print(f"  Overall suitability improvement: {suitability_improvement:+.3f}")
        
        improved = distancing_improvement > 0 and open_ended_improvement > 0
        
        return {
            "test_id": test_id,
            "description": "Initial generation + 'too factual' feedback → Verify improvement",
            "timestamp": datetime.now().isoformat(),
            "initial_questions": len(initial_questions),
            "regenerated_questions": len(regenerated_questions),
            "optimizer_used": optimizer_was_used,
            "initial_generator_type": initial_result.get("generator_type", "base") if initial_result else "unknown",
            "regenerated_generator_type": regenerated_result.get("generator_type", "base") if regenerated_result else "unknown",
            "initial_metrics": initial_metrics,
            "regenerated_metrics": regenerated_metrics,
            "distancing_improvement": round(distancing_improvement, 4),
            "open_ended_improvement": round(open_ended_improvement, 4),
            "suitability_improvement": round(suitability_improvement, 4),
            "improved": improved,
            "status": "completed"
        }
    
    def run_test_e2(self) -> Dict[str, Any]:
        """Test E2: 'too complex' feedback → Verify simpler vocabulary"""
        print(f"\n{'='*80}")
        print(f"Running Test E2: Complexity Feedback → Simpler Vocabulary")
        print(f"{'='*80}")
        
        test_id = "E2"
        feedback_file = self.get_test_feedback_file(test_id)
        eval_file = self.get_test_eval_file(test_id)
        
        # Initialize with fresh files (but can build on E1 if needed)
        if os.path.exists(feedback_file):
            os.remove(feedback_file)
        if os.path.exists(eval_file):
            os.remove(eval_file)
        with open(eval_file, 'w') as f:
            json.dump({"evaluations": []}, f)
        
        # Load previous question set (using optimized generator)
        print("\n[E2.1] Loading previous question set (with optimizer)...")
        prev_result = self.generate_questions(
            TEST_STORY_CONTENT, 
            TEST_SEGMENTS, 
            use_optimizer=True,
            feedback_file=None,
            eval_file=None
        )
        
        if not prev_result or not prev_result.get("questions"):
            return {"test_id": test_id, "status": "failed", "error": "Failed to generate previous questions"}
        
        prev_questions = prev_result.get("questions", [])
        prev_evaluations = self.evaluate_questions(prev_questions, "set_003", eval_file)
        prev_metrics = self.calculate_metrics(prev_questions, prev_evaluations)
        
        print(f"Previous average word length: {prev_metrics.get('avg_word_length', 0):.2f}")
        
        # Save previous evaluations
        self._save_evaluations(prev_evaluations, "set_003", eval_file)
        
        # Provide "too complex" feedback
        print("\n[E2.2] Providing 'too complex' feedback...")
        feedback_system = TeacherFeedbackSystem(storage_file=feedback_file)
        
        feedback_result = feedback_system.process_feedback(
            story_title=self.story_title,
            objective=self.objective,
            teacher_feedback="These questions use vocabulary that's too complex for 4-6 year olds. Please simplify the language and use shorter, more common words.",
            feedback_type="negative",
            question_evaluations=[{
                "scores": {
                    "open_ended_suitability": 0.5,
                    "distancing_suitability": 0.5
                }
            }],
            story_context=TEST_STORY_CONTENT,
            generated_questions=[q.get("question", "") if isinstance(q, dict) else str(q) for q in prev_questions]
        )
        
        # Check if new evaluator was created for complexity
        action_taken = feedback_result.get("action_taken", {})
        complexity_evaluator_created = (
            action_taken.get("type") == "add_new_evaluator" and
            "complexity" in action_taken.get("details", {}).get("evaluator_name", "").lower()
        )
        print(f"[E2.2a] Complexity evaluator created: {complexity_evaluator_created}")
        
        # Regenerate with complexity feedback (using optimized generator)
        print("\n[E2.3] Regenerating with complexity feedback (using optimizer)...")
        avoidance_instructions = "Use simple vocabulary appropriate for 4-6 year olds. Avoid complex words. Use shorter sentences."
        regenerated_result = self.generate_questions(
            TEST_STORY_CONTENT,
            TEST_SEGMENTS,
            feedback_context=avoidance_instructions,
            use_optimizer=True,  # USE OPTIMIZED GENERATOR
            feedback_file=feedback_file,
            eval_file=eval_file
        )
        
        optimizer_was_used = regenerated_result.get("optimizer_used", False) if regenerated_result else False
        print(f"[E2.3a] Optimizer used: {optimizer_was_used}")
        
        if not regenerated_result or not regenerated_result.get("questions"):
            return {"test_id": test_id, "status": "failed", "error": "Failed to generate regenerated questions"}
        
        regenerated_questions = regenerated_result.get("questions", [])
        regenerated_evaluations = self.evaluate_questions(regenerated_questions, "set_004", eval_file)
        regenerated_metrics = self.calculate_metrics(regenerated_questions, regenerated_evaluations)
        
        print(f"Regenerated average word length: {regenerated_metrics.get('avg_word_length', 0):.2f}")
        
        # Save regenerated evaluations
        self._save_evaluations(regenerated_evaluations, "set_004", eval_file)
        
        # Compare vocabulary complexity
        word_length_reduction = prev_metrics.get('avg_word_length', 0) - regenerated_metrics.get('avg_word_length', 0)
        simplified = word_length_reduction > 0.5  # At least 0.5 character reduction per word
        
        print(f"\n[E2.4] Vocabulary Simplification:")
        print(f"  Word length reduction: {word_length_reduction:+.2f} characters")
        print(f"  Simplified: {simplified}")
        print(f"  Complexity evaluator created: {complexity_evaluator_created}")
        
        return {
            "test_id": test_id,
            "description": "'too complex' feedback → Verify simpler vocabulary",
            "timestamp": datetime.now().isoformat(),
            "optimizer_used": optimizer_was_used,
            "previous_metrics": prev_metrics,
            "regenerated_metrics": regenerated_metrics,
            "word_length_reduction": round(word_length_reduction, 4),
            "simplified": simplified,
            "complexity_evaluator_created": complexity_evaluator_created,
            "status": "completed"
        }
    
    def run_test_e3(self) -> Dict[str, Any]:
        """Test E3: Positive feedback → Verify weight stabilization"""
        print(f"\n{'='*80}")
        print(f"Running Test E3: Positive Feedback → Weight Stabilization")
        print(f"{'='*80}")
        
        test_id = "E3"
        feedback_file = self.get_test_feedback_file(test_id)
        
        # Initialize with fresh file
        if os.path.exists(feedback_file):
            os.remove(feedback_file)
        
        # Get initial weights
        feedback_system = TeacherFeedbackSystem(storage_file=feedback_file)
        initial_weights = copy.deepcopy(feedback_system.collector.evaluator_weights)
        
        print(f"\n[E3.1] Initial weights: {initial_weights}")
        
        # Generate questions (using optimizer after feedback cycles)
        print("\n[E3.2] Generating questions (with optimizer)...")
        result = self.generate_questions(
            TEST_STORY_CONTENT, 
            TEST_SEGMENTS, 
            use_optimizer=True,
            feedback_file=None,
            eval_file=None
        )
        
        optimizer_was_used = result.get("optimizer_used", False) if result else False
        print(f"[E3.2a] Optimizer used: {optimizer_was_used}")
        
        if not result or not result.get("questions"):
            return {"test_id": test_id, "status": "failed", "error": "Failed to generate questions"}
        
        questions = result.get("questions", [])
        
        # Provide positive feedback multiple times to test stabilization
        print("\n[E3.3] Providing positive feedback (5 times)...")
        weight_history = [copy.deepcopy(initial_weights)]
        
        for i in range(5):
            feedback_result = feedback_system.process_feedback(
                story_title=self.story_title,
                objective=self.objective,
                teacher_feedback="Perfect for kids! These questions are engaging, age-appropriate, and encourage good thinking.",
                feedback_type="positive",
                question_evaluations=[{
                    "scores": {
                        "open_ended_suitability": 0.85,
                        "distancing_suitability": 0.80,
                        "recall_suitability": 0.70
                    }
                }],
                story_context=TEST_STORY_CONTENT,
                generated_questions=[q.get("question", "") if isinstance(q, dict) else str(q) for q in questions]
            )
            
            weight_history.append(copy.deepcopy(feedback_system.collector.evaluator_weights))
            print(f"  After feedback {i+1}: {feedback_system.collector.evaluator_weights}")
        
        final_weights = feedback_system.collector.evaluator_weights
        
        # Calculate weight stability
        weight_changes = {}
        max_change = 0.0
        for evaluator in initial_weights:
            change = abs(final_weights.get(evaluator, 0) - initial_weights.get(evaluator, 0))
            weight_changes[evaluator] = round(change, 4)
            max_change = max(max_change, change)
        
        is_stable = max_change < 0.05  # Weights should remain relatively stable
        
        print(f"\n[E3.4] Weight Stability Analysis:")
        print(f"  Final weights: {final_weights}")
        print(f"  Maximum weight change: {max_change:.4f}")
        print(f"  Stable: {is_stable}")
        
        return {
            "test_id": test_id,
            "description": "Positive feedback → Weight stabilization",
            "timestamp": datetime.now().isoformat(),
            "optimizer_used": optimizer_was_used,
            "initial_weights": initial_weights,
            "final_weights": final_weights,
            "weight_history": weight_history,
            "weight_changes": weight_changes,
            "max_weight_change": round(max_change, 4),
            "is_stable": is_stable,
            "status": "completed"
        }
    
    def run_all_tests(self):
        """Run all test cases"""
        print(f"\n{'='*80}")
        print(f"Phase E Test Suite: Full Pedagogical Loop Validation")
        print(f"Started at: {self.timestamp}")
        print(f"{'='*80}")
        
        # Run each test
        e1_result = self.run_test_e1()
        self.results.append(e1_result)
        
        e2_result = self.run_test_e2()
        self.results.append(e2_result)
        
        e3_result = self.run_test_e3()
        self.results.append(e3_result)
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate test summary and save to JSON"""
        summary = {
            "test_suite": "Phase E: Full Pedagogical Loop Validation",
            "timestamp": self.timestamp,
            "total_tests": len(self.results),
            "completed": sum(1 for r in self.results if r.get("status") == "completed"),
            "failed": sum(1 for r in self.results if r.get("status") == "failed"),
            "test_results": self.results,
            "summary": {
                "e1_improved": self.results[0].get("improved", False) if len(self.results) > 0 and self.results[0].get("status") == "completed" else False,
                "e2_simplified": self.results[1].get("simplified", False) if len(self.results) > 1 and self.results[1].get("status") == "completed" else False,
                "e3_stable": self.results[2].get("is_stable", False) if len(self.results) > 2 and self.results[2].get("status") == "completed" else False
            }
        }
        
        # Save to JSON file
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'test_results')
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, "phase_e_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"Test Suite Complete")
        print(f"{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"\nSummary:")
        print(f"  Test E1 (Factual → Depth): {'✓ Improved' if summary['summary']['e1_improved'] else '✗ Not improved'}")
        print(f"  Test E2 (Complex → Simple): {'✓ Simplified' if summary['summary']['e2_simplified'] else '✗ Not simplified'}")
        print(f"  Test E3 (Stability): {'✓ Stable' if summary['summary']['e3_stable'] else '✗ Not stable'}")
        print(f"\nResults saved to JSON for manual review and interpretation.")


if __name__ == "__main__":
    tester = PhaseETester()
    tester.run_all_tests()
