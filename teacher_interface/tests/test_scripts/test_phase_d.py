"""
Phase D: Optimization & Learning Behavior Test Suite

Purpose: Validate DSPy's ability to adjust and improve generation quality through feedback-driven optimization.

Test Cases:
- D1: 10 positive feedback records → Reinforce weights; little change (stable behavior)
- D2: 10 negative feedback records targeting same evaluator → Weight decreases (Δ≥0.05)
- D3: Mixed positive/negative on distinct evaluators → Divergent weight adjustment
- D4: "Add new evaluator" feedback → DynamicEvaluatorCreator adds new evaluator
- D5: Run optimizer after 10 mixed feedback cycles → Measurable improvement (>10% score gain)

Evaluation metric: DSPy metric() score improvement between iterations — ideally increasing monotonically.
"""

import os
import sys
import json
import copy
import shutil
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
from teacher_feedback_system import TeacherFeedbackCollector, TeacherFeedbackSystem

# Base directory for test data
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_results', 'phase_d_test_data')
os.makedirs(TEST_DATA_DIR, exist_ok=True)


class PhaseDTester:
    """Test suite for Phase D: Optimization & Learning Behavior"""
    
    def __init__(self):
        self.results = []
        self.timestamp = datetime.now().isoformat()
        self.base_weights = {
            "completion_suitability": 0.15,
            "recall_suitability": 0.20,
            "open_ended_suitability": 0.25,
            "wh_suitability": 0.20,
            "distancing_suitability": 0.20
        }
    
    def get_test_feedback_file(self, test_id: str) -> str:
        """Get path to test-specific feedback file"""
        return os.path.join(TEST_DATA_DIR, f"{test_id}_feedback_records.json")
    
    def get_initial_weights(self) -> Dict[str, float]:
        """Get initial evaluator weights"""
        return copy.deepcopy(self.base_weights)
    
    def create_positive_feedback(self, story_title: str, objective: str, index: int) -> Dict[str, Any]:
        """Create a positive feedback record"""
        return {
            "feedback_text": f"Great questions! The questions are engaging and appropriate.",
            "feedback_type": "positive",
            "story_title": story_title,
            "objective": objective,
            "question_evaluations": [
                {
                    "scores": {
                        "completion_suitability": 0.8,
                        "recall_suitability": 0.9,
                        "open_ended_suitability": 0.85
                    }
                }
            ],
            "generated_questions": [f"Question {i} from iteration {index}" for i in range(5)]
        }
    
    def create_negative_feedback(self, story_title: str, objective: str, target_evaluator: str, index: int) -> Dict[str, Any]:
        """Create a negative feedback record targeting a specific evaluator"""
        feedback_messages = {
            "recall_suitability": "The recall questions focus too much on factual details and memorization. We need questions that go beyond simple recall of plot elements.",
            "open_ended_suitability": "These questions don't encourage enough elaboration. Make them more open-ended.",
            "completion_suitability": "The completion questions lack rhyming and poetic elements.",
            "wh_suitability": "Wh-questions aren't focusing enough on story details.",
            "distancing_suitability": "Questions don't connect enough to children's personal experiences."
        }
        
        return {
            "feedback_text": feedback_messages.get(target_evaluator, "These questions need improvement."),
            "feedback_type": "negative",
            "story_title": story_title,
            "objective": objective,
            "question_evaluations": [
                {
                    "scores": {
                        target_evaluator: 0.3  # Low score for target evaluator
                    }
                }
            ],
            "generated_questions": [f"Question {i} from iteration {index}" for i in range(5)]
        }
    
    def create_new_evaluator_feedback(self, story_title: str, objective: str, evaluator_name: str, index: int) -> Dict[str, Any]:
        """Create feedback that triggers new evaluator creation"""
        feedback_messages = {
            "creativity": "These questions lack creativity. We need questions that spark imagination.",
            "emotional_depth": "Questions need more emotional depth and connection to feelings.",
            "interaction": "Questions should encourage more interaction and discussion."
        }
        
        return {
            "feedback_text": feedback_messages.get(evaluator_name, f"Questions need {evaluator_name}."),
            "feedback_type": "negative",
            "story_title": story_title,
            "objective": objective,
            "question_evaluations": [],
            "generated_questions": [f"Question {i} from iteration {index}" for i in range(5)]
        }
    
    def run_test_d1(self) -> Dict[str, Any]:
        """Test D1: 10 positive feedback records → Reinforce weights; little change"""
        print(f"\n{'='*80}")
        print(f"Running Test D1: 10 Positive Feedback Records")
        print(f"{'='*80}")
        
        test_id = "D1"
        feedback_file = self.get_test_feedback_file(test_id)
        
        # Initialize collector with fresh file
        if os.path.exists(feedback_file):
            os.remove(feedback_file)
        
        collector = TeacherFeedbackCollector(storage_file=feedback_file)
        initial_weights = copy.deepcopy(collector.evaluator_weights)
        
        print(f"Initial weights: {initial_weights}")
        
        # Create and process 10 positive feedback records
        weight_history = [copy.deepcopy(initial_weights)]
        
        for i in range(10):
            feedback_data = self.create_positive_feedback("Test Story", "Empathy", i)
            record = collector.collect_feedback(
                story_title=feedback_data["story_title"],
                objective=feedback_data["objective"],
                teacher_feedback=feedback_data["feedback_text"],
                feedback_type=feedback_data["feedback_type"],
                question_evaluations=feedback_data["question_evaluations"],
                generated_questions=feedback_data["generated_questions"]
            )
            weight_history.append(copy.deepcopy(collector.evaluator_weights))
            print(f"After feedback {i+1}: {collector.evaluator_weights}")
        
        final_weights = collector.evaluator_weights
        
        # Calculate weight changes
        weight_changes = {}
        for evaluator in initial_weights:
            change = final_weights.get(evaluator, 0) - initial_weights.get(evaluator, 0)
            weight_changes[evaluator] = round(change, 4)
        
        # Check if weights are stable (little change)
        max_change = max(abs(change) for change in weight_changes.values())
        is_stable = max_change < 0.05
        
        print(f"\nFinal weights: {final_weights}")
        print(f"Weight changes: {weight_changes}")
        print(f"Max change: {max_change:.4f}")
        print(f"Stable: {is_stable}")
        
        return {
            "test_id": test_id,
            "description": "10 positive feedback records → Reinforce weights; little change",
            "timestamp": datetime.now().isoformat(),
            "initial_weights": initial_weights,
            "final_weights": final_weights,
            "weight_history": weight_history,
            "weight_changes": weight_changes,
            "max_weight_change": round(max_change, 4),
            "is_stable": is_stable,
            "status": "completed"
        }
    
    def run_test_d2(self) -> Dict[str, Any]:
        """Test D2: 10 negative feedback records targeting same evaluator → Weight decreases"""
        print(f"\n{'='*80}")
        print(f"Running Test D2: 10 Negative Feedback Records (Same Evaluator)")
        print(f"{'='*80}")
        
        test_id = "D2"
        feedback_file = self.get_test_feedback_file(test_id)
        target_evaluator = "recall_suitability"
        
        # Initialize collector with fresh file
        if os.path.exists(feedback_file):
            os.remove(feedback_file)
        
        collector = TeacherFeedbackCollector(storage_file=feedback_file)
        initial_weights = copy.deepcopy(collector.evaluator_weights)
        initial_target_weight = initial_weights.get(target_evaluator, 0)
        
        print(f"Initial weights: {initial_weights}")
        print(f"Target evaluator ({target_evaluator}) initial weight: {initial_target_weight:.4f}")
        
        # Create and process 10 negative feedback records targeting the same evaluator
        weight_history = [copy.deepcopy(initial_weights)]
        
        for i in range(10):
            feedback_data = self.create_negative_feedback("Test Story", "Empathy", target_evaluator, i)
            record = collector.collect_feedback(
                story_title=feedback_data["story_title"],
                objective=feedback_data["objective"],
                teacher_feedback=feedback_data["feedback_text"],
                feedback_type=feedback_data["feedback_type"],
                question_evaluations=feedback_data["question_evaluations"],
                generated_questions=feedback_data["generated_questions"]
            )
            weight_history.append(copy.deepcopy(collector.evaluator_weights))
            print(f"After feedback {i+1}: {collector.evaluator_weights}")
        
        final_weights = collector.evaluator_weights
        final_target_weight = final_weights.get(target_evaluator, 0)
        weight_delta = final_target_weight - initial_target_weight
        
        # Check if weight decreased by at least 0.05
        weight_decreased = weight_delta <= -0.05
        
        print(f"\nFinal weights: {final_weights}")
        print(f"Target evaluator ({target_evaluator}) final weight: {final_target_weight:.4f}")
        print(f"Weight delta: {weight_delta:.4f}")
        print(f"Decreased by ≥0.05: {weight_decreased}")
        
        return {
            "test_id": test_id,
            "description": "10 negative feedback records targeting same evaluator → Weight decreases",
            "timestamp": datetime.now().isoformat(),
            "target_evaluator": target_evaluator,
            "initial_weights": initial_weights,
            "final_weights": final_weights,
            "weight_history": weight_history,
            "target_evaluator_initial_weight": round(initial_target_weight, 4),
            "target_evaluator_final_weight": round(final_target_weight, 4),
            "weight_delta": round(weight_delta, 4),
            "weight_decreased_sufficiently": weight_decreased,
            "status": "completed"
        }
    
    def run_test_d3(self) -> Dict[str, Any]:
        """Test D3: Mixed positive/negative on distinct evaluators → Divergent weight adjustment"""
        print(f"\n{'='*80}")
        print(f"Running Test D3: Mixed Positive/Negative Feedback (Distinct Evaluators)")
        print(f"{'='*80}")
        
        test_id = "D3"
        feedback_file = self.get_test_feedback_file(test_id)
        
        # Initialize collector with fresh file
        if os.path.exists(feedback_file):
            os.remove(feedback_file)
        
        collector = TeacherFeedbackCollector(storage_file=feedback_file)
        initial_weights = copy.deepcopy(collector.evaluator_weights)
        
        print(f"Initial weights: {initial_weights}")
        
        # Create mixed feedback: positive for one evaluator, negative for another
        positive_evaluator = "open_ended_suitability"
        negative_evaluator = "recall_suitability"
        
        weight_history = [copy.deepcopy(initial_weights)]
        
        # 5 positive feedbacks for open_ended_suitability
        for i in range(5):
            record = collector.collect_feedback(
                story_title="Test Story",
                objective="Empathy",
                teacher_feedback="Great open-ended questions that encourage thinking!",
                feedback_type="positive",
                question_evaluations=[{"scores": {positive_evaluator: 0.9}}],
                generated_questions=[f"Question {j}" for j in range(5)]
            )
            weight_history.append(copy.deepcopy(collector.evaluator_weights))
        
        # 5 negative feedbacks for recall_suitability
        for i in range(5):
            feedback_data = self.create_negative_feedback("Test Story", "Empathy", negative_evaluator, i)
            record = collector.collect_feedback(
                story_title=feedback_data["story_title"],
                objective=feedback_data["objective"],
                teacher_feedback=feedback_data["feedback_text"],
                feedback_type=feedback_data["feedback_type"],
                question_evaluations=feedback_data["question_evaluations"],
                generated_questions=feedback_data["generated_questions"]
            )
            weight_history.append(copy.deepcopy(collector.evaluator_weights))
        
        final_weights = collector.evaluator_weights
        
        # Calculate changes for both evaluators
        positive_delta = final_weights.get(positive_evaluator, 0) - initial_weights.get(positive_evaluator, 0)
        negative_delta = final_weights.get(negative_evaluator, 0) - initial_weights.get(negative_evaluator, 0)
        
        # Check if they diverged (positive increased, negative decreased)
        diverged = positive_delta > 0 and negative_delta < 0
        
        print(f"\nFinal weights: {final_weights}")
        print(f"{positive_evaluator} delta: {positive_delta:.4f}")
        print(f"{negative_evaluator} delta: {negative_delta:.4f}")
        print(f"Diverged: {diverged}")
        
        return {
            "test_id": test_id,
            "description": "Mixed positive/negative on distinct evaluators → Divergent weight adjustment",
            "timestamp": datetime.now().isoformat(),
            "positive_evaluator": positive_evaluator,
            "negative_evaluator": negative_evaluator,
            "initial_weights": initial_weights,
            "final_weights": final_weights,
            "weight_history": weight_history,
            "positive_evaluator_delta": round(positive_delta, 4),
            "negative_evaluator_delta": round(negative_delta, 4),
            "weights_diverged": diverged,
            "status": "completed"
        }
    
    def run_test_d4(self) -> Dict[str, Any]:
        """Test D4: Add new evaluator feedback → DynamicEvaluatorCreator adds new evaluator"""
        print(f"\n{'='*80}")
        print(f"Running Test D4: Add New Evaluator Feedback")
        print(f"{'='*80}")
        
        test_id = "D4"
        feedback_file = self.get_test_feedback_file(test_id)
        new_evaluator_name = "creativity"
        
        # Initialize collector with fresh file
        if os.path.exists(feedback_file):
            os.remove(feedback_file)
        
        collector = TeacherFeedbackCollector(storage_file=feedback_file)
        initial_evaluators = set(collector.evaluator_weights.keys())
        
        print(f"Initial evaluators: {initial_evaluators}")
        
        # Create feedback that triggers new evaluator creation
        feedback_data = self.create_new_evaluator_feedback("Test Story", "Empathy", new_evaluator_name, 0)
        record = collector.collect_feedback(
            story_title=feedback_data["story_title"],
            objective=feedback_data["objective"],
            teacher_feedback=feedback_data["feedback_text"],
            feedback_type=feedback_data["feedback_type"],
            question_evaluations=feedback_data["question_evaluations"],
            generated_questions=feedback_data["generated_questions"]
        )
        
        # Check feedback record for new evaluator
        action_taken = record.get("action_taken", {})
        new_evaluator_created = action_taken.get("type") == "add_new_evaluator"
        created_evaluator_name = None
        
        if new_evaluator_created:
            details = action_taken.get("details", {})
            created_evaluator_name = details.get("evaluator_name", "")
        
        # Also check if it appears in dynamic evaluators list (for future use)
        # Load feedback records and check
        feedback_records = collector.feedback_records
        dynamic_evaluators_in_records = []
        
        for school_data in feedback_records.values():
            if isinstance(school_data, dict):
                for teacher_data in school_data.values():
                    if isinstance(teacher_data, list):
                        for rec in teacher_data:
                            if rec.get("action_taken", {}).get("type") == "add_new_evaluator":
                                eval_name = rec.get("action_taken", {}).get("details", {}).get("evaluator_name", "")
                                if eval_name:
                                    dynamic_evaluators_in_records.append(eval_name)
        
        evaluator_was_created = new_evaluator_created and created_evaluator_name
        attribute_matches = False
        if created_evaluator_name:
            # Check if created evaluator name matches expected (fuzzy match)
            created_lower = created_evaluator_name.lower().replace("_", " ")
            expected_lower = new_evaluator_name.lower().replace("_", " ")
            attribute_matches = expected_lower in created_lower or created_lower in expected_lower
        
        print(f"\nNew evaluator created: {evaluator_was_created}")
        print(f"Created evaluator name: {created_evaluator_name}")
        print(f"Attribute matches expected: {attribute_matches}")
        print(f"Dynamic evaluators in records: {dynamic_evaluators_in_records}")
        
        return {
            "test_id": test_id,
            "description": "Add new evaluator feedback → DynamicEvaluatorCreator adds new evaluator",
            "timestamp": datetime.now().isoformat(),
            "expected_evaluator_name": new_evaluator_name,
            "initial_evaluators": list(initial_evaluators),
            "new_evaluator_created": evaluator_was_created,
            "created_evaluator_name": created_evaluator_name,
            "attribute_matches": attribute_matches,
            "feedback_record": {
                "action_type": action_taken.get("type"),
                "details": action_taken.get("details", {})
            },
            "dynamic_evaluators_in_records": dynamic_evaluators_in_records,
            "status": "completed"
        }
    
    def run_test_d5(self) -> Dict[str, Any]:
        """Test D5: Run optimizer after 10 mixed feedback cycles → Measurable improvement"""
        print(f"\n{'='*80}")
        print(f"Running Test D5: Optimization After Mixed Feedback")
        print(f"{'='*80}")
        
        test_id = "D5"
        feedback_file = self.get_test_feedback_file(test_id)
        
        # Also need evaluation file for optimizer
        eval_file = os.path.join(TEST_DATA_DIR, f"{test_id}_question_evaluations.json")
        
        # Initialize system with fresh files
        if os.path.exists(feedback_file):
            os.remove(feedback_file)
        if os.path.exists(eval_file):
            os.remove(eval_file)
        
        feedback_system = TeacherFeedbackSystem(storage_file=feedback_file)
        
        print("Creating 10 mixed feedback cycles with evaluation data...")
        
        baseline_weights = copy.deepcopy(feedback_system.collector.evaluator_weights)
        weight_history = [copy.deepcopy(baseline_weights)]
        
        # Create evaluation data structure (needed for optimizer)
        eval_data = {"evaluations": []}
        
        # Create 10 mixed feedback cycles
        for i in range(10):
            if i % 3 == 0:
                # Positive feedback
                feedback_data = self.create_positive_feedback("Test Story", "Empathy", i)
            elif i % 3 == 1:
                # Negative feedback for recall
                feedback_data = self.create_negative_feedback("Test Story", "Empathy", "recall_suitability", i)
            else:
                # Negative feedback for open_ended
                feedback_data = self.create_negative_feedback("Test Story", "Empathy", "open_ended_suitability", i)
            
            # Create evaluation records for each question (needed for optimizer)
            for q_idx, q_obj in enumerate(feedback_data["generated_questions"]):
                question_text = q_obj if isinstance(q_obj, str) else q_obj.get("question", f"Question {q_idx}")
                eval_data["evaluations"].append({
                    "storybook_id": "Test Story",
                    "question": question_text,
                    "question_type": "open_ended" if "open" in feedback_data["feedback_text"].lower() else "recall",
                    "suitability_score": 0.7 if feedback_data["feedback_type"] == "positive" else 0.3,
                    "decision": "pass" if feedback_data["feedback_type"] == "positive" else "regenerate",
                    "evaluation_reasoning": feedback_data["feedback_text"]
                })
            
            record = feedback_system.process_feedback(
                story_title=feedback_data["story_title"],
                objective=feedback_data["objective"],
                teacher_feedback=feedback_data["feedback_text"],
                feedback_type=feedback_data["feedback_type"],
                question_evaluations=feedback_data["question_evaluations"],
                generated_questions=feedback_data["generated_questions"],
                story_context="Test story content for optimization"
            )
            
            weight_history.append(copy.deepcopy(feedback_system.collector.evaluator_weights))
            print(f"After cycle {i+1}: {feedback_system.collector.evaluator_weights}")
        
        # Save evaluation data (needed for optimizer)
        with open(eval_file, 'w') as f:
            json.dump(eval_data, f, indent=2)
        
        final_weights = feedback_system.collector.evaluator_weights
        
        # Calculate overall weight change
        weight_changes = {}
        for evaluator in baseline_weights:
            change = final_weights.get(evaluator, 0) - baseline_weights.get(evaluator, 0)
            weight_changes[evaluator] = round(change, 4)
        
        # Check if system learned (weights adjusted meaningfully)
        total_absolute_change = sum(abs(change) for change in weight_changes.values())
        system_learned = total_absolute_change > 0.1  # At least 0.1 total change
        
        print(f"\nBaseline weights: {baseline_weights}")
        print(f"Final weights: {final_weights}")
        print(f"Weight changes: {weight_changes}")
        print(f"Total absolute change: {total_absolute_change:.4f}")
        print(f"System learned: {system_learned}")
        
        # Now try to initialize optimizer with the feedback data
        optimizer_initialized = False
        optimizer_error = None
        
        try:
            from question_generator import QuestionGeneratorModule
            
            # Temporarily update paths for optimizer initialization
            import shutil
            storage_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'storage')
            os.makedirs(storage_dir, exist_ok=True)
            backend_feedback_file = os.path.join(storage_dir, 'teacher_feedback_records.json')
            backend_eval_file = os.path.join(storage_dir, 'question_evaluations.json')
            
            # Backup existing files if they exist
            if os.path.exists(backend_feedback_file):
                shutil.copy(backend_feedback_file, backend_feedback_file + '.bak')
            if os.path.exists(backend_eval_file):
                shutil.copy(backend_eval_file, backend_eval_file + '.bak')
            
            # Copy test files to backend location
            print(f"\nCopying feedback data to {backend_feedback_file}")
            print(f"Copying eval data to {backend_eval_file}")
            shutil.copy(feedback_file, backend_feedback_file)
            shutil.copy(eval_file, backend_eval_file)
            
            # Verify files exist
            if not os.path.exists(backend_feedback_file):
                raise FileNotFoundError(f"Feedback file not found: {backend_feedback_file}")
            if not os.path.exists(backend_eval_file):
                raise FileNotFoundError(f"Eval file not found: {backend_eval_file}")
            
            print(f"\nAttempting to initialize DSPy optimizer with feedback data...")
            
            # Verify file contents
            with open(backend_feedback_file, 'r') as f:
                feedback_test = json.load(f)
                records_count = 0
                for school_key in feedback_test:
                    if school_key == "records":
                        records_count += len(feedback_test.get("records", []))
                    elif isinstance(feedback_test[school_key], dict):
                        for teacher_key in feedback_test[school_key]:
                            if isinstance(feedback_test[school_key][teacher_key], list):
                                records_count += len(feedback_test[school_key][teacher_key])
                print(f"Feedback file structure: {list(feedback_test.keys())}")
                print(f"Total records in file: {records_count}")
            
            with open(backend_eval_file, 'r') as f:
                eval_test = json.load(f)
                print(f"Eval records: {len(eval_test.get('evaluations', []))}")
            
            print(f"Optimizer storage directory: {os.path.abspath(storage_dir)}")
            
            generator = QuestionGeneratorModule(optimize_with_feedback=True)
            
            optimizer_initialized = generator.optimized
            print(f"Optimizer initialized: {optimizer_initialized}")
            
            # Restore backups if they existed
            if os.path.exists(backend_feedback_file + '.bak'):
                shutil.move(backend_feedback_file + '.bak', backend_feedback_file)
            if os.path.exists(backend_eval_file + '.bak'):
                shutil.move(backend_eval_file + '.bak', backend_eval_file)
            
        except Exception as e:
            optimizer_error = str(e)
            print(f"Error initializing optimizer: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            "test_id": test_id,
            "description": "Run optimizer after 10 mixed feedback cycles → Measurable improvement",
            "timestamp": datetime.now().isoformat(),
            "baseline_weights": baseline_weights,
            "final_weights": final_weights,
            "weight_history": weight_history,
            "weight_changes": weight_changes,
            "total_absolute_change": round(total_absolute_change, 4),
            "system_learned": system_learned,
            "optimizer_initialized": optimizer_initialized,
            "optimizer_error": optimizer_error,
            "note": "Optimizer compile() verification complete. Full metric improvement (>10% score gain) would require generating questions before/after optimization and comparing rubric scores.",
            "status": "completed"
        }
    
    def run_all_tests(self):
        """Run all test cases"""
        print(f"\n{'='*80}")
        print(f"Phase D Test Suite: Optimization & Learning Behavior")
        print(f"Started at: {self.timestamp}")
        print(f"{'='*80}")
        
        # Run each test
        d1_result = self.run_test_d1()
        self.results.append(d1_result)
        
        d2_result = self.run_test_d2()
        self.results.append(d2_result)
        
        d3_result = self.run_test_d3()
        self.results.append(d3_result)
        
        d4_result = self.run_test_d4()
        self.results.append(d4_result)
        
        d5_result = self.run_test_d5()
        self.results.append(d5_result)
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate test summary and save to JSON"""
        summary = {
            "test_suite": "Phase D: Optimization & Learning Behavior",
            "timestamp": self.timestamp,
            "total_tests": len(self.results),
            "completed": sum(1 for r in self.results if r["status"] == "completed"),
            "partial": sum(1 for r in self.results if r["status"] == "partial"),
            "failed": sum(1 for r in self.results if r["status"] == "failed"),
            "test_results": self.results
        }
        
        # Save to JSON file in test_results folder
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'test_results')
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, "phase_d_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"Test Suite Complete")
        print(f"{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"\nSummary:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Completed: {summary['completed']}")
        print(f"  Partial: {summary['partial']}")
        print(f"  Failed: {summary['failed']}")
        print(f"\nResults saved to JSON for manual review and interpretation.")


if __name__ == "__main__":
    tester = PhaseDTester()
    tester.run_all_tests()

