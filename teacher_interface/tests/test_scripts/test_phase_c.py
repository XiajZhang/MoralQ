"""
Phase C: Orchestrator Interpretation Accuracy Test Suite

Purpose: Test how well the orchestrator understands natural, ambiguous teacher feedback.

Test Cases:
- C1: "Great questions — keep them this way." → reinforce_existing
- C2: "They're too factual; add more reasoning." → adjust_evaluator, recall_suitability
- C3: "These need emotional depth." → add_new_evaluator, emotional_resonance
- C4: "Questions were fine, but too complex for my group." → adjust_evaluator, open_ended_suitability, decrease
- C5: "These are off-topic." → adjust_evaluator, relevance, increase

Evaluation metric: Accuracy of mapping (≥85%) and confidence threshold consistency.
"""

import os
import sys
import json
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
from teacher_feedback_system import FeedbackOrchestrator

# Test cases definition
TEST_CASES = {
    "C1": {
        "test_id": "C1",
        "teacher_feedback": "Great questions — keep them this way.",
        "story_context": "Story: Ada Twist, Scientist\nObjective: Teaching curiosity and scientific thinking",
        "existing_evaluators": [
            "completion_suitability", "recall_suitability", "open_ended_suitability", 
            "wh_suitability", "distancing_suitability"
        ],
        "expected_action_type": "reinforce_existing",
        "expected_attribute": None,  # Not critical for reinforce
        "expected_adjustment_direction": "none",
        "expected_confidence_min": 0.8,
        "description": "Positive feedback should trigger reinforce_existing"
    },
    "C2": {
        "test_id": "C2",
        "teacher_feedback": "They're too factual; add more reasoning.",
        "story_context": "Story: Grumpy Monkey\nObjective: Emotional regulation",
        "existing_evaluators": [
            "completion_suitability", "recall_suitability", "open_ended_suitability", 
            "wh_suitability", "distancing_suitability"
        ],
        "expected_action_type": "adjust_evaluator",
        "expected_attribute": "recall_suitability",  # "Too factual" suggests recall questions need adjustment
        "expected_adjustment_direction": "increase",
        "expected_confidence_min": 0.7,
        "description": "Feedback about being too factual should adjust recall evaluator"
    },
    "C3": {
        "test_id": "C3",
        "teacher_feedback": "These need emotional depth.",
        "story_context": "Story: A Letter to Amy\nObjective: Empathy",
        "existing_evaluators": [
            "completion_suitability", "recall_suitability", "open_ended_suitability", 
            "wh_suitability", "distancing_suitability"
        ],
        "expected_action_type": "add_new_evaluator",
        "expected_attribute": "emotional_resonance",  # Or "emotional_depth" or similar
        "expected_adjustment_direction": "none",
        "expected_confidence_min": 0.7,
        "description": "Feedback about emotional depth should create new evaluator"
    },
    "C4": {
        "test_id": "C4",
        "teacher_feedback": "Questions were fine, but too complex for my group.",
        "story_context": "Story: If You Give a Mouse a Cookie\nObjective: Consequence awareness",
        "existing_evaluators": [
            "completion_suitability", "recall_suitability", "open_ended_suitability", 
            "wh_suitability", "distancing_suitability", "complexity"
        ],
        "expected_action_type": "adjust_evaluator",
        "expected_attribute": "complexity",  # Or "open_ended_suitability" if complexity doesn't exist
        "expected_adjustment_direction": "decrease",
        "expected_confidence_min": 0.7,
        "description": "Feedback about complexity should adjust complexity evaluator"
    },
    "C5": {
        "test_id": "C5",
        "teacher_feedback": "These are off-topic.",
        "story_context": "Story: Last Stop on Market Street\nObjective: Gratitude",
        "existing_evaluators": [
            "completion_suitability", "recall_suitability", "open_ended_suitability", 
            "wh_suitability", "distancing_suitability"
        ],
        "expected_action_type": "adjust_evaluator",
        "expected_attribute": "relevance",  # May need to check what attribute it maps to
        "expected_adjustment_direction": "increase",
        "expected_confidence_min": 0.7,
        "description": "Feedback about off-topic should adjust relevance"
    }
}

# Number of runs for consistency testing
NUM_RUNS = 3


class PhaseCTester:
    """Test suite for Phase C: Orchestrator Interpretation Accuracy"""
    
    def __init__(self):
        self.results = []
        self.timestamp = datetime.now().isoformat()
        # Initialize orchestrator
        self.orchestrator = FeedbackOrchestrator()
    
    def run_orchestrator_multiple_times(self, feedback_text: str, story_context: str, 
                                       existing_evaluators: List[str], num_runs: int = NUM_RUNS) -> List[Dict[str, Any]]:
        """Run orchestrator multiple times to check consistency"""
        results = []
        
        for run in range(num_runs):
            try:
                result = self.orchestrator(
                    feedback_text=feedback_text,
                    story_context=story_context,
                    current_questions=None,
                    existing_evaluators=existing_evaluators
                )
                results.append(result)
            except Exception as e:
                print(f"[ERROR] Run {run+1} failed: {e}")
                results.append({"error": str(e)})
        
        return results
    
    def check_action_type_match(self, actual: str, expected: str) -> bool:
        """Check if action type matches expected"""
        return actual.lower() == expected.lower()
    
    def check_attribute_match(self, actual: str, expected: Optional[str]) -> bool:
        """Check if attribute matches expected (fuzzy matching for variations)"""
        if expected is None:
            return True  # Not critical for reinforce actions
        
        actual_lower = actual.lower().replace("_", " ").replace("-", " ")
        expected_lower = expected.lower().replace("_", " ").replace("-", " ")
        
        # Exact match
        if actual_lower == expected_lower:
            return True
        
        # Partial match (e.g., "emotional_resonance" vs "emotional_depth")
        if expected_lower in actual_lower or actual_lower in expected_lower:
            return True
        
        # Check for semantic similarity (e.g., "recall" in attribute)
        if "recall" in expected_lower and "recall" in actual_lower:
            return True
        if "emotional" in expected_lower and "emotional" in actual_lower:
            return True
        if "complex" in expected_lower and "complex" in actual_lower:
            return True
        if "relevance" in expected_lower and ("relevance" in actual_lower or "open_ended" in actual_lower):
            return True
        
        return False
    
    def check_adjustment_direction_match(self, actual: str, expected: str) -> bool:
        """Check if adjustment direction matches expected"""
        if expected == "none":
            return actual.lower() in ["none", "maintain"]
        
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        if actual_lower == expected_lower:
            return True
        
        # Check for semantic equivalents
        if expected_lower == "increase" and actual_lower in ["increase", "strengthen"]:
            return True
        if expected_lower == "decrease" and actual_lower in ["decrease", "reduce", "weaken"]:
            return True
        
        return False
    
    def run_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case"""
        test_id = test_case["test_id"]
        feedback = test_case["teacher_feedback"]
        story_context = test_case["story_context"]
        existing_evaluators = test_case["existing_evaluators"]
        
        print(f"\n{'='*80}")
        print(f"Running Test {test_id}: {test_case['description']}")
        print(f"{'='*80}")
        print(f"Teacher Feedback: {feedback}")
        print(f"Expected Action Type: {test_case['expected_action_type']}")
        if test_case.get('expected_attribute'):
            print(f"Expected Attribute: {test_case['expected_attribute']}")
        if test_case.get('expected_adjustment_direction') != "none":
            print(f"Expected Adjustment Direction: {test_case['expected_adjustment_direction']}")
        print(f"\nRunning {NUM_RUNS} orchestrator calls for consistency...")
        
        result = {
            "test_id": test_id,
            "teacher_feedback": feedback,
            "story_context": story_context,
            "existing_evaluators": existing_evaluators,
            "description": test_case["description"],
            "expected_action_type": test_case["expected_action_type"],
            "expected_attribute": test_case.get("expected_attribute"),
            "expected_adjustment_direction": test_case.get("expected_adjustment_direction", "none"),
            "expected_confidence_min": test_case.get("expected_confidence_min", 0.5),
            "timestamp": datetime.now().isoformat(),
            "orchestrator_runs": [],
            "consistency_metrics": {},
            "accuracy_check": {},
            "status": "pending"
        }
        
        # Run orchestrator multiple times
        orchestrator_results = self.run_orchestrator_multiple_times(
            feedback, story_context, existing_evaluators, NUM_RUNS
        )
        result["orchestrator_runs"] = orchestrator_results
        
        # Extract and analyze results
        action_types = []
        attributes = []
        adjustment_directions = []
        confidences = []
        reformulated_instructions = []
        
        for i, orch_result in enumerate(orchestrator_results):
            if "error" in orch_result:
                print(f"[ERROR] Run {i+1} had an error")
                continue
            
            action_type = orch_result.get("action_type", "")
            attribute = orch_result.get("attribute", "")
            adj_direction = orch_result.get("adjustment_direction", "")
            confidence = orch_result.get("confidence", 0.0)
            reformulated = orch_result.get("reformulated_instruction", "")
            
            action_types.append(action_type)
            attributes.append(attribute)
            adjustment_directions.append(adj_direction)
            confidences.append(confidence)
            reformulated_instructions.append(reformulated)
            
            print(f"\nRun {i+1}:")
            print(f"  Action Type: {action_type}")
            print(f"  Attribute: {attribute}")
            print(f"  Adjustment Direction: {adj_direction}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Reformulated Instruction: {reformulated[:80]}...")
        
        # Calculate consistency
        action_type_consistency = len(set(action_types)) == 1 if action_types else False
        attribute_consistency = len(set(attributes)) == 1 if attributes else False
        adj_direction_consistency = len(set(adjustment_directions)) == 1 if adjustment_directions else False
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        confidence_consistent = max(confidences) - min(confidences) <= 0.2 if len(confidences) > 1 else True
        
        result["consistency_metrics"] = {
            "action_type_consistent": action_type_consistency,
            "action_types": list(set(action_types)) if action_types else [],
            "attribute_consistent": attribute_consistency,
            "attributes": list(set(attributes)) if attributes else [],
            "adjustment_direction_consistent": adj_direction_consistency,
            "adjustment_directions": list(set(adjustment_directions)) if adjustment_directions else [],
            "average_confidence": round(avg_confidence, 3),
            "confidence_range": [round(min(confidences), 3), round(max(confidences), 3)] if confidences else [],
            "confidence_consistent": confidence_consistent
        }
        
        # Check accuracy
        most_common_action = max(set(action_types), key=action_types.count) if action_types else ""
        most_common_attribute = max(set(attributes), key=attributes.count) if attributes else ""
        most_common_direction = max(set(adjustment_directions), key=adjustment_directions.count) if adjustment_directions else ""
        
        action_type_match = self.check_action_type_match(most_common_action, test_case["expected_action_type"])
        attribute_match = self.check_attribute_match(most_common_attribute, test_case.get("expected_attribute"))
        direction_match = self.check_adjustment_direction_match(
            most_common_direction, 
            test_case.get("expected_adjustment_direction", "none")
        )
        confidence_adequate = avg_confidence >= test_case.get("expected_confidence_min", 0.5)
        
        accuracy_score = sum([action_type_match, attribute_match, direction_match, confidence_adequate]) / 4.0
        
        result["accuracy_check"] = {
            "action_type_match": action_type_match,
            "actual_action_type": most_common_action,
            "attribute_match": attribute_match,
            "actual_attribute": most_common_attribute,
            "adjustment_direction_match": direction_match,
            "actual_adjustment_direction": most_common_direction,
            "confidence_adequate": confidence_adequate,
            "average_confidence": round(avg_confidence, 3),
            "expected_confidence_min": test_case.get("expected_confidence_min", 0.5),
            "overall_accuracy": round(accuracy_score, 3),
            "meets_threshold": accuracy_score >= 0.85
        }
        
        print(f"\nAccuracy Check:")
        print(f"  Action Type Match: {action_type_match} ({most_common_action})")
        print(f"  Attribute Match: {attribute_match} ({most_common_attribute})")
        print(f"  Adjustment Direction Match: {direction_match} ({most_common_direction})")
        print(f"  Confidence Adequate: {confidence_adequate} (avg: {avg_confidence:.3f})")
        print(f"  Overall Accuracy: {accuracy_score:.3f} ({'PASS' if accuracy_score >= 0.85 else 'FAIL'})")
        
        result["status"] = "completed"
        return result
    
    def run_all_tests(self):
        """Run all test cases"""
        print(f"\n{'='*80}")
        print(f"Phase C Test Suite: Orchestrator Interpretation Accuracy")
        print(f"Started at: {self.timestamp}")
        print(f"{'='*80}")
        
        for test_id, test_case in TEST_CASES.items():
            result = self.run_test_case(test_case)
            self.results.append(result)
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate test summary and save to JSON"""
        total_accuracy = sum(r["accuracy_check"]["overall_accuracy"] for r in self.results if "accuracy_check" in r)
        avg_accuracy = total_accuracy / len(self.results) if self.results else 0.0
        
        passed_threshold = sum(1 for r in self.results if r.get("accuracy_check", {}).get("meets_threshold", False))
        
        summary = {
            "test_suite": "Phase C: Orchestrator Interpretation Accuracy",
            "timestamp": self.timestamp,
            "num_runs_per_test": NUM_RUNS,
            "total_tests": len(self.results),
            "completed": sum(1 for r in self.results if r["status"] == "completed"),
            "partial": sum(1 for r in self.results if r["status"] == "partial"),
            "failed": sum(1 for r in self.results if r["status"] == "failed"),
            "average_accuracy": round(avg_accuracy, 3),
            "tests_meeting_threshold": passed_threshold,
            "threshold": 0.85,
            "test_results": self.results
        }
        
        # Save to JSON file in test_results folder
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'test_results')
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, "phase_c_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"Test Suite Complete")
        print(f"{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"\nSummary:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Completed: {summary['completed']}")
        print(f"  Average Accuracy: {summary['average_accuracy']:.3f}")
        print(f"  Tests Meeting Threshold (≥0.85): {passed_threshold}/{summary['total_tests']}")
        print(f"\nResults saved to JSON for manual review and interpretation.")


if __name__ == "__main__":
    tester = PhaseCTester()
    tester.run_all_tests()

