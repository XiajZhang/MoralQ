"""
Phase B: Evaluator Scoring & Consistency Test Suite

Purpose: Ensure rubric evaluators produce consistent scores (1–5) for similar quality questions.

Test Cases:
- B1: "Why do you think Ada keeps asking questions?" → open_ended_suitability = 5
- B2: "What color was Ada's dress?" → recall_suitability = 5; others ≈ 1–2
- B3: "What did Ada's parents do when she made a mess?" → wh_suitability = 4–5
- B4: "How would you feel if your experiment failed?" → distancing_suitability = 5
- B5: Random ungrammatical or off-topic question → All scores ≤ 2

Evaluation metric: Stability across repeated runs (std deviation ≤ 0.3).
"""

import os
import sys
import json
import statistics
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
from contextq_evaluators import ContextQEvaluationPipeline

# Test cases definition
TEST_CASES = {
    "B1": {
        "test_id": "B1",
        "question": "Why do you think Ada keeps asking questions?",
        "story_context": "Ada Twist, Scientist is a story about a curious young girl who asks many questions about the world around her.",
        "expected_evaluator": "open_ended_suitability",
        "expected_score_range": [4, 5],
        "score_scale": "1-5",  # Open-Ended uses 1-5 scale
        "expected_other_scores": "≈ 1–2",
        "description": "Open-ended question requiring elaboration and opinion"
    },
    "B2": {
        "test_id": "B2",
        "question": "What color was Ada's dress?",
        "story_context": "Ada Twist, Scientist is a story about a curious young girl who asks many questions about the world around her.",
        "expected_evaluator": "recall_suitability",
        "expected_score_range": [0.8, 1.0],  # Recall uses 0-1 scale (binary: 1 if criteria met)
        "score_scale": "0-1",  # Recall uses 0-1 scale
        "expected_other_scores": "≈ 0–0.5",
        "description": "Recall question about plot details"
    },
    "B3": {
        "test_id": "B3",
        "question": "What did Ada's parents do when she made a mess?",
        "story_context": "Ada Twist, Scientist is a story about a curious young girl who asks many questions about the world around her.",
        "expected_evaluator": "wh_suitability",
        "expected_score_range": [0.8, 1.0],  # Wh uses 0-1 scale (binary: 1 if criteria met)
        "score_scale": "0-1",  # Wh uses 0-1 scale
        "expected_other_scores": "N/A",
        "description": "Wh- question about story details"
    },
    "B4": {
        "test_id": "B4",
        "question": "How would you feel if your experiment failed?",
        "story_context": "Ada Twist, Scientist is a story about a curious young girl who asks many questions about the world around her.",
        "expected_evaluator": "distancing_suitability",
        "expected_score_range": [4, 5],
        "score_scale": "1-5",  # Distancing uses 1-5 scale
        "expected_other_scores": "N/A",
        "description": "Distancing question connecting to child's experience"
    },
    "B5": {
        "test_id": "B5",
        "question": "What color was Ada's dress? xyz abc random ungrammatical nonsense off-topic",
        "story_context": "Ada Twist, Scientist is a story about a curious young girl who asks many questions about the world around her.",
        "expected_evaluator": "all",
        "expected_score_range": [0, 0.5],  # Should be low regardless of scale
        "score_scale": "varies",  # Depends on which evaluator runs
        "expected_other_scores": "All low",
        "description": "Ungrammatical/off-topic question should score low"
    }
}

# Number of runs for consistency testing
NUM_RUNS = 5


class PhaseBTester:
    """Test suite for Phase B: Evaluator Scoring & Consistency"""
    
    def __init__(self):
        self.results = []
        self.timestamp = datetime.now().isoformat()
        # Initialize evaluator pipeline
        self.evaluator = ContextQEvaluationPipeline(
            storage_file=None,  # Don't store evaluations for testing
            feedback_records_file=None,
            dynamic_evaluators=None
        )
    
    def evaluate_question_multiple_times(self, question: str, story_context: str = "", num_runs: int = NUM_RUNS) -> List[Dict[str, Any]]:
        """Evaluate a question multiple times to check consistency"""
        results = []
        
        for run in range(num_runs):
            try:
                # Use the suitability program's forward method directly
                eval_result = self.evaluator.suitability_program.forward(
                    question=question,
                    story_context=story_context or ""
                )
                results.append(eval_result)
            except Exception as e:
                print(f"[ERROR] Run {run+1} failed: {e}")
                results.append({"error": str(e)})
        
        return results
    
    def extract_scores(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract scores and metadata from evaluation result"""
        result = {
            "question_type": evaluation_result.get("question_type", "unknown"),
            "suitability_score": evaluation_result.get("suitability_score", 0),
            "decision": evaluation_result.get("decision", "unknown"),
            "reasoning": evaluation_result.get("evaluation_reasoning", ""),
            "type_confidence": evaluation_result.get("type_confidence", 0),
            "type_reasoning": evaluation_result.get("type_reasoning", "")
        }
        
        # Map question type to evaluator name and scale
        type_to_evaluator = {
            "completion": ("completion_suitability", "0-1"),
            "recall": ("recall_suitability", "0-1"),
            "open-ended": ("open_ended_suitability", "1-5"),
            "open_ended": ("open_ended_suitability", "1-5"),
            "wh": ("wh_suitability", "0-1"),
            "distancing": ("distancing_suitability", "1-5")
        }
        
        # Get the primary evaluator name and scale
        question_type_normalized = result["question_type"].lower().replace("-", "_").replace(" ", "_")
        primary_evaluator = None
        score_scale = None
        for type_key, (evaluator_name, scale) in type_to_evaluator.items():
            if type_key in question_type_normalized:
                primary_evaluator = evaluator_name
                score_scale = scale
                break
        
        if primary_evaluator:
            result["primary_evaluator"] = primary_evaluator
            result["evaluator_score"] = result["suitability_score"]
            result["score_scale"] = score_scale
        else:
            result["primary_evaluator"] = "unknown"
            result["evaluator_score"] = result["suitability_score"]
            result["score_scale"] = "unknown"
        
        # Get dynamic evaluator scores if available
        dynamic_evaluations = evaluation_result.get("dynamic_evaluations", {})
        if dynamic_evaluations:
            result["dynamic_evaluations"] = dynamic_evaluations
        
        return result
    
    def calculate_consistency(self, scores_list: List[float]) -> Dict[str, float]:
        """Calculate consistency metrics for a list of scores"""
        if not scores_list or len(scores_list) < 2:
            return {
                "mean": scores_list[0] if scores_list else 0.0,
                "std_dev": 0.0,
                "min": scores_list[0] if scores_list else 0.0,
                "max": scores_list[0] if scores_list else 0.0,
                "range": 0.0,
                "is_stable": True  # Single value or empty is considered stable
            }
        
        mean = statistics.mean(scores_list)
        std_dev = statistics.stdev(scores_list) if len(scores_list) > 1 else 0.0
        min_score = min(scores_list)
        max_score = max(scores_list)
        score_range = max_score - min_score
        
        # Stability: std deviation ≤ 0.3
        is_stable = std_dev <= 0.3
        
        return {
            "mean": round(mean, 3),
            "std_dev": round(std_dev, 3),
            "min": round(min_score, 3),
            "max": round(max_score, 3),
            "range": round(score_range, 3),
            "is_stable": is_stable,
            "scores": [round(s, 3) for s in scores_list]
        }
    
    def run_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case"""
        test_id = test_case["test_id"]
        question = test_case["question"]
        story_context = test_case.get("story_context", "")
        expected_evaluator = test_case["expected_evaluator"]
        expected_range = test_case["expected_score_range"]
        
        print(f"\n{'='*80}")
        print(f"Running Test {test_id}: {test_case['description']}")
        print(f"{'='*80}")
        print(f"Question: {question}")
        print(f"Expected Evaluator: {expected_evaluator}")
        print(f"Expected Score Range: {expected_range}")
        print(f"Score Scale: {test_case.get('score_scale', 'varies')}")
        print(f"\nRunning {NUM_RUNS} evaluations for consistency...")
        
        result = {
            "test_id": test_id,
            "question": question,
            "story_context": story_context,
            "description": test_case["description"],
            "expected_evaluator": expected_evaluator,
            "expected_score_range": expected_range,
            "score_scale": test_case.get("score_scale", "varies"),
            "expected_other_scores": test_case.get("expected_other_scores", "N/A"),
            "timestamp": datetime.now().isoformat(),
            "evaluation_runs": [],
            "scores_by_evaluator": {},
            "consistency_metrics": {},
            "status": "pending"
        }
        
        # Run multiple evaluations
        evaluation_results = self.evaluate_question_multiple_times(question, story_context, NUM_RUNS)
        result["evaluation_runs"] = evaluation_results
        
        # Extract scores and metadata for the expected evaluator
        all_runs_data = []
        all_scores = {}
        
        for i, eval_result in enumerate(evaluation_results):
            if "error" in eval_result:
                print(f"[ERROR] Run {i+1} had an error")
                all_runs_data.append({"run": i+1, "error": eval_result["error"]})
                continue
            
            run_data = self.extract_scores(eval_result)
            run_data["run"] = i + 1
            all_runs_data.append(run_data)
            
            primary_evaluator = run_data.get("primary_evaluator", "unknown")
            score = run_data.get("evaluator_score", 0)
            
            print(f"\nRun {i+1}:")
            print(f"  Question Type: {run_data['question_type']}")
            print(f"  Primary Evaluator: {primary_evaluator}")
            print(f"  Score: {score:.3f} (scale: {run_data.get('score_scale', 'unknown')})")
            print(f"  Decision: {run_data['decision']}")
            
            # Store scores by evaluator
            if primary_evaluator not in all_scores:
                all_scores[primary_evaluator] = []
            all_scores[primary_evaluator].append(score)
        
        result["run_details"] = all_runs_data
        
        # Calculate consistency for each evaluator
        result["scores_by_evaluator"] = {}
        result["consistency_metrics"] = {}
        
        for evaluator_name, scores_list in all_scores.items():
            consistency = self.calculate_consistency(scores_list)
            result["scores_by_evaluator"][evaluator_name] = scores_list
            result["consistency_metrics"][evaluator_name] = consistency
            
            print(f"\n{evaluator_name}:")
            print(f"  Scores: {scores_list}")
            print(f"  Mean: {consistency['mean']:.3f}")
            print(f"  Std Dev: {consistency['std_dev']:.3f}")
            print(f"  Range: {consistency['range']:.3f}")
            print(f"  Stable (≤0.3): {consistency['is_stable']}")
        
        # Check if expected evaluator scores are in expected range
        # First, check what evaluators actually ran
        evaluators_that_ran = list(all_scores.keys())
        
        print(f"\nEvaluators that ran: {evaluators_that_ran}")
        
        if expected_evaluator != "all":
            # Check if the expected evaluator ran
            if expected_evaluator in all_scores:
                expected_scores = all_scores[expected_evaluator]
                mean_score = statistics.mean(expected_scores) if expected_scores else 0
                in_range = expected_range[0] <= mean_score <= expected_range[1]
                
                print(f"\nExpected Evaluator Check ({expected_evaluator}):")
                print(f"  Mean score: {mean_score:.3f}")
                print(f"  Expected range: {expected_range}")
                print(f"  In range: {in_range}")
                
                result["expected_evaluator_check"] = {
                    "evaluator": expected_evaluator,
                    "mean_score": round(mean_score, 3),
                    "in_expected_range": in_range,
                    "expected_range": expected_range,
                    "actual_scores": expected_scores
                }
            else:
                # Expected evaluator didn't run - check what did
                print(f"\n[NOTE] Expected evaluator '{expected_evaluator}' did not run.")
                print(f"  Actual evaluators: {evaluators_that_ran}")
                
                result["expected_evaluator_check"] = {
                    "evaluator": expected_evaluator,
                    "did_not_run": True,
                    "actual_evaluators": evaluators_that_ran,
                    "note": "Expected evaluator was not selected by type classifier"
                }
        else:
            # For B5, check that all scores are low
            all_means = {}
            for evaluator_name, scores_list in all_scores.items():
                if scores_list:
                    all_means[evaluator_name] = round(statistics.mean(scores_list), 3)
            
            all_low = all(mean <= expected_range[1] for mean in all_means.values()) if all_means else False
            
            print(f"\nAll Evaluators Check (should be ≤ {expected_range[1]}):")
            for evaluator_name, mean in all_means.items():
                print(f"  {evaluator_name}: {mean:.3f}")
            print(f"  All scores low: {all_low}")
            
            result["expected_evaluator_check"] = {
                "all_means": all_means,
                "all_low": all_low,
                "expected_max": expected_range[1]
            }
        
        result["status"] = "completed"
        return result
    
    def run_all_tests(self):
        """Run all test cases"""
        print(f"\n{'='*80}")
        print(f"Phase B Test Suite: Evaluator Scoring & Consistency")
        print(f"Started at: {self.timestamp}")
        print(f"{'='*80}")
        
        for test_id, test_case in TEST_CASES.items():
            result = self.run_test_case(test_case)
            self.results.append(result)
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate test summary and save to JSON"""
        summary = {
            "test_suite": "Phase B: Evaluator Scoring & Consistency",
            "timestamp": self.timestamp,
            "num_runs_per_test": NUM_RUNS,
            "total_tests": len(self.results),
            "completed": sum(1 for r in self.results if r["status"] == "completed"),
            "partial": sum(1 for r in self.results if r["status"] == "partial"),
            "failed": sum(1 for r in self.results if r["status"] == "failed"),
            "test_results": self.results
        }
        
        # Save to JSON file in test_results folder
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'test_results')
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, "phase_b_results.json")
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
    tester = PhaseBTester()
    tester.run_all_tests()

