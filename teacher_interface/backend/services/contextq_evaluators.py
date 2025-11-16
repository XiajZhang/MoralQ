"""
ContextQ Suitability-Based Evaluator
=====================================

A suitability-based approach.
Uses the ContextQ Suitability Rubric to evaluate questions based on pedagogical criteria only.

Based on: "ContextQ: Generated Questions to Support Meaningful Parent-Child Dialogue While Co-Reading"
"""

import dspy
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from teacher_interface.backend.models import EvaluationLogModel, EvaluationResultModel


# ============================================================================
# SUITABILITY-BASED EVALUATION
# ============================================================================

# Configure DSPy with OpenAI
dspy.configure(lm=dspy.LM("openai/gpt-4.1-2025-04-14"))


#
# LEGACY NOTE:
# Signatures are now defined centrally in
# teacher_interface.backend.evaluators.suitability_evaluators.
# This module imports and uses them to avoid duplication.
#
from teacher_interface.backend.evaluators.suitability_evaluators import (
    TypeClassificationSignature,
    CompletionSuitabilitySignature,
    RecallSuitabilitySignature,
    OpenEndedSuitabilitySignature,
    WhSuitabilitySignature,
    DistancingSuitabilitySignature,
)


# ============================================================================
# DYNAMIC EVALUATOR SIGNATURE (Type-Agnostic)
# ============================================================================

class UniversalEvaluatorSignature(dspy.Signature):
    """Legacy fallback signature used before templated dynamic evaluators.
    Currently unused because dynamic_template creates evaluator-specific signatures."""
    question = dspy.InputField(desc="The question to evaluate")
    story_context = dspy.InputField(desc="The story context for the question")
    question_type = dspy.InputField(desc="Type of question (Completion, Recall, Open-Ended, Wh, Distancing)")
    evaluator_criteria = dspy.InputField(desc="The specific criteria this evaluator is checking for (e.g., 'Assess if the question uses simple language appropriate for 4-6 year olds and avoids complex vocabulary.')")
    score = dspy.OutputField(desc="Numerical score from 1-5, where 5 means the question excellently meets the criteria and 1 means it does not meet it at all")
    reasoning = dspy.OutputField(desc="Detailed explanation of how well the question meets the criteria, with specific examples from the question")
    pass_decision = dspy.OutputField(desc="Output 'pass' if the score is 3 or higher (question meets the criteria), otherwise output 'regenerate'")


# ============================================================================
# SUITABILITY EVALUATION PROGRAM
# ============================================================================

class SuitabilityEvaluationProgram(dspy.Module):
    """Routes questions to appropriate suitability agent based on their type."""
    
    def __init__(self, dynamic_evaluators: List[Dict[str, Any]] = None):
        super().__init__()
        self.type_agent = dspy.Predict(TypeClassificationSignature)
        self.completion_agent = dspy.Predict(CompletionSuitabilitySignature)
        self.recall_agent = dspy.Predict(RecallSuitabilitySignature)
        self.open_ended_agent = dspy.Predict(OpenEndedSuitabilitySignature)
        self.wh_agent = dspy.Predict(WhSuitabilitySignature)
        self.distancing_agent = dspy.Predict(DistancingSuitabilitySignature)
        
        # Store dynamic evaluators (created from teacher feedback)
        self.dynamic_evaluators = dynamic_evaluators or []
        if self.dynamic_evaluators:
            for ev in self.dynamic_evaluators:
                print(f"   Evaluator in program: {ev}")
    
    def forward(self, question: str, story_context: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate question suitability based on ContextQ rubric."""
        
        # Step 1: Classify question type
        type_result = self.type_agent(question=question)
        question_type = type_result.question_type.replace("_", "-")  # Normalize naming
        
        # Step 2: Route to appropriate suitability agent
        suitability_result = None
        
        try:
            if question_type == "Completion":
                suitability_result = self.completion_agent(question=question)
            elif question_type == "Recall":
                suitability_result = self.recall_agent(question=question)
            elif question_type == "Open-Ended":
                suitability_result = self.open_ended_agent(question=question)
            elif question_type == "Wh":
                suitability_result = self.wh_agent(question=question)
            elif question_type == "Distancing":
                suitability_result = self.distancing_agent(question=question)
            else:
                # Unknown type - skip evaluation
                suitability_result = {
                    "suitability_score": 0.5,
                    "decision": "skip",
                    "reasoning": f"Unknown question type: {question_type}"
                }
        except Exception as e:
            print(f"Error evaluating suitability: {e}")
            suitability_result = {
                "suitability_score": 0.0,
                "decision": "error",
                "reasoning": str(e)
            }
        
        # Extract suitability_score and ensure proper type conversion
        if isinstance(suitability_result, dict):
            score = suitability_result.get("suitability_score", 0.0)
            decision = suitability_result.get("decision", "unknown")
            reasoning = suitability_result.get("reasoning", "")
        else:
            score = suitability_result.suitability_score
            decision = suitability_result.decision
            reasoning = suitability_result.reasoning
        
        # Convert score to float if string (for 1-5 scale)
        try:
            if isinstance(score, str):
                score = float(score)
            else:
                score = float(score)
        except (ValueError, TypeError):
            score = 0.0
        
        # Override decision logic for ALL question types
        
        # Open-Ended and Distancing: numeric scores must be >= 3
        if question_type in ["Open-Ended", "Distancing"]:
            if score < 3:
                decision = "regenerate"
                reasoning += f" Score {score} is below threshold of 3."
        
        # Binary types (Completion, Recall, Wh-): score must be exactly 1.0 (all True)
        elif question_type in ["Completion", "Recall", "Wh"]:
            if score < 1.0:
                decision = "regenerate"
                reasoning += f" Score {score} indicates criteria not met."
        
        # Step 3: Apply ALL dynamic evaluators to this question (type-agnostic)
        dynamic_evaluations = {}
        if self.dynamic_evaluators:
            universal_agent = dspy.Predict(UniversalEvaluatorSignature)
            for evaluator in self.dynamic_evaluators:
                evaluator_name = evaluator.get("name", "unknown")
                evaluator_criteria = evaluator.get("criteria", "")
                
                try:
                    result = universal_agent(
                        question=question,
                        story_context=story_context or "",
                        question_type=question_type,
                        evaluator_criteria=evaluator_criteria
                    )
                    
                    # Extract score (1-5)
                    dynamic_score = float(result.score)
                    # Use pass_decision from result, or default based on score >= 3
                    dynamic_decision = result.pass_decision if hasattr(result, 'pass_decision') else ("pass" if dynamic_score >= 3 else "regenerate")
                    
                    dynamic_evaluations[evaluator_name] = {
                        "score": dynamic_score,
                        "decision": dynamic_decision,
                        "reasoning": result.reasoning
                    }
                    
                    print(f"[INFO] Dynamic evaluator '{evaluator_name}' result: score={dynamic_score}, decision={dynamic_decision}")
                    
                    # If any dynamic evaluator fails, mark for regeneration
                    if dynamic_decision == "regenerate":
                        decision = "regenerate"
                        reasoning += f"\n[{evaluator_name}]: Score {dynamic_score} below threshold."
                        
                except Exception as e:
                    print(f"[WARN]  Error applying dynamic evaluator {evaluator_name}: {e}")
        
        return {
            "question": question,
            "question_type": question_type,
            "type_confidence": type_result.confidence,
            "type_reasoning": type_result.reasoning,
            "suitability_score": score,
            "decision": decision,
            "evaluation_reasoning": reasoning,
            "dynamic_evaluations": dynamic_evaluations,  # Store dynamic evaluator results
            "details": {}  # Don't store suitability_result as it contains non-serializable objects
        }


# ============================================================================
# EVALUATION MANAGER
# ============================================================================

class EvaluationManager:
    """
    Manages storage/retrieval of suitability evaluation logs only.
    (Not the same as evaluator weight manager in backend/evaluators/manager.py.)
    """
    
    def __init__(self, storage_file: str = None):
        storage_dir = os.path.join(os.path.dirname(__file__), "..", "storage")
        os.makedirs(storage_dir, exist_ok=True)
        self.storage_file = storage_file or os.path.join(
            storage_dir,
            "question_evaluations.json"
        )
        self.evaluations = self._load_evaluations()
    
    def _load_evaluations(self) -> Dict[str, Any]:
        """Load existing evaluations from storage file."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                    if isinstance(raw, dict):
                        log = EvaluationLogModel.parse_obj(raw)
                    else:
                        log = EvaluationLogModel.parse_obj({"evaluations": raw})
                    return log.dict()
            except Exception as e:
                print(f"Error loading evaluations: {e}")
                return {}
        return {}
    
    def _save_evaluations(self):
        """Save evaluations to storage file."""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.evaluations, f, indent=2, ensure_ascii=False)
            print(f"Suitability evaluations saved to {self.storage_file}")
        except Exception as e:
            print(f"[ERROR] DEBUG: Error saving evaluations: {e}")
            print(f"Error saving evaluations: {e}")
            import traceback
            traceback.print_exc()
    
    def store_evaluations(self, evaluations: List[Dict[str, Any]], clear_existing: bool = False):
        """Store evaluation results."""
        if clear_existing:
            self.evaluations = {"evaluations": []}
        
        if "evaluations" not in self.evaluations:
            self.evaluations["evaluations"] = []
        
        validated = [
            EvaluationResultModel.parse_obj(record).dict(exclude_none=True)
            for record in evaluations
        ]
        self.evaluations["evaluations"].extend(validated)
        self._save_evaluations()
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        eval_list = self.evaluations.get("evaluations", [])
        
        if not eval_list:
            return {"total": 0, "by_type": {}, "by_decision": {}}
        
        stats = {
            "total": len(eval_list),
            "by_type": {},
            "by_decision": {},
            "avg_suitability_score": 0.0
        }
        
        total_score = 0
        for eval_record in eval_list:
            q_type = eval_record.get("question_type", "unknown")
            decision = eval_record.get("decision", "unknown")
            score = eval_record.get("suitability_score", 0.0)
            
            stats["by_type"][q_type] = stats["by_type"].get(q_type, 0) + 1
            stats["by_decision"][decision] = stats["by_decision"].get(decision, 0) + 1
            total_score += score
        
        if len(eval_list) > 0:
            stats["avg_suitability_score"] = total_score / len(eval_list)
        
        return stats


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

class ContextQEvaluationPipeline:
    """
    Main pipeline for evaluating questions using ContextQ Suitability Rubric.
    Replaces the old 5-agent quality scoring system.
    """
    
    def __init__(self, storage_file: str = None, feedback_records_file: str = None, dynamic_evaluators: List[Dict[str, Any]] = None):
        """Initialize the suitability evaluation pipeline.
        
        Args:
            storage_file: Path to JSON file for storing evaluations
            feedback_records_file: Path to feedback records (legacy, not used)
            dynamic_evaluators: List of dynamically created evaluators from teacher feedback
                                Format: [{"name": "emotional_resonance", "criteria": "Checks if question evokes emotions"}, ...]
        """
        self.storage = EvaluationManager(storage_file)
        self.dynamic_evaluators = dynamic_evaluators or []
        print(f"[INFO] ContextQEvaluationPipeline.__init__: Received {len(self.dynamic_evaluators)} dynamic evaluators")
        if self.dynamic_evaluators:
            for ev in self.dynamic_evaluators:
                print(f"   Evaluator: {ev}")
        self.suitability_program = SuitabilityEvaluationProgram(dynamic_evaluators=self.dynamic_evaluators)
        
        print("\n" + "="*80)
        print("ContextQ Suitability Evaluation Pipeline Initialized")
        print("="*80)
        print("   - TypeClassificationAgent: Classifies question type")
        print("   - CompletionSuitabilityAgent: Checks rhyming/repeated phrases")
        print("   - RecallSuitabilityAgent: Checks 3 subcriteria")
        print("   - OpenEndedSuitabilityAgent: Checks 4 subcriteria")
        print("   - WhSuitabilityAgent: Checks story details focus")
        print("   - DistancingSuitabilityAgent: Checks 3 subcriteria")
        if self.dynamic_evaluators:
            print(f"   - Dynamic Evaluators ({len(self.dynamic_evaluators)}):")
            for eval in self.dynamic_evaluators:
                print(f"     • {eval.get('name', 'unknown')}: {eval.get('criteria', '')}")
        print("="*80 + "\n")
    
    def evaluate_and_store(self, 
                          questions: List[str], 
                          story_context: str = None, 
                          moral_or_objective: str = None,
                          clear_existing: bool = False) -> List[Dict[str, Any]]:
        """
        Evaluate questions for suitability and store results.
        
        Args:
            questions: List of question texts
            story_context: Story content and context (optional, for logging)
            moral_or_objective: Target moral lesson (optional, for logging)
            clear_existing: Whether to clear existing evaluations
            
        Returns:
            List of evaluation results
        """
        evaluations = []
        
        for i, question in enumerate(questions, 1):
            print(f"Evaluating question {i}/{len(questions)}: {question[:60]}...")
            
            # Run suitability evaluation
            result = self.suitability_program(
                question=question,
                story_context=story_context
            )
            
            # Add metadata
            timestamp = datetime.now().isoformat()
            evaluation_record = {
                "timestamp": timestamp,
                "question": question,
                "question_type": result["question_type"],
                "type_confidence": result["type_confidence"],
                "type_reasoning": result["type_reasoning"],
                "suitability_score": result["suitability_score"],
                "decision": result["decision"],
                "evaluation_reasoning": result["evaluation_reasoning"],
                "story_context": story_context,
                "moral_or_objective": moral_or_objective,
                "details": result.get("details", {}),
                "dynamic_evaluations": result.get("dynamic_evaluations", {})  # Store dynamic evaluator results
            }
            
            evaluations.append(evaluation_record)
        
        # Store results
        self.storage.store_evaluations(evaluations, clear_existing)
        
        # Print summary
        print(f"\n[INFO] Suitability Evaluation Complete:")
        print(f"   - Total questions: {len(evaluations)}")
        print(f"   - Passing: {sum(1 for e in evaluations if e['decision'] == 'pass')}")
        print(f"   - Regenerate: {sum(1 for e in evaluations if e['decision'] == 'regenerate')}")
        
        return evaluations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        return self.storage.get_evaluation_stats()
    
    def generate_and_evaluate_questions(
        self,
        generator_func,
        generator_params: Dict[str, Any],
        num_questions: int = 5,
        story_context: str = None
    ) -> List[Dict[str, Any]]:
        """
        Generate questions with automatic regeneration until they pass suitability.
        
        This method coordinates question generation and evaluation in one place,
        eliminating the need for separate filtering and regeneration loops.
        
        Args:
            generator_func: Callable that generates question candidates
            generator_params: Dictionary of parameters for generator_func
            num_questions: Number of suitable questions to generate
            story_context: Full story context for evaluation
            
        Returns:
            List of suitable questions (only those with decision="pass")
        """
        suitable_questions = []
        total_regenerations = 0
        
        print(f"\n[INFO] Generating {num_questions} suitable questions with regeneration...")
        
        for q_idx in range(num_questions):
            print(f"\n[INFO] Question {q_idx + 1}/{num_questions}")
            attempts = 0
            feedback_history = []
            best_candidate = None
            
            while attempts < 3:  # Max 3 attempts per question
                try:
                    # Generate question candidate
                    if attempts == 0:
                        candidate_params = None  # No feedback on first attempt
                    else:
                        # Add feedback to parameters - pass as single argument to generator function
                        candidate_params = "\n".join(feedback_history) if feedback_history else None
                    
                    # Call generator
                    if candidate_params is None:
                        # First attempt - no feedback
                        candidate_result = generator_func()
                    else:
                        # Subsequent attempts - pass feedback as string argument
                        candidate_result = generator_func(candidate_params)
                    
                    # Handle case where generator returns a list of questions
                    if isinstance(candidate_result, list) and len(candidate_result) > 0:
                        # Generator returned multiple questions - evaluate each one
                        found_passing = False
                        for candidate_q in candidate_result:
                            # Extract question text
                            if isinstance(candidate_q, dict):
                                question_text = candidate_q.get("question", "")
                            else:
                                question_text = str(candidate_q) if candidate_q else ""
                            
                            if question_text:
                                # Evaluate suitability
                                eval_result = self.suitability_program(
                                    question=question_text,
                                    story_context=story_context
                                )
                                
                                # Check decision
                                if eval_result.get("decision") == "pass":
                                    
                                    # Add evaluation metadata
                                    if isinstance(candidate_q, dict):
                                        candidate_q["suitability_evaluation"] = {
                                            "decision": eval_result.get("decision"),
                                            "suitability_score": eval_result.get("suitability_score"),
                                            "question_type": eval_result.get("question_type"),
                                            "evaluation_reasoning": eval_result.get("evaluation_reasoning"),
                                            "type_confidence": eval_result.get("type_confidence"),
                                            "type_reasoning": eval_result.get("type_reasoning")
                                        }
                                        suitable_questions.append(candidate_q)
                                        found_passing = True
                                        break  # Found passing question from batch
                        
                        if found_passing:
                            break  # Success, move to next question
                        else:
                            # None of the batch passed - collect feedback for regeneration
                            attempts += 1
                            # We'll use feedback from the last evaluated question
                            feedback = "All questions in batch failed suitability checks. Generate different types of questions."
                            feedback_history.append(feedback)
                            print(f"[WARN]  Question {q_idx + 1} batch failed suitability (attempt {attempts}/3)")
                            continue
                    
                    # Handle case where generator returns single question
                    # Extract question text
                    if isinstance(candidate_result, dict):
                        question_text = candidate_result.get("question", "")
                    elif isinstance(candidate_result, str):
                        question_text = candidate_result
                    else:
                        question_text = str(candidate_result)
                    
                    if question_text:
                        # Evaluate suitability
                        eval_result = self.suitability_program(
                            question=question_text,
                            story_context=story_context
                        )
                        
                        # Check decision
                        if eval_result.get("decision") == "pass":
                            
                            # Add evaluation metadata to candidate
                            if isinstance(candidate_result, dict):
                                candidate_result["suitability_evaluation"] = {
                                    "decision": eval_result.get("decision"),
                                    "suitability_score": eval_result.get("suitability_score"),
                                    "question_type": eval_result.get("question_type"),
                                    "evaluation_reasoning": eval_result.get("evaluation_reasoning"),
                                    "type_confidence": eval_result.get("type_confidence"),
                                    "type_reasoning": eval_result.get("type_reasoning")
                                }
                                suitable_questions.append(candidate_result)
                            else:
                                # If candidate is just a string, wrap it
                                suitable_questions.append({
                                    "question": question_text,
                                    "suitability_evaluation": {
                                        "decision": eval_result.get("decision"),
                                        "suitability_score": eval_result.get("suitability_score"),
                                        "question_type": eval_result.get("question_type"),
                                        "evaluation_reasoning": eval_result.get("evaluation_reasoning"),
                                        "type_confidence": eval_result.get("type_confidence"),
                                        "type_reasoning": eval_result.get("type_reasoning")
                                    }
                                })
                            
                            # Store evaluation
                            self._store_evaluation_record(question_text, eval_result, story_context)
                            break  # Success!
                        else:
                            # Failed - collect feedback
                            attempts += 1
                            feedback = f"Failed suitability: {eval_result.get('evaluation_reasoning')} (score: {eval_result.get('suitability_score')})"
                            feedback_history.append(feedback)
                            print(f"[WARN]  Attempt {attempts}/3: {feedback}")
                            best_candidate = candidate_result  # Keep track of best attempt
                    else:
                        attempts += 1
                        print(f"[WARN]  Generated empty question (attempt {attempts}/3)")
                
                except Exception as e:
                    attempts += 1
                    print(f"[WARN]  Error generating question (attempt {attempts}/3): {e}")
                    import traceback
                    traceback.print_exc()
            
            # If all attempts failed, add best candidate anyway (marked as potentially unsuitable)
            if attempts >= 3 and best_candidate:
                print(f"[WARN]  Question {q_idx + 1} failed after 3 attempts, including best candidate")
                if isinstance(best_candidate, dict):
                    best_candidate["suitability_evaluation"] = {
                        "decision": "failed_after_max_attempts",
                        "warning": "Question did not pass suitability checks after 3 attempts"
                    }
                    suitable_questions.append(best_candidate)
            
            total_regenerations += attempts
        
        print(f"\n[INFO] Generated {len(suitable_questions)}/{num_questions} suitable questions after {total_regenerations} regeneration attempts")
        
        return suitable_questions
    
    def _store_evaluation_record(self, question: str, eval_result: Dict[str, Any], story_context: str, 
                                 storybook_id: str = None, objective: str = None, set_number: str = None):
        """Store a single evaluation record."""
        try:
            timestamp = datetime.now().isoformat()
            
            # Extract serializable data from details (avoid DSPy Prediction objects)
            details = eval_result.get("details", {})
            serializable_details = {}
            if isinstance(details, dict):
                for key, value in details.items():
                    if isinstance(value, (str, int, float, bool, type(None))):
                        serializable_details[key] = value
                    elif isinstance(value, dict):
                        # Recursively extract serializable nested dicts
                        serializable_nested = {}
                        for k, v in value.items():
                            if isinstance(v, (str, int, float, bool, type(None))):
                                serializable_nested[k] = v
                        serializable_details[key] = serializable_nested
            
            # Get relevant evaluator scores based on question type
            question_type = eval_result.get("question_type", "")
            
            # Map question type to relevant suitability evaluator
            type_to_evaluator = {
                "Completion": "completion_suitability",
                "Recall": "recall_suitability",
                "Open-Ended": "open_ended_suitability",
                "Wh": "wh_suitability",
                "Distancing": "distancing_suitability"
            }
            
            # Get relevant evaluator name
            relevant_evaluator = type_to_evaluator.get(question_type, "")
            
            # Get dynamic evaluator scores if they exist
            dynamic_eval_scores = eval_result.get("dynamic_evaluations", {})
            
            evaluation_record = {
                "timestamp": timestamp,
                "question": question,
                "question_type": question_type,
                "type_confidence": eval_result.get("type_confidence"),
                "type_reasoning": eval_result.get("type_reasoning"),
                "suitability_score": eval_result.get("suitability_score"),
                "decision": eval_result.get("decision"),
                "evaluation_reasoning": eval_result.get("evaluation_reasoning"),
                # Do not store full story text to avoid excessive output/noise
                # Keep concise identifiers instead
                "storybook_id": storybook_id,
                "objective": objective,
                "details": serializable_details,
                "regenerated": eval_result.get("regenerated", False),
                "regeneration_history": eval_result.get("regeneration_history", []),
                "relevant_evaluator": relevant_evaluator,  # Only the evaluator for this question type
                "dynamic_evaluations": dynamic_eval_scores  # Dynamic evaluators apply to all
            }
            
            # Note: Metadata (storybook_id, objective, set_number) is now stored in set_metadata
            # to avoid redundancy per question
            
            # Add to storage with set-based structure
            if "evaluations" not in self.storage.evaluations:
                self.storage.evaluations["evaluations"] = []
            
            # Check if we need to add set metadata (if this is the first question in a set)
            evaluations_list = self.storage.evaluations["evaluations"]
            
            # Check if there's already a set metadata entry for this set_number
            set_exists = any(
                item.get("set_metadata") and 
                item.get("set_metadata", {}).get("set_number") == set_number and 
                item.get("set_metadata", {}).get("objective") == objective and
                item.get("set_metadata", {}).get("storybook_id") == storybook_id
                for item in evaluations_list
            )
            
            # If this is the first question in a new set, add set metadata first
            if not set_exists and (storybook_id and objective and set_number):
                set_metadata = {
                    "set_metadata": {
                        "storybook_id": storybook_id,
                        "objective": objective,
                        "set_number": set_number,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                evaluations_list.append(set_metadata)
            
            # Then add the question record (without redundant metadata)
            evaluations_list.append(evaluation_record)
            self.storage._save_evaluations()
        except Exception as e:
            print(f"Warning: Could not store evaluation: {e}")


# ============================================================================
# COMMAND LINE INTERFACE (for testing)
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ContextQ Suitability Evaluation Pipeline")
    parser.add_argument("--questions", nargs="+", help="Questions to evaluate")
    parser.add_argument("--story", type=str, help="Story context")
    parser.add_argument("--objective", type=str, help="Learning objective")
    parser.add_argument("--clear", action="store_true", help="Clear existing evaluations")
    
    args = parser.parse_args()
    
    if args.questions:
        pipeline = ContextQEvaluationPipeline()
        evaluations = pipeline.evaluate_and_store(
            questions=args.questions,
            story_context=args.story,
            moral_or_objective=args.objective,
            clear_existing=args.clear
        )
        
        print("\nEvaluation Results:")
        for eval_record in evaluations:
            print(f"\n  Question: {eval_record['question']}")
            print(f"  Type: {eval_record['question_type']}")
            print(f"  Suitability Score: {eval_record['suitability_score']}")
            print(f"  Decision: {eval_record['decision']}")
            print(f"  Reasoning: {eval_record['evaluation_reasoning']}")
    else:
        print("ContextQ Suitability Evaluation Pipeline")
        print("Use --questions to specify questions to evaluate")
