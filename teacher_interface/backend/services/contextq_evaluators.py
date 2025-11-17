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
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path for imports
# File is at: MoralQ/teacher_interface/backend/services/contextq_evaluators.py
# Need to add: MoralQ/ to path
_file_dir = os.path.dirname(os.path.abspath(__file__))  # services/
_backend_dir = os.path.dirname(_file_dir)  # backend/
_teacher_interface_dir = os.path.dirname(_backend_dir)  # teacher_interface/
_project_root = os.path.dirname(_teacher_interface_dir)  # MoralQ/
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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
from teacher_interface.backend.evaluators.manager import EvaluatorManager


# ============================================================================
# DYNAMIC EVALUATOR SIGNATURE (Type-Agnostic)
# ============================================================================

# LEGACY: UniversalEvaluatorSignature was used before dynamic_template-based evaluators.
# Dynamic evaluators are now first-class via the EvaluatorManager registry.


# ============================================================================
# SUITABILITY EVALUATION PROGRAM
# ============================================================================

class SuitabilityEvaluationProgram(dspy.Module):
    """Evaluates questions using central EvaluatorManager. Keeps type labeling via DSPy."""
    
    def __init__(self, dynamic_evaluators: List[Dict[str, Any]] = None):
        super().__init__()
        self.type_agent = dspy.Predict(TypeClassificationSignature)
        # Centralized evaluator manager (weights + dynamic evaluators)
        self.evaluator_manager = EvaluatorManager()
    
    def forward(self, question: str, story_context: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate question with centralized evaluators and label its type.
        
        OPTIMIZED: Only runs the relevant core evaluator (based on type) + dynamic evaluators,
        instead of running all 5 core evaluators. This reduces LLM calls from ~9 to ~4 per question.
        """
        
        # Type classification for labeling (not gating)
        type_result = self.type_agent(question=question)
        question_type = type_result.question_type.replace("_", "-") if hasattr(type_result, "question_type") else "Unknown"
        
        # Map question type to core evaluator name - only run the relevant one
        type_to_core = {
            "Completion": "completion_suitability",
            "Recall": "recall_suitability",
            "Open-Ended": "open_ended_suitability",
            "Wh": "wh_suitability",
            "Distancing": "distancing_suitability",
        }
        core_eval_name = type_to_core.get(question_type, None)
        
        # Only evaluate the relevant core evaluator (not all 5)
        core = {}
        if core_eval_name and core_eval_name in self.evaluator_manager.evaluators:
            try:
                core = self.evaluator_manager.evaluators[core_eval_name].evaluate(
                    question=question,
                    context={"story_context": story_context or ""}
                )
            except Exception as e:
                print(f"[WARN] Error evaluating {core_eval_name}: {e}")
                core = {"score": 0.0, "decision": "regenerate", "reasoning": f"Error: {e}"}
        
        # Evaluate only dynamic evaluators (not all evaluators)
        # Identify dynamic evaluators by checking if they're not in the core set
        dynamic_evaluations = {}
        core_names = {
            "completion_suitability",
            "recall_suitability",
            "open_ended_suitability",
            "wh_suitability",
            "distancing_suitability",
        }
        for name, evaluator in self.evaluator_manager.evaluators.items():
            if name not in core_names:
                try:
                    dyn_result = evaluator.evaluate(
                        question=question,
                        context={"story_context": story_context or ""}
                    )
                    dynamic_evaluations[name] = dyn_result
                    # Early exit: if any dynamic evaluator fails, we know the decision will be "regenerate"
                    # Continue evaluating others to collect all failure reasons, but we can skip if we want
                except Exception as e:
                    print(f"[WARN] Error evaluating dynamic evaluator {name}: {e}")
                    dynamic_evaluations[name] = {"score": 0.0, "decision": "regenerate", "reasoning": f"Error: {e}"}

        # Fallbacks if core missing
        score = core.get("score", 0.0) if isinstance(core, dict) else 0.0
        decision = core.get("decision", "regenerate") if isinstance(core, dict) else "regenerate"
        reasoning = core.get("reasoning", "") if isinstance(core, dict) else ""

        # If any dynamic evaluator fails, force regeneration and append reasons
        for dyn_name, dyn in dynamic_evaluations.items():
            try:
                if isinstance(dyn, dict) and dyn.get("decision") == "regenerate":
                    decision = "regenerate"
                    if dyn.get("reasoning"):
                        reasoning += f"\n[{dyn_name}]: {dyn.get('reasoning')}"
            except Exception:
                continue

        return {
            "question": question,
            "question_type": question_type,
            "type_confidence": getattr(type_result, "confidence", ""),
            "type_reasoning": getattr(type_result, "reasoning", ""),
            "suitability_score": score,
            "decision": decision,
            "evaluation_reasoning": reasoning,
            "dynamic_evaluations": dynamic_evaluations,
            "details": {},
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
        """Load existing evaluations from storage file.
        
        Resilient loading: Skips invalid entries instead of failing completely.
        Preserves set_metadata entries and valid question evaluations.
        """
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                    
                # Handle both dict and list formats
                if isinstance(raw, dict):
                    evaluations_list = raw.get("evaluations", [])
                else:
                    evaluations_list = raw if isinstance(raw, list) else []
                
                # Filter and validate entries - preserve set_metadata, regeneration_summary, and valid questions
                validated_list = []
                for item in evaluations_list:
                    # Preserve set_metadata entries (they don't have "question" field)
                    if isinstance(item, dict) and "set_metadata" in item:
                        validated_list.append(item)
                    # Preserve regeneration_summary entries
                    elif isinstance(item, dict) and "regeneration_summary" in item:
                        validated_list.append(item)
                    # Validate question evaluation entries
                    elif isinstance(item, dict) and "question" in item:
                        try:
                            validated = EvaluationResultModel.parse_obj(item)
                            validated_list.append(validated.dict(exclude_none=True))
                        except Exception as e:
                            print(f"[WARN] Skipping invalid evaluation entry: {e}")
                            # Still preserve it as-is if it's close to valid
                            validated_list.append(item)
                    else:
                        # Try to preserve other entries
                        validated_list.append(item)
                
                return {"evaluations": validated_list}
            except Exception as e:
                print(f"Error loading evaluations: {e}")
                import traceback
                traceback.print_exc()
                # Return empty structure instead of empty dict to preserve structure
                return {"evaluations": []}
        return {"evaluations": []}
    
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
        """Store evaluation results.
        
        Preserves existing evaluations unless clear_existing=True.
        Reloads file before saving to ensure we don't lose data from other processes.
        """
        # Reload existing evaluations to ensure we have the latest data
        # (important if multiple processes or rapid successive calls)
        existing = self._load_evaluations()
        
        if clear_existing:
            self.evaluations = {"evaluations": []}
        else:
            # Preserve existing evaluations
            self.evaluations = existing
        
        if "evaluations" not in self.evaluations:
            self.evaluations["evaluations"] = []
        
        # Validate and add new evaluations
        validated = []
        for record in evaluations:
            try:
                # Only validate entries that have a "question" field
                if isinstance(record, dict) and "question" in record:
                    validated_record = EvaluationResultModel.parse_obj(record)
                    validated.append(validated_record.dict(exclude_none=True))
                else:
                    # Preserve non-question entries (like set_metadata, regeneration_summary) as-is
                    validated.append(record)
            except Exception as e:
                print(f"[WARN] Skipping invalid evaluation record: {e}")
                # Still try to preserve it
                validated.append(record)
        
        self.evaluations["evaluations"].extend(validated)
        self._save_evaluations()
    
    def store_regeneration_summary(self, storybook_id: str, objective: str, set_number: str, summary: Dict[str, Any]):
        """Store regeneration summary for a set. Called from server.py after all questions are stored."""
        try:
            # Reload to get latest data
            existing = self._load_evaluations()
            self.evaluations = existing
            
            if "evaluations" not in self.evaluations:
                self.evaluations["evaluations"] = []
            
            evaluations_list = self.evaluations["evaluations"]
            
            # Find the set_metadata entry for this set
            set_metadata_idx = None
            for idx, item in enumerate(evaluations_list):
                if isinstance(item, dict) and "set_metadata" in item:
                    meta = item.get("set_metadata", {})
                    if (meta.get("storybook_id") == storybook_id and 
                        meta.get("objective") == objective and 
                        meta.get("set_number") == set_number):
                        set_metadata_idx = idx
                        break
            
            # If set_metadata exists, add summary right after it
            if set_metadata_idx is not None:
                # Check if summary already exists (avoid duplicates)
                next_idx = set_metadata_idx + 1
                if (next_idx < len(evaluations_list) and 
                    isinstance(evaluations_list[next_idx], dict) and 
                    "regeneration_summary" in evaluations_list[next_idx]):
                    # Update existing summary
                    evaluations_list[next_idx]["regeneration_summary"] = summary
                else:
                    # Insert new summary after set_metadata
                    summary_entry = {"regeneration_summary": summary}
                    evaluations_list.insert(next_idx, summary_entry)
            else:
                # If no set_metadata found, just append (shouldn't happen, but handle gracefully)
                summary_entry = {"regeneration_summary": summary}
                evaluations_list.append(summary_entry)
            
            self._save_evaluations()
            print(f"[INFO] Stored regeneration summary for {storybook_id} - {set_number}")
        except Exception as e:
            print(f"[WARN] Error storing regeneration summary: {e}")
            import traceback
            traceback.print_exc()
    
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
        evaluation_records: List[Dict[str, Any]] = []
        total_regenerations = 0
        # Capture full regeneration traces across all questions
        overall_regeneration_log = []
        
        print(f"\n[INFO] Generating {num_questions} suitable questions with regeneration...")
        
        for q_idx in range(num_questions):
            print(f"\n[INFO] Question {q_idx + 1}/{num_questions}")
            attempts = 0
            feedback_history = []
            best_candidate = None
            per_question_regen_history = []
            
            while attempts < 5:  # Max 5 attempts per question
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
                                
                                # Log this attempt
                                per_question_regen_history.append({
                                    "attempt": attempts + 1,
                                    "candidate_question": question_text,
                                    "avoidance_instructions": candidate_params or "",
                                    "decision": eval_result.get("decision"),
                                    "suitability_score": eval_result.get("suitability_score"),
                                })
                                
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
                                            # intentionally omit type_reasoning from storage record
                                        }
                                        # Persist regeneration details on pass
                                        candidate_q["suitability_evaluation"]["regenerated"] = attempts > 0
                                        candidate_q["suitability_evaluation"]["regeneration_history"] = per_question_regen_history
                                        suitable_questions.append(candidate_q)
                                        # Build evaluation record for storage
                                        evaluation_records.append({
                                            "timestamp": datetime.now().isoformat(),
                                            "question": question_text,
                                            "question_type": eval_result.get("question_type"),
                                            "type_confidence": eval_result.get("type_confidence"),
                                            "suitability_score": eval_result.get("suitability_score"),
                                            "decision": eval_result.get("decision"),
                                            "evaluation_reasoning": eval_result.get("evaluation_reasoning"),
                                            "story_context": story_context,
                                            "moral_or_objective": generator_params.get("objective") if isinstance(generator_params, dict) else None,
                                            "details": {},
                                            "dynamic_evaluations": eval_result.get("dynamic_evaluations", {}),
                                            "regenerated": attempts > 0,
                                            "regeneration_history": per_question_regen_history,
                                        })
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
                            per_question_regen_history.append({
                                "attempt": attempts,
                                "avoidance_instructions": feedback,
                                "note": "Batch failed; requesting diverse alternatives"
                            })
                            print(f"[WARN]  Question {q_idx + 1} batch failed suitability (attempt {attempts}/5)")
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
                        
                        # Log this attempt
                        per_question_regen_history.append({
                            "attempt": attempts + 1,
                            "candidate_question": question_text,
                            "avoidance_instructions": candidate_params or "",
                            "decision": eval_result.get("decision"),
                            "suitability_score": eval_result.get("suitability_score"),
                        })
                        
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
                                    # intentionally omit type_reasoning from storage record
                                }
                                candidate_result["suitability_evaluation"]["regenerated"] = attempts > 0
                                candidate_result["suitability_evaluation"]["regeneration_history"] = per_question_regen_history
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
                                        # intentionally omit type_reasoning from storage record
                                    }
                                })
                            
                            # Store evaluation (build record for storage)
                            # Attach regeneration info for logging
                            eval_result["regenerated"] = attempts > 0
                            eval_result["regeneration_history"] = per_question_regen_history
                            evaluation_records.append({
                                "timestamp": datetime.now().isoformat(),
                                "question": question_text,
                                "question_type": eval_result.get("question_type"),
                                "type_confidence": eval_result.get("type_confidence"),
                                "suitability_score": eval_result.get("suitability_score"),
                                "decision": eval_result.get("decision"),
                                "evaluation_reasoning": eval_result.get("evaluation_reasoning"),
                                "story_context": story_context,
                                "moral_or_objective": generator_params.get("objective") if isinstance(generator_params, dict) else None,
                                "details": {},
                                "dynamic_evaluations": eval_result.get("dynamic_evaluations", {}),
                                "regenerated": eval_result.get("regenerated", False),
                                "regeneration_history": eval_result.get("regeneration_history", []),
                            })
                            break  # Success!
                        else:
                            # Failed - collect feedback
                            attempts += 1
                            feedback = f"Failed suitability: {eval_result.get('evaluation_reasoning')} (score: {eval_result.get('suitability_score')})"
                            feedback_history.append(feedback)
                            per_question_regen_history.append({
                                "attempt": attempts,
                                "avoidance_instructions": feedback,
                                "note": "Failed; refining with avoidance instructions"
                            })
                            print(f"[WARN]  Attempt {attempts}/3: {feedback}")
                            best_candidate = candidate_result  # Keep track of best attempt
                    else:
                        attempts += 1
                        print(f"[WARN]  Generated empty question (attempt {attempts}/5)")
                
                except Exception as e:
                    attempts += 1
                    print(f"[WARN]  Error generating question (attempt {attempts}/5): {e}")
                    import traceback
                    traceback.print_exc()
            
            # If all attempts failed, add best candidate anyway (marked as potentially unsuitable)
            if attempts >= 5 and best_candidate:
                print(f"[WARN]  Question {q_idx + 1} failed after 5 attempts, including best candidate")
                if isinstance(best_candidate, dict):
                    best_candidate["suitability_evaluation"] = {
                        "decision": "failed_after_max_attempts",
                        "warning": "Question did not pass suitability checks after 5 attempts"
                    }
                    # Include regeneration trace even on failure
                    best_candidate["suitability_evaluation"]["regenerated"] = True
                    best_candidate["suitability_evaluation"]["regeneration_history"] = per_question_regen_history
                    suitable_questions.append(best_candidate)
                    # Build failed evaluation record for storage
                    evaluation_records.append({
                        "timestamp": datetime.now().isoformat(),
                        "question": best_candidate.get("question") if isinstance(best_candidate, dict) else "",
                        "question_type": "",  # unknown on failure
                        "type_confidence": "",
                        "suitability_score": 0.0,
                        "decision": "failed_after_max_attempts",
                        "evaluation_reasoning": "Question did not pass suitability checks after 3 attempts",
                        "story_context": story_context,
                        "moral_or_objective": generator_params.get("objective") if isinstance(generator_params, dict) else None,
                        "details": {},
                        "dynamic_evaluations": {},
                        "regenerated": True,
                        "regeneration_history": per_question_regen_history,
                    })
            
            total_regenerations += attempts
            overall_regeneration_log.append({
                "question_index": q_idx + 1,
                "attempts": attempts,
                "regeneration_history": per_question_regen_history
            })
        
        print(f"\n[INFO] Generated {len(suitable_questions)}/{num_questions} suitable questions after {total_regenerations} regeneration attempts")
        
        # Persist evaluation records for this generation batch
        if evaluation_records:
            try:
                self.storage.store_evaluations(evaluation_records, clear_existing=False)
            except Exception as e:
                print(f"[WARN] Could not persist regeneration evaluations: {e}")
        
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
                "batch_number": eval_result.get("batch_number"),  # Which batch this question came from (1 = initial, 2+ = regenerated)
                "regeneration_history": eval_result.get("regeneration_history", []),  # Use actual regeneration history from eval_result
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
