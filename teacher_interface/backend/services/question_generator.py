"""
Main Question Generation System for MoralQ Teacher Interface
============================================================

This is the central orchestration module that coordinates:
1. Moral extraction from storybooks
2. Story segmentation
3. Question generation based on learning objectives
4. Feedback collection and optimization

All DSPy signatures and optimization logic are contained here for easy experimentation.
"""

import dspy
import json
import os
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports (if not already added)
# File is at: MoralQ/teacher_interface/backend/services/question_generator.py
# Need to add: MoralQ/ to path
import sys
_file_dir = os.path.dirname(os.path.abspath(__file__))  # services/
_backend_dir = os.path.dirname(_file_dir)  # backend/
_teacher_interface_dir = os.path.dirname(_backend_dir)  # teacher_interface/
_project_root = os.path.dirname(_teacher_interface_dir)  # MoralQ/
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from teacher_interface.backend.models import EvaluationLogModel
from teacher_interface.backend.evaluators import EvaluatorManager

# Configure DSPy with OpenAI
dspy.configure(lm=dspy.LM("openai/gpt-4.1-2025-04-14"))


# ============================================================================
# DSPy SIGNATURES - All structured output definitions in one place
# ============================================================================

class MoralExtractor(dspy.Signature):
    """Extract moral lessons and segment stories for children aged 4-6."""
    
    story = dspy.InputField(desc="Children's storybook content with page markers")
    
    moral = dspy.OutputField(desc="""The main lessons or takeaways from the character's experience in the story. 
    The lessons need to be age appropriate for a child aged 4 to 6, and simplified to their language levels. 
    Generate a clear, concise moral that teaches a valuable life lesson.""")
    
    segments = dspy.OutputField(desc="""Return a JSON array of story segments. Each segment is a sub-sequence 
    of the storybook that conveys a same event, shares the same linguistic style, or falls into the same time frame. 
    Each segment must be a JSON object with exactly these fields:
    {
        "START": <page_number>,
        "END": <page_number>,
        "SUMMARY": "<brief summary of main development during this segment>",
        "REASONING": "<explanation of why this subset constitutes a good segment>"
    }
    
    The combined pages from all segments must cover the entire storybook without skipping or overlapping pages. 
    From one segment to the next, there must be a clear transition of development in event, time, or style. 
    Ensure the output is valid JSON with proper quotes and structure.""")


class QuestionGenerationSignature(dspy.Signature):
    """Generate educational questions aligned with a teacher-defined learning objective."""
    
    story = dspy.InputField(desc="Full story content, marked with page numbers")
    segments = dspy.InputField(desc="Summarized segments of the story with page ranges and reasoning")
    objective = dspy.InputField(desc="Teacher-specified learning objective (e.g., empathy, critical thinking, moral reasoning, comprehension)")
    story_title = dspy.InputField(desc="Title of the story")
    variation_prompt = dspy.InputField(desc="Additional context to ensure variety in question generation")
    moral_text = dspy.InputField(desc="Optional moral lesson extracted from the story for additional context", prefix="Moral Context:")
    avoidance_instructions = dspy.InputField(desc="Feedback from previous failed question attempts - avoid these issues when generating", default="")
    
    questions = dspy.OutputField(desc="""A JSON array of 5-10 diverse, age-appropriate questions with these fields:
    {
        "question": "Clear, engaging question appropriate for ages 4-6",
        "type": "comprehension/reflection/application/extension",
        "difficulty": "easy/medium/hard",
        "explanation": "Why this question supports the learning objective",
        "page_number": integer (page where this question should be discussed)
    }
    
    Ensure questions:
    - Use simple, age-appropriate vocabulary for 4-6 year olds
    - Are diverse in type and difficulty
    - Align clearly with the specified objective
    - Reference specific story events
    - Encourage meaningful discussion""")
    
    learning_objectives = dspy.OutputField(desc="A list of 3-4 specific learning objectives this question set addresses")


# ============================================================================
# SUB-MODULES - Specialized components for different tasks
# ============================================================================

class MoralGenerator:
    """
    Sub-module for moral extraction and story segmentation.
    Uses BestOfN optimization to generate multiple high-quality candidates.
    """
    
    def __init__(self):
        """Initialize moral generator with BestOfN optimization."""
        self.moral_extractor = dspy.Predict(MoralExtractor)
        
        # Wrap with BestOfN for multiple candidate generation
        self.best_of_n = dspy.BestOfN(
            module=self.moral_extractor,
            N=3,  # Generate 3 candidates
            reward_fn=self._evaluate_moral_quality,
            threshold=0.5
        )
        
    def _evaluate_moral_quality(self, example, prediction, trace=None, pred_name=None, pred_trace=None):
        """
        Evaluate moral quality using LLM-based critique.
        This is our reward function for BestOfN optimization.
        """
        try:
            moral = prediction.moral
            segments = prediction.segments
            
            # Format segments for evaluation
            segments_text = json.dumps(segments, indent=2) if isinstance(segments, list) else str(segments)
            
            # LLM-based quality evaluation
            critique_prompt = f"""You are an expert evaluator of educational content for children aged 4-6.

Moral: {moral}
Segments: {segments_text}

Evaluate this moral lesson on:
1. Age-appropriateness (4-6 years): Is the language simple and clear?
2. Educational value: Does it teach a meaningful life lesson?
3. Story relevance: Does it reflect the main lesson from the story?
4. Clarity: Is it easy to understand and remember?
5. Segment quality: Are the story segments well-structured?

Respond with ONLY a single number between 0.0 and 1.0 representing overall quality."""

            # Get LLM critique
            critique_result = dspy.LM("openai/gpt-4.1-2025-04-14")(critique_prompt)
            
            # Parse score
            if isinstance(critique_result, list):
                score_text = str(critique_result[0]) if critique_result else "0.5"
            else:
                score_text = str(critique_result).strip() if critique_result else "0.5"
            
            # Extract numeric score
            import re
            numbers = re.findall(r"0?\.\d+", score_text)
            quality_score = float(numbers[0]) if numbers else 0.5
            
            print(f"  → Quality score: {quality_score:.2f}")
            return quality_score
            
        except Exception as e:
            print(f"  → Error in moral evaluation: {e}")
            return 0.5  # Default score on error
    
    def generate(self, story: str, story_title: str = "Unknown") -> Dict[str, Any]:
        """
        Generate multiple moral candidates with quality scores.
        
        Returns:
            {
                "success": bool,
                "candidates": List[{moral, segments, quality_score, ...}],
                "optimization_applied": bool
            }
        """
        print(f"\n[INFO] Generating moral candidates for: {story_title}")
        print("=" * 80)
        
        candidates = []
        
        try:
            # Use BestOfN to generate first candidate
            print("Generating Candidate 1 (BestOfN)...")
            best_prediction = self.best_of_n(story=story)
            
            # Parse the best candidate
            moral_text = best_prediction.moral
            segments_raw = best_prediction.segments
            
            # Parse segments if string
            if isinstance(segments_raw, str):
                try:
                    segments_data = json.loads(segments_raw)
                except:
                    segments_data = []
            else:
                segments_data = segments_raw
            
            # Get quality score for best candidate
            quality_score = self._evaluate_moral_quality(None, best_prediction)
            
            candidates.append({
                "candidate_id": 1,
                "moral": moral_text,
                "segments": segments_data,
                "quality_score": quality_score,
                "generation_method": "BestOfN"
            })
            
            # Generate additional candidates for variety
            variation_prompts = [
                "Focus on different aspects of the character's journey.",
                "Emphasize the relationships and interactions between characters.",
                "Highlight the problem-solving and decision-making in the story."
            ]
            
            for i, variation in enumerate(variation_prompts[:2], start=2):
                print(f"Generating Candidate {i} (Regular with variation)...")
                
                # Generate with variation
                result = self.moral_extractor(story=f"{story}\n\nVariation focus: {variation}")
                
                # Parse segments
                segments_raw = result.segments
                if isinstance(segments_raw, str):
                    try:
                        segments_data = json.loads(segments_raw)
                    except:
                        segments_data = []
                else:
                    segments_data = segments_raw
                
                # Evaluate quality
                quality_score = self._evaluate_moral_quality(None, result)
                
                candidates.append({
                    "candidate_id": i,
                    "moral": result.moral,
                    "segments": segments_data,
                    "quality_score": quality_score,
                    "generation_method": "Regular"
                })
            
            # Sort by quality score (highest first)
            candidates.sort(key=lambda x: x["quality_score"], reverse=True)
            
            print("\n[INFO] Moral generation complete!")
            print(f"Generated {len(candidates)} candidates")
            print("=" * 80)
            
            return {
                "success": True,
                "story_title": story_title,
                "candidates": candidates,
                "optimization_applied": True
            }
            
        except Exception as e:
            print(f"\n[ERROR] Error generating morals: {e}")
            return {
                "success": False,
                "error": str(e),
                "candidates": [],
                "story_title": story_title
            }


class SuitabilityProgram(dspy.Module):
    """Aggregate evaluator scores using current weights for DSPy optimization."""

    def __init__(self, evaluator_manager: EvaluatorManager):
        super().__init__()
        self.evaluator_manager = evaluator_manager
        self.weight_params: Dict[str, Any] = {}  # Can be dspy.Parameter or float
        self.refresh_parameters()

    def forward(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = context or {}
        raw_scores = self.evaluator_manager.evaluate_all(question, context)

        weighted_sum = 0.0
        weight_total = 0.0
        for name, result in raw_scores.items():
            score = result.get("score")
            if score is None:
                continue
            param = self.weight_params.get(name)
            if param is not None:
                # Handle both dspy.Parameter and direct float values
                if isinstance(param, (int, float)):
                    weight = float(param)
                else:
                    try:
                        weight = float(param())
                    except (TypeError, AttributeError):
                        weight = float(getattr(param, "value", self.evaluator_manager.weights.get(name, 0.0)))
            else:
                weight = self.evaluator_manager.weights.get(name, 0.0)
            weighted_sum += score * weight
            weight_total += weight

        weighted_score = weighted_sum / weight_total if weight_total > 0 else 0.0
        return {
            "raw_scores": raw_scores,
            "weighted_score": weighted_score,
            "weight_total": weight_total,
        }

    def refresh_parameters(self) -> None:
        # Use dspy.Parameter if available (for optimizer tuning), otherwise use weights directly
        if hasattr(dspy, 'Parameter'):
            self.weight_params = {
                name: dspy.Parameter(init=weight)
                for name, weight in self.evaluator_manager.weights.items()
            }
        else:
            # Fallback: store weights directly without Parameter wrapper
            self.weight_params = {
                name: weight
                for name, weight in self.evaluator_manager.weights.items()
            }


class QuestionGeneratorModule:
    """
    Sub-module for generating educational questions.
    This is the core of the system, with feedback collection and optimization.
    """
    
    def __init__(self, feedback_file: str = "feedback_dataset.json", optimize_with_feedback: bool = False):
        """Initialize question generator with feedback collection and optional optimization.
        
        Args:
            feedback_file: Path to feedback dataset file
            optimize_with_feedback: If True, initialize DSPy optimizer with teacher feedback.
                                   Should only be True when regenerating after feedback collection.
                                   Default False for initial generation.
        """
        self.question_generator = dspy.Predict(QuestionGenerationSignature)
        self.feedback_collector = FeedbackCollector(feedback_file)
        self.optimize_with_feedback = optimize_with_feedback
        self.optimizer = None
        self.optimized = False

        self.evaluator_manager = EvaluatorManager()
        self.suitability_program = SuitabilityProgram(self.evaluator_manager)
        
        # Initialize optimizer ONLY if explicitly requested (after teacher feedback)
        # Initial generation should NOT optimize - just generate and do simple regeneration
        if optimize_with_feedback:
            self._check_and_initialize_optimizer()
        
        # Variation prompts to ensure diversity
        self.variation_prompts = [
            "Encourage different thinking styles and perspectives.",
            "Include questions that spark imagination and creativity.",
            "Mix literal comprehension with deeper reflection.",
            "Balance easy recall with challenging application.",
            "Create opportunities for personal connection to the story."
        ]
    
    def _check_and_initialize_optimizer(self):
        """Check if we have enough feedback to optimize and initialize if so."""
        try:
            # Load feedback records (using absolute path from this file's location)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            storage_dir = os.path.join(current_dir, "..", "storage")
            os.makedirs(storage_dir, exist_ok=True)
            feedback_file = os.path.join(storage_dir, "teacher_feedback_records.json")
            question_eval_file = os.path.join(storage_dir, "question_evaluations.json")

            # CRITICAL: Refresh evaluator manager to load latest weights and dynamic evaluators
            # This ensures optimizer uses manager-adjusted weights (from orchestrator feedback)
            print("[INFO] Refreshing evaluator manager to load latest weights and dynamic evaluators...")
            self.evaluator_manager.refresh()
            print(f"[INFO] Current evaluator weights: {self.evaluator_manager.weights}")
            self.suitability_program.refresh_parameters()
            
            if not os.path.exists(feedback_file) or not os.path.exists(question_eval_file):
                print("[INFO] No feedback files found yet. Will generate without optimization.")
                return
            
            # Load feedback data
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
            
            with open(question_eval_file, 'r') as f:
                raw_eval = json.load(f)
                if isinstance(raw_eval, dict):
                    eval_data = EvaluationLogModel.parse_obj(raw_eval).model_dump()
                else:
                    eval_data = EvaluationLogModel.parse_obj({"evaluations": raw_eval}).model_dump()
            
            # Count feedback records (check all school/teacher combinations)
            records = []
            for school_key in feedback_data:
                if school_key == "records":
                    records.extend(feedback_data.get("records", []))
                elif isinstance(feedback_data[school_key], dict):
                    for teacher_key in feedback_data[school_key]:
                        if isinstance(feedback_data[school_key][teacher_key], list):
                            records.extend(feedback_data[school_key][teacher_key])
            
            num_records = len(records)
            
            # Check if we need to re-compile the optimizer
            # Re-compile if: not optimized yet OR new feedback records exist
            if num_records >= 1:
                if not self.optimized:
                    print(f"[INFO] Found {num_records} feedback records. Initializing DSPy optimizer...")
                    self._initialize_optimizer(feedback_data, eval_data)
                else:
                    # Already optimized - check if there's new feedback
                    # For now, re-compile every time (can be optimized to track record count)
                    print(f"[INFO] Re-compiling optimizer with {num_records} feedback records...")
                    self._initialize_optimizer(feedback_data, eval_data)
            else:
                print("[INFO] No feedback records yet. Will generate without optimization.")
                
        except Exception as e:
            print(f"[WARN]  Error checking for optimizer initialization: {e}")
            import traceback
            traceback.print_exc()
    
    def _initialize_optimizer(self, feedback_data: dict, eval_data: dict):
        """Initialize and compile the DSPy optimizer using SuitabilityProgram.
        
        Builds training examples from stored evaluation logs and feedback
        records, then compiles BootstrapFewShot with the weighted suitability
        metric (SuitabilityProgram + evaluator weights).
        """
        try:
            from dspy.teleprompt import BootstrapFewShot
            
            print("[INFO] Initializing DSPy BootstrapFewShot optimizer...")
            
            # Load question evaluations to get rubric scores
            evaluations = eval_data.get("evaluations", [])
            
            # Map evaluations by story_title for quick lookup
            eval_map = {}
            for eval_item in evaluations:
                # Skip metadata entries
                if "set_metadata" in eval_item:
                    continue
                    
                # Prefer concise identifiers to avoid depending on full story text
                story_context = (
                    eval_item.get("storybook_id", "")
                    or eval_item.get("story_title", "")
                    or eval_item.get("story_context", "")
                )
                question = eval_item.get("question", "")
                
                # Store rubric scores AND dynamic evaluator scores
                if story_context and question:
                    if story_context not in eval_map:
                        eval_map[story_context] = {}
                    
                    # Get dynamic evaluator scores if they exist
                    dynamic_evaluations = eval_item.get("dynamic_evaluations", {})
                    dynamic_scores = {}
                    for eval_name, eval_data in dynamic_evaluations.items():
                        dynamic_scores[eval_name] = eval_data.get("score", 0.0)
                    
                    eval_map[story_context][question] = {
                        "question_type": eval_item.get("question_type", ""),
                        "suitability_score": eval_item.get("suitability_score", 0.0),
                        "decision": eval_item.get("decision", "unknown"),
                        "evaluation_reasoning": eval_item.get("evaluation_reasoning", ""),
                        "dynamic_evaluations": dynamic_scores  # Add dynamic evaluator scores
                    }
            
            # Extract dynamic evaluators from feedback records (check all school/teacher combinations)
            dynamic_evaluators = []
            records = []
            for school_key in feedback_data:
                if school_key == "records":
                    records.extend(feedback_data.get("records", []))
                elif isinstance(feedback_data[school_key], dict):
                    for teacher_key in feedback_data[school_key]:
                        if isinstance(feedback_data[school_key][teacher_key], list):
                            records.extend(feedback_data[school_key][teacher_key])
            
            for record in records:
                action_taken = record.get("action_taken", {})
                if action_taken.get("type") == "create_new_evaluator":
                    details = action_taken.get("details", {})
                    dynamic_evaluators.append({
                        "name": details.get("evaluator_name", ""),
                        "criteria": details.get("evaluator_description", ""),
                        "status": details.get("status", "active")
                    })
            
            if dynamic_evaluators:
                print(f"[INFO] Using {len(dynamic_evaluators)} dynamic evaluators from orchestrator:")
                for eval in dynamic_evaluators:
                    print(f"   - {eval['name']}: {eval['criteria']}")
            
            # Prepare training examples from feedback with rubric scores
            trainset = []
            # records already loaded above
            
            for record in records:
                # Get the orchestrator's decision
                action_taken = record.get("action_taken", {})
                course_of_action = record.get("course_of_action", {})
                
                # Extract story, segments, objective from record
                story_context = record.get("story_context", "")
                objective = record.get("objective", "")
                story_title = record.get("story_title", "")
                generated_questions = record.get("generated_questions", [])
                
                # Get good/bad feedback for each question
                question_feedbacks = record.get("question_feedbacks", {})
                
                # Get rubric scores for each question
                # Use the same key scheme as above (prefer title/id over full text)
                story_key = (
                    record.get("story_title", "")
                    or record.get("storybook_id", "")
                    or record.get("story_context", "")
                )
                if story_key in eval_map and generated_questions:
                    for idx, q_obj in enumerate(generated_questions):
                        q_text = q_obj.get("question", "") if isinstance(q_obj, dict) else str(q_obj)
                        
                        if q_text and q_text in eval_map[story_key]:
                            rubric_data = eval_map[story_key][q_text]
                            
                            # Get dynamic evaluator scores for this question
                            dynamic_eval_scores = rubric_data.get("dynamic_evaluations", {})
                            
                            # Get individual question feedback (good/bad)
                            individual_feedback = question_feedbacks.get(str(idx), {})
                            teacher_feedback = individual_feedback.get("feedback", "neutral")  # "good", "bad", "neutral"
                            teacher_reasoning = individual_feedback.get("reasoning", "")
                            
                            # Convert teacher feedback to numeric score (for metric)
                            feedback_score = 1.0 if teacher_feedback == "good" else (0.2 if teacher_feedback == "bad" else 0.5)
                            
                            # Create training example with rubric scores AND dynamic evaluator scores AND teacher feedback
                            example = dspy.Example(
                                story=story_context,
                                segments="",  # Can be enhanced with actual segments
                                objective=objective,
                                story_title=story_title,
                                moral_text="",  # Can be enhanced
                                questions=json.dumps([q_text]),
                                question_type=rubric_data.get("question_type", ""),
                                suitability_score=rubric_data.get("suitability_score", 0.0),
                                decision=rubric_data.get("decision", "unknown"),
                                dynamic_evaluations=dynamic_eval_scores,  # Include dynamic evaluator scores
                                teacher_feedback=feedback_score,  # Individual question good/bad score
                                teacher_reasoning=teacher_reasoning  # Reasoning for feedback
                            ).with_inputs("story", "segments", "objective", "story_title", "moral_text")
                            
                            trainset.append(example)
            
            if len(trainset) >= 1:
                # Initialize optimizer with rubric-based metric
                # Note: BootstrapFewShot optimizes prompts, not weights
                # Weights are managed by orchestrator/manager and already loaded via refresh()
                print(f"[INFO] Initializing optimizer with {len(trainset)} training examples...")
                print(f"   Using manager-adjusted weights: {self.evaluator_manager.weights}")
                print(f"   SuitabilityProgram will use these weights in metric evaluation")
                
                self.optimizer = BootstrapFewShot(metric=self._rubric_based_quality_metric)
                
                # Optimize the generator (prompts only, weights remain as set by manager)
                print(f"[INFO] Compiling optimizer...")
                print(f"   Optimizing question generation prompts")
                print(f"   Weights remain as adjusted by orchestrator/manager system")
                self.question_generator = self.optimizer.compile(
                    self.question_generator, 
                    trainset=trainset
                )
                
                self.optimized = True
                print("[INFO] Question generator optimized with rubric-based feedback!")
                print(f"   Final weights (unchanged by optimizer): {self.evaluator_manager.weights}")
                # Note: _update_weights_from_parameters() is called but BootstrapFewShot doesn't tune weights
                # It's kept for potential future use with different optimizers
                self._update_weights_from_parameters()
            else:
                print("[WARN]  Not enough training examples (need at least 1)")
                
        except Exception as e:
            print(f"[WARN]  Error initializing optimizer: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_weights_from_parameters(self) -> None:
        """Synchronize evaluator manager weights from learned DSPy parameters."""
        if not hasattr(self, "suitability_program"):
            return

        param_values: Dict[str, float] = {}
        for name, param in self.suitability_program.weight_params.items():
            try:
                value = float(param())
            except TypeError:
                value = float(getattr(param, "value", self.evaluator_manager.weights.get(name, 0.0)))
            if value < 0.0:
                value = 0.0
            param_values[name] = value

        total = sum(param_values.values())
        if total <= 0:
            return

        for name, value in param_values.items():
            self.evaluator_manager.weights[name] = round(value / total, 4)

        self.evaluator_manager.renormalize()
        self.suitability_program.refresh_parameters()

    def _rubric_based_quality_metric(self, example, prediction, trace=None):
        """Rubric-based quality metric for DSPy optimizer.
        
        This metric converts the suitability rubric into a numerical optimization signal.
        It normalizes binary and scaled criteria to 0-1 and combines them.
        
        The metric learns to maximize suitability scores across all question types.
        It also incorporates dynamic evaluator scores created by the orchestrator.
        """
        try:
            # Get the target suitability score from the example
            target_score = getattr(example, 'suitability_score', 0.5)
            target_decision = getattr(example, 'decision', 'pass')
            
            dynamic_evaluations = getattr(example, 'dynamic_evaluations', {})

            questions_str = prediction.questions
            if isinstance(questions_str, str):
                try:
                    questions = json.loads(questions_str)
                except json.JSONDecodeError:
                    questions = []
            else:
                questions = questions_str

            if isinstance(questions, list) and len(questions) >= 5:
                base_score = 1.0
            else:
                base_score = 0.0

            weighted_scores = []
            if isinstance(questions, list):
                for q in questions:
                    if isinstance(q, dict):
                        q_text = q.get("question", "")
                    else:
                        q_text = str(q)
                    if not q_text:
                        continue
                    context = {
                        "story": getattr(example, 'story', ""),
                        "objective": getattr(example, 'objective', ""),
                        "story_title": getattr(example, 'story_title', ""),
                    }
                    suitability_result = self.suitability_program(question=q_text, context=context)
                    weighted_scores.append(suitability_result.get("weighted_score", 0.0))

            suitability_signal = 0.0
            if weighted_scores:
                suitability_signal = sum(weighted_scores) / len(weighted_scores)
                # Normalize assuming evaluator scores are 1-5
                suitability_signal = max(0.0, min(suitability_signal / 5.0, 1.0))
            else:
                suitability_signal = target_score

            dynamic_adjustment = 1.0
            if dynamic_evaluations:
                dynamic_scores = []
                for eval_data in dynamic_evaluations.values():
                    score = eval_data.get("score", 0.0)
                    normalized_score = (score - 1) / 4 if score > 0 else 0
                    dynamic_scores.append(normalized_score)
                if dynamic_scores:
                    avg_dynamic = sum(dynamic_scores) / len(dynamic_scores)
                    dynamic_adjustment = max(0.5, avg_dynamic)

            teacher_feedback_score = getattr(example, 'teacher_feedback', 0.5)

            if target_decision == "pass":
                optimization_signal = max(suitability_signal, target_score)
            else:
                optimization_signal = min(suitability_signal, 0.2)

            return base_score * optimization_signal * dynamic_adjustment * teacher_feedback_score
            
        except Exception as e:
            print(f"[WARN]  Error in rubric-based quality metric: {e}")
            return 0.5
    
    def generate(self, story: str, segments: List[Dict], objective: str, story_title: str, moral_text: Optional[str] = None, avoidance_instructions: str = "") -> Dict[str, Any]:
        """
        Generate questions based on learning objective.
        
        Args:
            story: Full story content
            segments: Story segments
            objective: Learning objective
            story_title: Title of the story
            moral_text: Optional moral text for additional context
            avoidance_instructions: Feedback from previous failed attempts
        
        Returns:
            {
                "questions": List[Question],
                "learning_objectives": List[str],
                "positive_examples_used": int
            }
        """
        print(f"\n[INFO] Generating questions for: {story_title}")
        print(f"Objective: {objective}")
        if moral_text:
            print(f"Moral context: {moral_text[:50]}...")
        print("=" * 80)
        
        # Note: Optimizer is only initialized in __init__ if optimize_with_feedback=True
        # We do NOT re-check here during generation - optimization happens after teacher feedback
        # This method just generates questions with simple regeneration if they don't pass threshold
        
        try:
            # Format segments
            formatted_segments = self._format_segments(segments)
            
            # Get random variation prompt for diversity
            variation = random.choice(self.variation_prompts)
            
            # Get positive examples from feedback
            positive_examples = self.feedback_collector.get_positive_examples(objective, story_title)
            example_context = f"Building on {len(positive_examples)} successful examples." if positive_examples else ""
            
            print(f"Using {len(positive_examples)} positive examples from feedback")
            print(f"Variation focus: {variation}")
            
            # Generate questions
            result = self.question_generator(
                story=story,
                segments=formatted_segments,
                objective=objective,
                story_title=story_title,
                variation_prompt=f"{variation} {example_context}",
                moral_text=moral_text or "",
                avoidance_instructions=avoidance_instructions if avoidance_instructions else ""
            )
            
            # Parse questions
            questions_raw = result.questions
            if isinstance(questions_raw, str):
                try:
                    questions = json.loads(questions_raw)
                except:
                    questions = []
            else:
                questions = questions_raw
            
            # Parse learning objectives
            learning_objectives_raw = result.learning_objectives
            if isinstance(learning_objectives_raw, str):
                try:
                    learning_objectives = json.loads(learning_objectives_raw)
                except:
                    learning_objectives = []
            else:
                learning_objectives = learning_objectives_raw
            
            print(f"[INFO] Generated {len(questions)} questions")
            print("=" * 80)
            
            return {
                "questions": questions,
                "learning_objectives": learning_objectives,
                "positive_examples_used": len(positive_examples)
            }
            
        except Exception as e:
            print(f"[ERROR] Error generating questions: {e}")
            return {
                "questions": [],
                "learning_objectives": [],
                "error": str(e)
            }
    
    def _format_segments(self, segments: List[Dict]) -> str:
        """Format segments for question generation prompt."""
        return "\n".join([
            f"Segment {i+1} (Pages {seg.get('START', '?')}-{seg.get('END', '?')}): {seg.get('SUMMARY', '')}"
            f"\nReasoning: {seg.get('REASONING', '')}"
            for i, seg in enumerate(segments)
        ])
    
    def record_feedback(self, inputs: Dict, outputs: Dict, question_feedbacks: Dict = None, overall_feedback: str = "positive"):
        """
        Record teacher feedback for optimization.
        
        Args:
            inputs: {story, segments, objective, story_title, original_questions}
            outputs: {questions, learning_objectives}
            question_feedbacks: {question_index: {feedback, reasoning}}
            overall_feedback: "positive" or "negative"
        """
        if overall_feedback == "positive":
            self.feedback_collector.add_positive_feedback(inputs, outputs, question_feedbacks, "")
        else:
            self.feedback_collector.add_negative_feedback(inputs, outputs, question_feedbacks, "")


class FeedbackCollector:
    """
    Sub-module for collecting and managing teacher feedback.
    Stores feedback in DSPy-compatible format for future optimization.
    """
    
    def __init__(self, feedback_file: str = "feedback_dataset.json"):
        self.feedback_file = feedback_file
        self.feedback_data = self._load_feedback()
    
    def _load_feedback(self) -> List[Dict]:
        """Load existing feedback data."""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else [data] if isinstance(data, dict) else []
            except Exception as e:
                print(f"Error loading feedback: {e}")
        return []
    
    def _save_feedback(self):
        """Save feedback data to file."""
        try:
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Feedback saved to {self.feedback_file}")
        except Exception as e:
            print(f"[ERROR] Error saving feedback: {e}")
    
    def add_positive_feedback(self, inputs: Dict, outputs: Dict, question_feedbacks: Dict = None, overall_feedback: str = ""):
        """Add positive feedback with individual question-level details."""
        objective = inputs.get("objective", "moral")
        story_title = inputs.get("story_title", "Unknown")
        
        # Determine iteration number
        existing_entries = [e for e in self.feedback_data if e.get("objective") == objective and e.get("story_title") == story_title]
        iteration = len(existing_entries) + 1
        
        # Use original questions if provided, otherwise use generated
        current_results = inputs.get("original_questions", [])
        if not current_results and outputs.get("questions"):
            current_results = [outputs["questions"]]
        
        # Build question entries with feedback
        question_entries = []
        for i, questions_set in enumerate(current_results):
            for q_idx, question in enumerate(questions_set):
                global_idx = f"{i:04d}{q_idx:04d}"
                
                # Get individual question feedback if available
                q_feedback = question_feedbacks.get(global_idx, {}) if question_feedbacks else {}
                
                question_entry = {
                    "question_id": f"{i:04d}{q_idx:02d}",
                    "question": question.get("question", ""),
                    "type": question.get("type", ""),
                    "difficulty": question.get("difficulty", ""),
                    "explanation": question.get("explanation", ""),
                    "page_number": question.get("page_number", 0),
                    "teacher_feedback": {
                        "feedback": q_feedback.get("feedback", "positive"),
                        "reasoning": q_feedback.get("reasoning", "")
                    } if q_feedback else None
                }
                question_entries.append(question_entry)
        
        # Create feedback entry
        entry = {
            "objective": objective,
            "story_title": story_title,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "overall_feedback": "positive",
            "segments": inputs.get("segments", []),
            "questions": question_entries,
            "learning_objectives": outputs.get("learning_objectives", []),
            "teacher_notes": overall_feedback
        }
        
        self.feedback_data.append(entry)
        self._save_feedback()
    
    def add_negative_feedback(self, inputs: Dict, outputs: Dict, question_feedbacks: Dict = None, overall_feedback: str = ""):
        """Add negative feedback (regeneration requested)."""
        objective = inputs.get("objective", "moral")
        story_title = inputs.get("story_title", "Unknown")
        
        existing_entries = [e for e in self.feedback_data if e.get("objective") == objective and e.get("story_title") == story_title]
        iteration = len(existing_entries) + 1
        
        # Use original questions
        current_results = inputs.get("original_questions", [])
        if not current_results and outputs.get("questions"):
            current_results = [outputs["questions"]]
        
        # Build question entries
        question_entries = []
        for i, questions_set in enumerate(current_results):
            for q_idx, question in enumerate(questions_set):
                global_idx = f"{i:04d}{q_idx:04d}"
                q_feedback = question_feedbacks.get(global_idx, {}) if question_feedbacks else {}
                
                question_entry = {
                    "question_id": f"{i:04d}{q_idx:02d}",
                    "question": question.get("question", ""),
                    "type": question.get("type", ""),
                    "difficulty": question.get("difficulty", ""),
                    "explanation": question.get("explanation", ""),
                    "page_number": question.get("page_number", 0),
                    "teacher_feedback": {
                        "feedback": q_feedback.get("feedback", "negative"),
                        "reasoning": q_feedback.get("reasoning", "")
                    } if q_feedback else None
                }
                question_entries.append(question_entry)
        
        entry = {
            "objective": objective,
            "story_title": story_title,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "overall_feedback": "negative",
            "segments": inputs.get("segments", []),
            "questions": question_entries,
            "learning_objectives": outputs.get("learning_objectives", []),
            "teacher_notes": overall_feedback
        }
        
        self.feedback_data.append(entry)
        self._save_feedback()
    
    def get_positive_examples(self, objective: str, story_title: str) -> List[Dict]:
        """Get positive feedback examples for few-shot learning."""
        positive_examples = [
            entry for entry in self.feedback_data
            if entry.get("objective") == objective 
            and entry.get("story_title") == story_title
            and entry.get("overall_feedback") == "positive"
        ]
        return positive_examples


# ============================================================================
# MAIN ORCHESTRATION CLASS
# ============================================================================

class TeacherInterfaceQuestionGenerator:
    """
    Main orchestration class for the MoralQ Teacher Interface.
    
    This coordinates all sub-modules:
    - MoralGenerator: Extract morals and segment stories
    - QuestionGeneratorModule: Generate questions based on objectives
    - FeedbackCollector: Manage teacher feedback for optimization
    
    All optimization experiments and structured outputs are centralized here.
    """
    
    def __init__(self, feedback_file: str = "feedback_dataset.json"):
        """Initialize the complete question generation system."""
        print("\n" + "="*80)
        print("🎓 MoralQ Teacher Interface - Question Generation System")
        print("="*80)
        
        # Initialize sub-modules
        self.moral_generator = MoralGenerator()
        self.question_generator = QuestionGeneratorModule(feedback_file)
        
        print("[INFO] System initialized successfully")
        print("="*80 + "\n")
    
    def generate_moral_candidates(self, story: str, story_title: str) -> Dict[str, Any]:
        """
        Step 1: Generate multiple moral candidates for teacher selection.
        
        Args:
            story: Full story content
            story_title: Title of the story
            
        Returns:
            {
                "success": bool,
                "candidates": List[{moral, segments, quality_score, ...}],
                "optimization_applied": bool
            }
        """
        return self.moral_generator.generate(story, story_title)
    
    def generate_questions(self, story: str, segments: List[Dict], objective: str, story_title: str, moral_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Step 2: Generate questions based on selected moral and learning objective.
        
        Args:
            story: Full story content
            segments: Story segments from selected moral
            objective: Learning objective (e.g., "empathy", "critical thinking")
            story_title: Title of the story
            moral_text: Optional moral text for additional context
            
        Returns:
            {
                "questions": List[Question],
                "learning_objectives": List[str],
                "positive_examples_used": int
            }
        """
        return self.question_generator.generate(story, segments, objective, story_title, moral_text)
    
    def record_feedback(self, 
                       story: str,
                       segments: List[Dict],
                       objective: str, 
                       story_title: str,
                       original_questions: List[List[Dict]],
                       generated_questions: List[Dict],
                       learning_objectives: List[str],
                       question_feedbacks: Dict[str, Dict],
                       overall_feedback: str = "positive"):
        """
        Step 3: Record teacher feedback for continuous improvement.
        
        Args:
            story: Full story content
            segments: Story segments
            objective: Learning objective
            story_title: Story title
            original_questions: Questions that feedback was given for
            generated_questions: Questions that were generated
            learning_objectives: Learning objectives
            question_feedbacks: Individual question feedback
            overall_feedback: "positive" or "negative"
        """
        inputs = {
            "story": story,
            "segments": segments,
            "objective": objective,
            "story_title": story_title,
            "original_questions": original_questions
        }
        
        outputs = {
            "questions": generated_questions,
            "learning_objectives": learning_objectives
        }
        
        self.question_generator.record_feedback(inputs, outputs, question_feedbacks, overall_feedback)


# ============================================================================
# COMMAND LINE INTERFACE (for testing)
# ============================================================================

def main():
    """Command line interface for testing the system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MoralQ Question Generation System")
    parser.add_argument("--story-file", required=True, help="Path to story JSON file")
    parser.add_argument("--objective", default="moral", help="Learning objective")
    parser.add_argument("--feedback-file", default="feedback_dataset.json", help="Feedback storage file")
    
    args = parser.parse_args()
    
    # Load story
    with open(args.story_file, 'r', encoding='utf-8') as f:
        story_data = json.load(f)
    
    story_content = story_data.get("story", "")
    story_title = story_data.get("title", "Unknown")
    
    # Initialize system
    system = TeacherInterfaceQuestionGenerator(args.feedback_file)
    
    # Step 1: Generate moral candidates
    print("\n[INFO] STEP 1: Generating Moral Candidates")
    moral_result = system.generate_moral_candidates(story_content, story_title)
    
    if not moral_result["success"]:
        print(f"[ERROR] Failed to generate morals: {moral_result.get('error')}")
        return
    
    # Use best candidate
    best_candidate = moral_result["candidates"][0]
    print(f"\n[INFO] Using best moral candidate (score: {best_candidate['quality_score']:.2f})")
    print(f"Moral: {best_candidate['moral']}")
    
    # Step 2: Generate questions
    print("\n[INFO] STEP 2: Generating Questions")
    questions_result = system.generate_questions(
        story_content,
        best_candidate["segments"],
        args.objective,
        story_title
    )
    
    print(f"\n[INFO] Generated {len(questions_result['questions'])} questions")
    for i, q in enumerate(questions_result['questions'], 1):
        print(f"\n{i}. {q['question']}")
        print(f"   Type: {q['type']} | Difficulty: {q['difficulty']} | Page: {q['page_number']}")


if __name__ == "__main__":
    main()

