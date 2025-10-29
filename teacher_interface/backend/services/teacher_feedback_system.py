"""
Teacher Feedback Collection and Interpretation System
==================================================

Implements Step 2 of the ContextQ evaluation pipeline:
- Collects overall teacher feedback for question sets
- Interprets feedback using LLM-based agent
- Maps feedback to existing evaluators or creates new ones
- Stores feedback in DSPy-compatible format for optimization
- Implements dynamic evaluator creation for new concepts

Based on the refined plan for question quality evaluation and optimization.

Three Feedback Cases:
Positive feedback → Reinforce current evaluator weights
Known issue → Adjust weights for existing evaluators
New issue → Create new evaluator agent dynamically
"""

import dspy
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

# Configure DSPy with OpenAI
dspy.configure(lm=dspy.LM("openai/gpt-4.1-2025-04-14"))


# ============================================================================
# FEEDBACK INTERPRETATION SIGNATURES
# ============================================================================

class FeedbackOrchestratorSignature(dspy.Signature):
    """
    Orchestrator Agent: Interprets ambiguous teacher feedback and determines the appropriate system action.
    
    This agent serves as a "teacher whisperer" that:
    1. Detects feedback polarity (positive/negative/neutral)
    2. Extracts which attributes/questions the feedback addresses
    3. Determines the intended action type (reinforce existing weights, adjust existing evaluator weights, or create a new evaluator)
    4. Reformulates feedback into clear, actionable instructions for question regeneration
    
    IMPORTANT: The system now uses suitability-based evaluation criteria:
    - For Completion questions: rhyming/repeated phrases
    - For Recall questions: plot elements, sequence of events
    - For Open-Ended questions: opinions/ideas, requires elaboration, personal experience
    - For Wh- questions: story details focus
    - For Distancing questions: child's experiences, requires multi-word answers
    
    Decision Logic:
    - If feedback is positive about existing questions → reinforce_existing
    - If feedback identifies a new quality criterion not covered by existing evaluators → add_new_evaluator
    - If feedback addresses an existing evaluator that needs weight adjustment → adjust_evaluator
    
    """
    
    feedback_text = dspy.InputField(desc="Teacher's written feedback (can be ambiguous)")
    story_context = dspy.InputField(desc="Brief context about the story and learning objective")
    current_questions = dspy.InputField(desc="The questions being evaluated")
    existing_evaluators = dspy.InputField(desc="Comma-separated list of ALL existing evaluator names including: built-in suitability evaluators (completion_suitability, recall_suitability, open_ended_suitability, wh_suitability, distancing_suitability), and previously created dynamic evaluators (e.g., complexity, personal_connection). Use this to determine if the feedback addresses a NEW criterion (add_new_evaluator) or an EXISTING one (adjust_evaluator or reinforce_existing)")
    
    polarity = dspy.OutputField(desc="One of: Positive, Negative, Neutral")
    attribute = dspy.OutputField(desc="What suitability aspect the feedback addresses. For question types: 'completion_suitability' (rhyming/phrases), 'recall_suitability' (plot elements), 'open_ended_suitability' (opinions/elaboration), 'wh_suitability' (story details), 'distancing_suitability' (child experiences). For pedagogical quality: 'complexity', 'personal_connection', 'conversational_quality'")
    action_type = dspy.OutputField(desc="The action to take: 'reinforce_existing' (positive feedback, maintain current weights), 'adjust_evaluator' (modify existing evaluator weights), or 'add_new_evaluator' (create new evaluator for new quality criterion)")
    adjustment_direction = dspy.OutputField(desc="Only if action_type is 'adjust_evaluator': 'increase' (strengthen this evaluator), 'decrease' (weaken this evaluator), or 'modify' (change behavior). Otherwise 'none'")
    adjustment_magnitude = dspy.OutputField(desc="Only if action_type is 'adjust_evaluator': 'small', 'medium', or 'large'. Otherwise 'none'")
    reformulated_instruction = dspy.OutputField(desc="Clear, actionable instruction for regenerating questions (e.g., 'Generate more questions that connect to child's personal experiences' or 'Make questions require more detailed explanations rather than one-word answers')")
    confidence = dspy.OutputField(desc="Confidence in interpretation (0.0-1.0)")

class FeedbackOrchestrator(dspy.Module):
    """Orchestrator agent that interprets teacher feedback using semantic understanding."""
    
    def __init__(self):
        super().__init__()
        self.orchestrator = dspy.Predict(FeedbackOrchestratorSignature)
    
    def forward(self, feedback_text: str, story_context: str = "", current_questions: List[str] = None, existing_evaluators: List[str] = None):
        """
        Interpret teacher feedback and determine the appropriate action type.
        
        Args:
            feedback_text: Teacher's raw feedback
            story_context: Story context for better interpretation
            current_questions: Current questions being evaluated
            existing_evaluators: List of existing evaluator names to determine if attribute is new
            
        Returns:
            Dict with polarity, attribute, action_type, adjustment_direction, adjustment_magnitude, and reformulated instruction
        """
        questions_text = "\n".join(current_questions) if current_questions else "Current questions not available"
        evaluators_text = ", ".join(existing_evaluators) if existing_evaluators else "None"
        
        result = self.orchestrator(
            feedback_text=feedback_text,
            story_context=story_context,
            current_questions=questions_text,
            existing_evaluators=evaluators_text
        )
        
        return {
            "polarity": str(result.polarity).strip(),
            "attribute": str(result.attribute).strip(),
            "action_type": str(result.action_type).strip().lower(),
            "adjustment_direction": str(result.adjustment_direction).strip().lower(),
            "adjustment_magnitude": str(result.adjustment_magnitude).strip().lower(),
            "reformulated_instruction": str(result.reformulated_instruction).strip(),
            "confidence": float(str(result.confidence).strip()) if str(result.confidence).strip().replace(".", "").isdigit() else 0.5
        }


class FeedbackInterpreter(dspy.Signature):
    """Legacy feedback interpreter (kept for backward compatibility).
    
    NOTE: This is being replaced by FeedbackOrchestrator for more reliable interpretation.
    """
    
    story_title = dspy.InputField(desc="Title of the storybook")
    objective = dspy.InputField(desc="Learning objective (e.g., empathy, teamwork)")
    teacher_feedback = dspy.InputField(desc="Teacher's feedback text")
    feedback_type = dspy.InputField(desc="Type of feedback: 'positive' or 'negative'")
    current_evaluators = dspy.InputField(desc="List of existing evaluators: relevance, clarity, depth, engagement, appropriateness")
    
    interpretation = dspy.OutputField(desc="Interpretation of what the teacher is saying")
    affected_evaluators = dspy.OutputField(desc="List of evaluators that this feedback affects (comma-separated)")
    adjustment_direction = dspy.OutputField(desc="Direction of adjustment: 'increase', 'decrease', or 'maintain'")
    adjustment_magnitude = dspy.OutputField(desc="Magnitude of adjustment: 'small', 'medium', or 'large'")
    new_evaluator_needed = dspy.OutputField(desc="Whether a new evaluator is needed: 'yes' or 'no'")
    new_evaluator_name = dspy.OutputField(desc="Name of new evaluator if needed, otherwise 'none'")
    confidence = dspy.OutputField(desc="Confidence in interpretation (0.0-1.0)")


# ============================================================================
# FEEDBACK COLLECTION SYSTEM
# ============================================================================

class TeacherFeedbackCollector:
    """
    Collects and processes teacher feedback for question sets.
    Maps feedback to evaluator dimensions and stores in DSPy-compatible format.
    """
    
    def __init__(self, storage_file: str = "teacher_feedback_records.json"):
        """Initialize the feedback collector."""
        self.storage_file = storage_file
        self.feedback_records = self._load_feedback_records()
        
        # Use orchestrator for semantic feedback interpretation (not rule-based)
        self.orchestrator = FeedbackOrchestrator()
        
        # Legacy interpreter kept for backward compatibility
        self.interpreter = dspy.Predict(FeedbackInterpreter)
        
        # Default suitability-based evaluation weights (NOT the old quality evaluators)
        # These correspond to the ContextQ suitability criteria we now use
        self.evaluator_weights = {
            "completion_suitability": 0.15,  # For rhyming/repeated phrases questions
            "recall_suitability": 0.20,      # For plot element/sequence questions  
            "open_ended_suitability": 0.25,  # For opinion/elaboration questions
            "wh_suitability": 0.20,          # For story details questions
            "distancing_suitability": 0.20   # For personal experience questions
        }
        # Note: Old quality evaluators (relevance, clarity, depth, engagement, appropriateness) are DEPRECATED
        # We now use type-specific suitability criteria instead
        
        print("Teacher Feedback Collector initialized")
        print(f"   - Storage: {storage_file}")
        print(f"   - Evaluator weights: {self.evaluator_weights}")
        print(f"   - Using FeedbackOrchestrator for semantic interpretation")
    
    def _load_feedback_records(self) -> Dict[str, Any]:
        """Load existing feedback records from storage."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except Exception as e:
                print(f"Error loading feedback records: {e}")
        return {}
    
    def _save_feedback_records(self):
        """Save feedback records to storage."""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_records, f, indent=2, ensure_ascii=False)
            print(f"Feedback records saved to {self.storage_file}")
        except Exception as e:
            print(f"Error saving feedback records: {e}")
    
    def collect_feedback(self, 
                        story_title: str,
                        objective: str,
                        teacher_feedback: str,
                        feedback_type: str,
                        question_evaluations: List[Dict[str, Any]],
                        story_context: str = None,
                        generated_questions: List[str] = None,
                        question_feedbacks: Dict[int, Any] = None,
                        teacher_id: str = "default_teacher",
                        school_id: str = "default_school") -> Dict[str, Any]:
        """
        Collect and process teacher feedback for a question set.
        
        Args:
            story_title: Title of the storybook
            objective: Learning objective
            teacher_feedback: Teacher's feedback text
            feedback_type: 'positive' or 'negative'
            question_evaluations: List of question evaluations from ContextQ agents
            story_context: Full story text content (for DSPy optimizer)
            generated_questions: List of generated question texts (for DSPy optimizer)
            question_feedbacks: Dict of individual question feedback {index: {feedback: 'good'|'bad', reasoning: str}}
            teacher_id: Teacher identifier
            school_id: School identifier
            
        Returns:
            Processed feedback record with interpretation and actions
        """
        print(f"\nCollecting teacher feedback for: {story_title}")
        print(f"Feedback type: {feedback_type}")
        print(f"Feedback text: {teacher_feedback}")
        
        # Generate iteration ID
        iteration_id = f"set_{len(self.feedback_records.get(school_id, {}).get(teacher_id, {})) + 1:03d}"
        
        # Get current evaluator scores (average across questions)
        evaluator_scores = self._calculate_evaluator_scores(question_evaluations)
        
        # Interpret feedback using LLM
        interpretation_result = self._interpret_feedback(
            story_title, objective, teacher_feedback, feedback_type, evaluator_scores
        )
        
        # Determine action based on interpretation
        action_result = self._determine_action(interpretation_result, feedback_type)
        
        # Create feedback record
        feedback_record = {
            "story_title": story_title,
            "objective": objective,
            "iteration_id": iteration_id,
            "teacher_feedback": {
                "feedback": feedback_type,
                "reason": teacher_feedback
            },
            "interpretation": interpretation_result,
            "action_taken": action_result,
            "evaluator_scores": evaluator_scores,
            "question_evaluations": question_evaluations,  # Store full evaluation data
            "course_of_action": self._determine_course_of_action(action_result, interpretation_result),
            "timestamp": datetime.now().isoformat(),
            "teacher_id": teacher_id,
            "school_id": school_id,
            "story_context": story_context,  # Full story text for DSPy optimizer
            "generated_questions": generated_questions or [],  # Generated question texts for DSPy optimizer
            "question_feedbacks": question_feedbacks or {}  # Individual question good/bad feedback
        }
        
        # Store feedback record
        self._store_feedback_record(feedback_record, school_id, teacher_id)
        
        # Log individual question feedback if present
        if question_feedbacks:
            print(f"✅ Stored individual feedback for {len(question_feedbacks)} questions")
        
        # Update evaluator weights if needed
        if action_result["type"] == "adjust_evaluator":
            self._update_evaluator_weights(action_result)
        elif action_result["type"] == "reinforce_existing":
            self._reinforce_evaluator_weights()
        
        print(f"✅ Feedback processed and stored")
        print(f"   Action: {action_result['type']}")
        if action_result["type"] == "adjust_evaluator":
            print(f"   Affected evaluators: {action_result['affected']}")
        
        return feedback_record
    
    def _calculate_evaluator_scores(self, question_evaluations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average scores for each evaluator across all questions."""
        if not question_evaluations:
            return {}
        
        evaluator_totals = {}
        evaluator_counts = {}
        
        for eval_data in question_evaluations:
            scores = eval_data.get("scores", {})
            for evaluator, score in scores.items():
                if evaluator not in evaluator_totals:
                    evaluator_totals[evaluator] = 0
                    evaluator_counts[evaluator] = 0
                evaluator_totals[evaluator] += score
                evaluator_counts[evaluator] += 1
        
        # Calculate averages
        evaluator_averages = {}
        for evaluator in evaluator_totals:
            evaluator_averages[evaluator] = evaluator_totals[evaluator] / evaluator_counts[evaluator]
        
        return evaluator_averages
    
    def _load_dynamic_evaluators_from_records(self) -> List[str]:
        """Load names of previously created dynamic evaluators from feedback records."""
        dynamic_evaluator_names = []
        try:
            # Navigate through nested structure: school -> teacher -> records
            for school_id in self.feedback_records:
                if school_id == "records":
                    # Flat structure
                    records = self.feedback_records.get("records", [])
                elif isinstance(self.feedback_records[school_id], dict):
                    # Nested structure: school -> teacher
                    records = []
                    for teacher_id in self.feedback_records[school_id]:
                        if isinstance(self.feedback_records[school_id][teacher_id], list):
                            records.extend(self.feedback_records[school_id][teacher_id])
                else:
                    continue
                
                # Extract evaluator names from records with add_new_evaluator action
                for record in records:
                    action_taken = record.get("action_taken", {})
                    if action_taken.get("type") == "add_new_evaluator":
                        details = action_taken.get("details", {})
                        if details.get("status") == "active":
                            evaluator_name = details.get("evaluator_name", "")
                            if evaluator_name and evaluator_name not in dynamic_evaluator_names:
                                dynamic_evaluator_names.append(evaluator_name)
        except Exception as e:
            print(f"   Warning: Error loading dynamic evaluators from records: {e}")
        
        return dynamic_evaluator_names
    
    def _interpret_feedback(self, 
                           story_title: str,
                           objective: str,
                           teacher_feedback: str,
                           feedback_type: str,
                           evaluator_scores: Dict[str, float]) -> Dict[str, Any]:
        """Use orchestrator agent to semantically interpret teacher feedback and determine action type."""
        
        # Build comprehensive list of existing evaluators:
        # 1. Built-in suitability evaluators (from evaluator_weights)
        # 2. Evaluators from current question evaluations (evaluator_scores)
        # 3. Previously created dynamic evaluators (from feedback records)
        built_in_evaluators = set(self.evaluator_weights.keys())
        scored_evaluators = set(evaluator_scores.keys())
        dynamic_evaluators = set(self._load_dynamic_evaluators_from_records())
        
        current_evaluators = list(built_in_evaluators | scored_evaluators | dynamic_evaluators)
        
        if dynamic_evaluators:
            print(f"   Including {len(dynamic_evaluators)} previously created dynamic evaluators in context")
        
        story_context = f"Story: {story_title}\nObjective: {objective}"
        
        try:
            # Use orchestrator for semantic interpretation and decision
            orchestrator_result = self.orchestrator(
                feedback_text=teacher_feedback,
                story_context=story_context,
                current_questions=None,  # Will be filled when available
                existing_evaluators=current_evaluators
            )
            
            # Map orchestrator output to interpretation structure
            # The orchestrator now directly outputs action_type, so we use it directly
            action_type = orchestrator_result['action_type']
            attribute = orchestrator_result['attribute']
            
            interpretation = {
                "interpretation": f"Feedback addressed {attribute} - orchestrator determined: {action_type}",
                "attribute": attribute,
                "action_type": action_type,
                "affected_evaluators": [attribute] if attribute else [],
                "adjustment_direction": orchestrator_result['adjustment_direction'] if action_type == "adjust_evaluator" else "none",
                "adjustment_magnitude": orchestrator_result['adjustment_magnitude'] if action_type == "adjust_evaluator" else "none",
                "new_evaluator_needed": (action_type == "add_new_evaluator"),
                "new_evaluator_name": attribute if action_type == "add_new_evaluator" else "none",
                "confidence": orchestrator_result['confidence'],
                "reformulated_instruction": orchestrator_result['reformulated_instruction']
            }
            
            print(f"   Orchestrator interpretation: {interpretation['interpretation']}")
            print(f"   Attribute: {attribute}")
            print(f"   Action Type: {action_type}")
            if action_type == "adjust_evaluator":
                print(f"   Adjustment: {interpretation['adjustment_direction']} ({interpretation['adjustment_magnitude']})")
            print(f"   Reformulated instruction: {interpretation['reformulated_instruction']}")
            print(f"   Confidence: {interpretation['confidence']}")
            
            return interpretation
            
        except Exception as e:
            print(f"   Error interpreting feedback: {e}")
            # Return default interpretation
            return {
                "interpretation": f"Teacher provided {feedback_type} feedback: {teacher_feedback}",
                "attribute": "",
                "action_type": "no_action",
                "affected_evaluators": [],
                "adjustment_direction": "none",
                "adjustment_magnitude": "none",
                "new_evaluator_needed": False,
                "new_evaluator_name": "none",
                "confidence": 0.3
            }
    
    def _determine_action(self, interpretation: Dict[str, Any], feedback_type: str) -> Dict[str, Any]:
        """
        Determine what action to take based on orchestrator's decision.
        
        The orchestrator now directly outputs the action_type, so this function
        primarily validates and formats the action with necessary details.
        """
        
        # Override with reinforce if explicitly positive feedback (for backward compatibility)
        action_type = interpretation.get("action_type", "no_action")
        if feedback_type == "positive" and action_type != "add_new_evaluator":
            action_type = "reinforce_existing"
        
        # Handle each action type based on orchestrator's decision
        if action_type == "reinforce_existing":
            return {
                "type": "reinforce_existing",
                "affected": [],
                "delta": {},
                "reason": "Orchestrator determined: reinforce existing evaluator weights"
            }
        
        elif action_type == "add_new_evaluator":
            evaluator_name = interpretation.get("new_evaluator_name") or interpretation.get("attribute", "")
            if not evaluator_name:
                return {
                    "type": "no_action",
                    "affected": [],
                    "delta": {},
                    "reason": "Orchestrator indicated add_new_evaluator but no evaluator name provided"
                }
            
            description = f"Evaluates questions for {evaluator_name.replace('_', ' ')} based on teacher feedback"
            
            evaluator_details = {
                "evaluator_name": evaluator_name,
                "evaluator_description": description,
                "status": "active"
            }
            
            print(f"   Creating new evaluator: {evaluator_name}")
            print(f"   Description: {description}")
            
            return {
                "type": "add_new_evaluator",
                "affected": [evaluator_name],
                "delta": {},
                "reason": f"Orchestrator determined: new evaluator needed for {evaluator_name}",
                "reformulated_instruction": interpretation.get("reformulated_instruction", ""),
                "details": evaluator_details
            }
        
        elif action_type == "adjust_evaluator":
            # Calculate delta based on adjustment direction and magnitude
            delta = {}
            magnitude_map = {"small": 0.1, "medium": 0.2, "large": 0.3}
            adjustment_magnitude = magnitude_map.get(interpretation.get("adjustment_magnitude", "medium"), 0.2)
            
            adjustment_direction = interpretation.get("adjustment_direction", "none").lower()
            if adjustment_direction == "increase":
                adjustment_value = adjustment_magnitude
            elif adjustment_direction in ["decrease", "reduce"]:
                adjustment_value = -adjustment_magnitude
            elif adjustment_direction == "modify":
                # For modify, apply a small adjustment
                adjustment_value = adjustment_magnitude * 0.5
            else:
                adjustment_value = 0
            
            # Apply delta to affected evaluators
            affected_evaluators = interpretation.get("affected_evaluators", [])
            if not affected_evaluators:
                # Fallback to attribute if no affected evaluators listed
                attribute = interpretation.get("attribute", "")
                if attribute:
                    affected_evaluators = [attribute]
            
            for evaluator in affected_evaluators:
                delta[evaluator] = adjustment_value
            
            return {
                "type": "adjust_evaluator",
                "affected": affected_evaluators,
                "delta": delta,
                "reason": f"Orchestrator determined: {adjustment_direction} ({adjustment_magnitude}) for {affected_evaluators}",
                "reformulated_instruction": interpretation.get("reformulated_instruction", "")
            }
        
        else:
            # No action or unclear
            return {
                "type": "no_action",
                "affected": [],
                "delta": {},
                "reason": f"Orchestrator returned action_type: {action_type} (not recognized or no action needed)"
            }
    
    def _determine_course_of_action(self, action_taken: Dict[str, Any], interpretation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the complete course of action for DSPy optimization.
        
        This includes:
        - What to do (reinforce/adjust/create)
        - Which evaluators are affected
        - How to modify them (weights/deltas/new agent)
        - Training examples for DSPy
        """
        course_of_action = {
            "action_type": action_taken["type"],
            "rationale": action_taken.get("reason", ""),
            "affected_evaluators": action_taken.get("affected", []),
            "weight_changes": action_taken.get("delta", {}),
            "training_strategy": "",
            "reformulated_instruction": action_taken.get("reformulated_instruction", "")  # Add reformulated instruction
        }
        
        # Determine training strategy based on action type
        if action_taken["type"] == "reinforce_existing":
            course_of_action["training_strategy"] = "maintain_current_weights"
            course_of_action["dspy_action"] = "no_optimization"
        
        elif action_taken["type"] == "adjust_evaluator":
            course_of_action["training_strategy"] = "bootstrap_fewshot_with_adjusted_weights"
            course_of_action["dspy_action"] = "optimize_question_generator"
            course_of_action["optimization_target"] = action_taken.get("affected", [])
            # Include reformulated instruction for regeneration
            course_of_action["instruction"] = action_taken.get("reformulated_instruction", "")
        
        elif action_taken["type"] == "add_new_evaluator":
            course_of_action["training_strategy"] = "create_dynamic_evaluator"
            course_of_action["dspy_action"] = "create_and_integrate_evaluator"
            course_of_action["new_evaluator_name"] = interpretation.get("new_evaluator_name", "")
            course_of_action["new_evaluator_description"] = f"Evaluates questions for {interpretation.get('new_evaluator_name', '')} based on teacher feedback"
            # Include reformulated instruction for new evaluator criteria
            course_of_action["instruction"] = action_taken.get("reformulated_instruction", "")
        
        else:
            course_of_action["training_strategy"] = "no_action"
            course_of_action["dspy_action"] = "no_optimization"
        
        return course_of_action
    
    def _store_feedback_record(self, feedback_record: Dict[str, Any], school_id: str, teacher_id: str):
        """Store feedback record in the hierarchical structure."""
        
        if school_id not in self.feedback_records:
            self.feedback_records[school_id] = {}
        
        if teacher_id not in self.feedback_records[school_id]:
            self.feedback_records[school_id][teacher_id] = []
        
        self.feedback_records[school_id][teacher_id].append(feedback_record)
        self._save_feedback_records()
    
    def _update_evaluator_weights(self, action_result: Dict[str, Any]):
        """Update evaluator weights based on feedback action."""
        
        if action_result["type"] != "adjust_evaluator":
            return
        
        delta = action_result["delta"]
        
        for evaluator, adjustment in delta.items():
            if evaluator in self.evaluator_weights:
                # Apply adjustment (clamp between 0.05 and 0.4)
                new_weight = max(0.05, min(0.4, self.evaluator_weights[evaluator] + adjustment))
                self.evaluator_weights[evaluator] = new_weight
                print(f"   Updated {evaluator} weight: {self.evaluator_weights[evaluator]:.3f}")
        
        # Renormalize weights to sum to 1.0
        total_weight = sum(self.evaluator_weights.values())
        for evaluator in self.evaluator_weights:
            self.evaluator_weights[evaluator] /= total_weight
        
        print(f"   Renormalized weights: {self.evaluator_weights}")

    def _reinforce_evaluator_weights(self, stabilization_rate: float = 0.05):
        """Slightly reinforce current evaluator weights (positive feedback)."""
        # Increase each weight by a small factor, then renormalize
        for key in list(self.evaluator_weights.keys()):
            self.evaluator_weights[key] = min(
                0.5,  # hard cap to avoid any single weight dominating
                round(self.evaluator_weights[key] * (1.0 + stabilization_rate), 4)
            )
        total = sum(self.evaluator_weights.values())
        if total > 0:
            for key in self.evaluator_weights:
                self.evaluator_weights[key] = round(self.evaluator_weights[key] / total, 4)
        print(f"   Reinforced weights: {self.evaluator_weights}")


# ============================================================================
# DSPy-COMPATIBLE TRAINING DATA FORMAT
# ============================================================================

class DSPyTrainingDataFormatter:
    """
    Formats feedback data into DSPy-compatible training examples.
    Creates normalized training records for optimizer consumption.
    """
    
    def __init__(self):
        """Initialize the formatter."""
        print("DSPy Training Data Formatter initialized")
    
    def format_training_record(self, 
                              feedback_record: Dict[str, Any],
                              story_context: str,
                              generated_questions: List[str]) -> Dict[str, Any]:
        """
        Format a feedback record into DSPy-compatible training data.
        
        Args:
            feedback_record: Processed feedback record
            story_context: Story content and context
            generated_questions: List of generated question texts
            
        Returns:
            DSPy-compatible training record
        """
        
        # Normalize evaluator scores to 0-1 range
        evaluator_scores = feedback_record.get("evaluator_scores", {})
        normalized_scores = {}
        for evaluator, score in evaluator_scores.items():
            normalized_scores[evaluator] = score / 5.0  # Convert from 1-5 to 0-1
        
        # Teacher feedback score (1.0 for positive, 0.0 for negative)
        teacher_feedback_score = 1.0 if feedback_record["teacher_feedback"]["feedback"] == "positive" else 0.0
        
        # Combine all metrics
        metrics = {
            **normalized_scores,
            "teacher_feedback": teacher_feedback_score
        }
        
        training_record = {
            "inputs": {
                "story_title": feedback_record["story_title"],
                "objective": feedback_record["objective"],
                "story_context": story_context
            },
            "outputs": {
                "questions": generated_questions
            },
            "metrics": metrics,
            "action_taken": feedback_record["action_taken"],
            "metadata": {
                "iteration_id": feedback_record["iteration_id"],
                "timestamp": feedback_record["timestamp"],
                "teacher_id": feedback_record["teacher_id"],
                "school_id": feedback_record["school_id"]
            }
        }
        
        return training_record
    
    def create_training_dataset(self, 
                               feedback_records: List[Dict[str, Any]],
                               story_contexts: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Create a complete training dataset from feedback records.
        
        Args:
            feedback_records: List of feedback records
            story_contexts: Mapping of story titles to contexts
            
        Returns:
            List of DSPy-compatible training records
        """
        
        training_dataset = []
        
        for record in feedback_records:
            story_title = record["story_title"]
            story_context = story_contexts.get(story_title, "")
            
            # Extract question texts from the record (if available)
            questions = record.get("generated_questions", [])
            
            training_record = self.format_training_record(
                record, story_context, questions
            )
            
            training_dataset.append(training_record)
        
        print(f"Created training dataset with {len(training_dataset)} records")
        return training_dataset


# ============================================================================
# MULTI-OBJECTIVE METRIC FOR DSPy OPTIMIZER
# ============================================================================

class QuestionQualityMetric:
    """
    Multi-objective metric for DSPy optimizer.
    Aggregates evaluator scores while respecting teacher feedback.
    """
    
    def __init__(self, evaluator_weights: Dict[str, float] = None):
        """Initialize the metric with evaluator weights."""
        self.evaluator_weights = evaluator_weights or {
            "relevance": 0.2,
            "clarity": 0.2,
            "depth": 0.25,
            "engagement": 0.2,
            "appropriateness": 0.15
        }
        
        print(f"Question Quality Metric initialized")
        print(f"   Evaluator weights: {self.evaluator_weights}")
    
    def __call__(self, prediction, target=None) -> float:
        """
        Compute the quality metric for a prediction.
        
        Args:
            prediction: DSPy prediction object
            target: Target metrics (if available)
            
        Returns:
            Quality score (0.0-1.0)
        """
        
        if target is None:
            # If no target, return a default score
            return 0.5
        
        metrics = target.get("metrics", {})
        
        # Calculate weighted sum of evaluator scores
        weighted_sum = 0.0
        total_weight = 0.0
        
        for evaluator, weight in self.evaluator_weights.items():
            if evaluator in metrics:
                weighted_sum += metrics[evaluator] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.5  # Default score if no weights
        
        base_score = weighted_sum / total_weight
        
        # Apply teacher feedback reinforcement
        teacher_feedback = metrics.get("teacher_feedback", 0.5)
        feedback_weight = 1.1 if teacher_feedback > 0.5 else 0.9
        
        final_score = base_score * feedback_weight
        
        # Clamp to valid range
        return max(0.0, min(1.0, final_score))
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update evaluator weights."""
        self.evaluator_weights.update(new_weights)
        print(f"Updated evaluator weights: {self.evaluator_weights}")


# ============================================================================
# MAIN FEEDBACK SYSTEM
# ============================================================================

class TeacherFeedbackSystem:
    """
    Main system for collecting, interpreting, and storing teacher feedback.
    Integrates with ContextQ evaluators and prepares data for DSPy optimization.
    """
    
    def __init__(self, storage_file: str = "teacher_feedback_records.json"):
        """Initialize the feedback system."""
        self.collector = TeacherFeedbackCollector(storage_file)
        self.formatter = DSPyTrainingDataFormatter()
        self.metric = QuestionQualityMetric(self.collector.evaluator_weights)
        
        print("\n" + "="*80)
        print("Teacher Feedback System Initialized")
        print("="*80)
    
    def process_feedback(self, 
                        story_title: str,
                        objective: str,
                        teacher_feedback: str,
                        feedback_type: str,
                        question_evaluations: List[Dict[str, Any]],
                        story_context: str = "",
                        generated_questions: List[str] = None,
                        question_feedbacks: Dict[int, Any] = None,
                        teacher_id: str = "default_teacher",
                        school_id: str = "default_school") -> Dict[str, Any]:
        """
        Process teacher feedback and create training data.
        
        Args:
            story_title: Title of the storybook
            objective: Learning objective
            teacher_feedback: Teacher's feedback text
            feedback_type: 'positive' or 'negative'
            question_evaluations: Question evaluations from ContextQ agents
            story_context: Story content
            generated_questions: List of generated question texts
            question_feedbacks: Dict of individual question feedback {index: {feedback: 'good'|'bad', reasoning: str}}
            teacher_id: Teacher identifier
            school_id: School identifier
            
        Returns:
            Complete feedback processing result
        """
        
        # Collect and interpret feedback
        feedback_record = self.collector.collect_feedback(
            story_title, objective, teacher_feedback, feedback_type,
            question_evaluations, story_context, generated_questions, question_feedbacks, teacher_id, school_id
        )
        
        # Format as DSPy training data
        training_record = self.formatter.format_training_record(
            feedback_record, story_context, generated_questions or []
        )
        
        # Update metric weights
        self.metric.update_weights(self.collector.evaluator_weights)
        
        result = {
            "feedback_record": feedback_record,
            "training_record": training_record,
            "action_taken": feedback_record["action_taken"],
            "evaluator_weights": self.collector.evaluator_weights
        }
        
        print(f"\n✅ Feedback processing complete")
        print(f"   Action: {result['action_taken']['type']}")
        print(f"   Training record created for DSPy optimization")
        
        return result
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about collected feedback."""
        total_records = 0
        positive_count = 0
        negative_count = 0
        
        for school_data in self.collector.feedback_records.values():
            for teacher_data in school_data.values():
                for record in teacher_data:
                    total_records += 1
                    if record["teacher_feedback"]["feedback"] == "positive":
                        positive_count += 1
                    else:
                        negative_count += 1
        
        return {
            "total_feedback_records": total_records,
            "positive_feedback": positive_count,
            "negative_feedback": negative_count,
            "current_evaluator_weights": self.collector.evaluator_weights
        }


# ============================================================================
# COMMAND LINE INTERFACE (for testing)
# ============================================================================

def main():
    """Test the teacher feedback system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Teacher Feedback Collection System")
    parser.add_argument("--story", default="Grumpy Monkey", help="Story title")
    parser.add_argument("--objective", default="empathy", help="Learning objective")
    parser.add_argument("--feedback", default="Questions were too factual and lacked emotional connection.", help="Teacher feedback")
    parser.add_argument("--type", default="negative", choices=["positive", "negative"], help="Feedback type")
    parser.add_argument("--storage", default="teacher_feedback_records.json", help="Storage file")
    
    args = parser.parse_args()
    
    # Initialize system
    feedback_system = TeacherFeedbackSystem(args.storage)
    
    # Sample question evaluations (from ContextQ agents)
    sample_evaluations = [
        {
            "question_id": "Q001",
            "scores": {"relevance": 5, "clarity": 4, "depth": 2, "engagement": 3, "appropriateness": 5},
            "average_score": 3.8
        },
        {
            "question_id": "Q002", 
            "scores": {"relevance": 4, "clarity": 5, "depth": 1, "engagement": 2, "appropriateness": 5},
            "average_score": 3.4
        }
    ]
    
    # Process feedback
    result = feedback_system.process_feedback(
        story_title=args.story,
        objective=args.objective,
        teacher_feedback=args.feedback,
        feedback_type=args.type,
        question_evaluations=sample_evaluations,
        story_context="Jim is a monkey who feels grumpy but learns about emotions.",
        generated_questions=["How did Jim feel?", "What happened next?"]
    )
    
    # Print results
    print(f"\n📊 Feedback Processing Results:")
    print(f"Action: {result['action_taken']['type']}")
    print(f"Affected evaluators: {result['action_taken']['affected']}")
    print(f"Evaluator weights: {result['evaluator_weights']}")
    
    # Print statistics
    stats = feedback_system.get_feedback_stats()
    print(f"\n📈 System Statistics:")
    print(f"Total records: {stats['total_feedback_records']}")
    print(f"Positive feedback: {stats['positive_feedback']}")
    print(f"Negative feedback: {stats['negative_feedback']}")


# ============================================================================
# DYNAMIC EVALUATOR CREATION SYSTEM
# ============================================================================

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
    
    def load_feedback_records(self) -> List[Dict[str, Any]]:
        """Load all feedback records from the file."""
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                data = json.load(f)
                all_records = []
                for school_id, school_data in data.items():
                    if isinstance(school_data, dict):
                        for teacher_id, teacher_records in school_data.items():
                            if isinstance(teacher_records, list):
                                all_records.extend(teacher_records)
                return all_records
        return []
    
    def get_pending_optimizations(self) -> List[Dict[str, Any]]:
        """Get all feedback records that need DSPy optimization."""
        records = self.load_feedback_records()
        pending = []
        
        for record in records:
            course_of_action = record.get("course_of_action")
            if course_of_action:
                dspy_action = course_of_action.get("dspy_action")
                if dspy_action != "no_optimization":
                    pending.append({
                        "record": record,
                        "course_of_action": course_of_action,
                        "dspy_action": dspy_action
                    })
        
        return pending
    
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


if __name__ == "__main__":
    main()
