# Teacher Interface

- For installation and running instructions, see: `SETUP.md`
- For a detailed module-level architecture, see: `ARCHITECTURE.md`
- See `backend/storage/question_evaluations.json` and `backend/storage/teacher_feedback_records.json` for examples

## Conceptual Overview

The system turns a storybook and a teacher-provided objective into a vetted set of questions. It evaluates each question against a rubric (ContextQ), applies teacher feedback via an LLM-based orchestrator, and learns from prior outcomes to improve subsequent generations. Dynamic evaluators can be created on the fly when teachers introduce new quality criteria.

## End-to-End Pipeline

### 1) Moral and Segmentation
- Input: selected storybook(s) and a teacher objective
- The backend loads story text from `ASSETS_PATH` and calls `services/gpt-moral-generation-structured.py` to produce:
  - A moral lesson for the story
  - A structured segmentation of the story (START, END, SUMMARY, REASONING per segment)
- Output feeds the question generator with a concise, structured representation of the story’s content and themes.

### 2) Objective-Based Question Generation
- `services/question_generator.py` (and/or `services/gpt-objective-questions-structured-DSpy.py`) generates a diverse batch of candidate questions aligned to the teacher’s objective.
- DSPy signatures and heuristics are used to encourage variety and coverage of skills while remaining grounded in the story and objective.
- Multiple candidates are produced per attempt to maximize diversity.

### 3) Suitability Evaluation (ContextQ) and Regeneration
- `services/contextq_evaluators.py` evaluates each question against the ContextQ rubric:
  - Classifies question type: Completion, Recall, Open-Ended, Wh, Distancing
  - Applies type-specific evaluators to score suitability and provide reasoning
  - Applies dynamic evaluators (see step 5) when present
  - Decides pass or regenerate using thresholds and dynamic evaluator pass criteria
- Questions that do not pass are regenerated with targeted guidance, then re-evaluated until a final set is reached or attempts are exhausted.

### 4) Teacher Review in the Frontend
- The final evaluated set is displayed in the React UI.
- Teachers mark each question good/bad and provide overall feedback for the set.
- On submit, the backend records the feedback and triggers the orchestrator.

### 5) Feedback Orchestration and Dynamic Evaluators
- `services/teacher_feedback_system.py` collects feedback and uses an LLM-based orchestrator to interpret it into structured actions:
  - Interpreted attributes (e.g., `recall_suitability`)
  - Reformulated instructions (clear, model-ready guidance)
  - Confidence scores
  - Action decision:
    - Reinforce existing: Slightly stabilize current evaluator weights
    - Adjust evaluator: Apply deltas to specific suitability weights (e.g., increase `recall_suitability` weight) and renormalize
    - Add new evaluator: Create a new dynamic evaluator when feedback introduces a new quality dimension (e.g., complexity)
- When adding a dynamic evaluator, the orchestrator’s reformulated instruction and description become the evaluator’s criteria. Future evaluations must satisfy this new rule in addition to the base rubric.

### 6) Optimization with DSPy (Learning from History)
- The system uses DSPy’s BootstrapFewShot-style learning to improve generation prompts over time.
- Training examples are built by correlating:
  - Question-level outcomes and evaluation diagnostics from `backend/storage/question_evaluations.json`
  - Teacher feedback and orchestrator decisions from `backend/storage/teacher_feedback_records.json`
- This builds a memory of what failed, why, and how to improve. The generator incorporates this guidance in subsequent attempts.

### 7) Regeneration Based on Action
- When the teacher clicks Regenerate after feedback, the backend generates a new set while applying the orchestrator’s action:
  - Reinforce: Slightly increases stability across weights
  - Adjust: Re-weights targeted suitability dimensions
  - Add evaluator: Enforces the new dynamic evaluator during evaluation
- Only the final set is logged to `q_testing.json` (no noisy intermediate attempts).

## Data Stores and What They Log

### `backend/storage/question_evaluations.json`
- Purpose: A detailed log of each evaluated question and its final decision.
- Organized by set metadata:
  - `set_metadata`: `{storybook_id, objective, set_number, timestamp}`
- Each question record typically includes:
  - `question`, `question_type`, `type_confidence`
  - `suitability_score` and `evaluation_reasoning`
  - `decision`: `pass` or `regenerate`
  - `dynamic_evaluations`: map of dynamic evaluator results, e.g. `{ "complexity": { "score": 3, "reason": "..." } }`
  - `regenerated`, `regeneration_history` (brief trace of changes)
- Notes:
  - The full story content is not stored; only identifiers like `storybook_id` and `objective` are recorded.

### `backend/storage/teacher_feedback_records.json`
- Purpose: A structured record of teacher feedback sessions, orchestrator interpretation, and actions taken.
- Hierarchy: `{school_id: {teacher_id: [records...]}}`
- For each record, you’ll find:
  - `storybook_id`, `objective`
  - `teacher_feedback` (raw), plus optional question-level good/bad annotations
  - `interpretation`: `{attribute, action, reformulated_instruction, confidence}`
  - `action_taken`: `{type: reinforce_existing|adjust_evaluator|add_new_evaluator, ...}`
  - `course_of_action`: canonicalized instructions/criteria that drive the next generation/evaluation
  - Timestamps and minimal set identifiers
- Notes:
  - The full story is not persisted in this file.

### `q_testing.json`
- Purpose: Lightweight testing log for UI/dev verification.
- Logs only the final set for a run to reduce noise (intermediate attempts are not recorded).
- Includes storybook, objective, set number, and the final question set presented to the teacher.

## Orchestrator Decisions in Detail

1. Interpret feedback: Extract an attribute to target (e.g., `recall_suitability`) and produce a precise instruction suitable for the generator/evaluator.
2. Decide action:
   - Reinforce existing: If feedback is broadly positive or confidence is high that weights should be stabilized
   - Adjust evaluator: If a known rubric dimension requires up/down weighting
   - Add new evaluator: If feedback introduces a new quality dimension not covered by the base rubric
3. Apply changes:
   - Reinforce: Slightly increase all weights, then renormalize
   - Adjust: Apply deltas to targeted weights, then renormalize
   - Add evaluator: Create a new evaluator signature/criteria; future evaluations must pass it

## Dynamic Evaluators

- Created when the orchestrator decides `add_new_evaluator`.
- Defined by a concise description and an instruction derived from the feedback.
- Evaluated as a separate check in `contextq_evaluators.py`.
- Pass criteria are thresholded (e.g., score ≥ 3) and contribute to the final decision.
- Persisted implicitly via `backend/storage/teacher_feedback_records.json` (as part of the course of action) and their results appear in `backend/storage/question_evaluations.json` under `dynamic_evaluations`.

## How the Optimizer Learns

- The optimizer builds examples from prior:
  - Evaluation failures and reasoning (what went wrong and why)
  - Teacher feedback and orchestrator instructions (what to fix and how)
- It then uses DSPy’s few-shot techniques to nudge the generator toward solutions that satisfy both the rubric and teacher preferences.
- Over time, repeated feedback and outcomes make the generator more likely to produce passing questions on the first attempt.

## Frontend User Experience

- Select storybook(s) and an objective
- Generate moral and segments, then approve
- Review the generated questions and their summaries
- Provide question-level good/bad and overall set feedback
- Click Regenerate to apply the orchestrator’s action and produce a refined set

## Notes on Configuration and Paths

- Environment variables live in `teacher_interface/.env` (template in `.env.example`).
- `ASSETS_PATH` points to local storybook assets and must include `qna_json/` and `image/` subfolders.
- Output paths are configurable; for the primary flow, JSON files are stored in `teacher_interface/backend/storage/`.

## Running the System

- See `SETUP.md` for step-by-step instructions (Python venv, dependencies, backend and frontend startup).
- Firebase authentication occurs automatically at backend startup; starting the Firebase client is optional for connectivity checks.

## Troubleshooting at a Glance

- No results in `backend/storage/question_evaluations.json`:
  - Confirm the evaluation pipeline executed; check backend logs for evaluation steps
- No `dynamic_evaluations` despite feedback:
  - Confirm the orchestrator chose `add_new_evaluator`; verify its instruction in `backend/storage/teacher_feedback_records.json`
  - Ensure the backend regeneration path is re-evaluating with dynamic evaluators
- Optimizer not learning:
  - Ensure question texts align across JSONs; mismatches can prevent example construction
  - Verify that evaluation reasons and feedback records are present for the same set

## Glossary

- ContextQ: The pedagogical rubric used to score question suitability by type
- Dynamic Evaluator: A new, feedback-driven criterion added to future evaluations
- Orchestrator: LLM-based component that interprets teacher feedback into actionable changes
- DSPy: Framework used for programmatic prompting and few-shot optimization

---

This README focuses on the conceptual flow. Refer to `ARCHITECTURE.md` for a deeper module breakdown and `SETUP.md` to install and run the project.
