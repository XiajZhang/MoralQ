# Teacher Interface

- For installation and running instructions, see: `SETUP.md`
- For a detailed module-level architecture, see: `ARCHITECTURE.md`
- See `backend/storage/question_evaluations.json` and `backend/storage/teacher_feedback_records.json` for examples

## Conceptual Overview

The system turns a storybook and a teacher-provided objective into a vetted set of questions. It evaluates each question against a rubric (ContextQ), applies teacher feedback via an LLM-based orchestrator, and learns from prior outcomes to improve subsequent generations. Dynamic evaluators can be created on the fly when teachers introduce new quality criteria.

## System Architecture

### Evaluator System (`backend/evaluators/`)

**Modular Design:**
- `base_evaluator.py`: Abstract base class (`BaseEvaluator`) defining unified evaluator interface
- `registry.py`: Dual registry system (`CORE_EVALUATOR_REGISTRY`, `DYNAMIC_EVALUATOR_REGISTRY`) for plugin-based discovery
- `manager.py`: `EvaluatorManager` - centralizes evaluator loading, weighting, and dynamic registration
- `suitability_evaluators.py`: Core suitability evaluators (Completion, Recall, Open-Ended, Wh, Distancing) + dynamic evaluator template
- `dynamic_template.py`: Helper functions for creating dynamic evaluators from LLM feedback

**Key Principles:**
- **Separation of Concerns**: Evaluator logic (Python classes) separate from metadata (JSON configs)
- **Plugin Pattern**: Evaluators registered via decorators, discovered dynamically
- **Unified Interface**: All evaluators inherit from `BaseEvaluator`
- **Dual Registry**: Core (built-in) vs Dynamic (runtime-created) evaluators

### Feedback Processing (`backend/services/teacher_feedback_system.py`)

**Components:**
- `FeedbackOrchestrator`: LLM-based semantic interpretation of teacher feedback
- `OrchestratorRouter`: Routes `FeedbackMessage` objects to appropriate handlers
- `EvaluatorManager`: Applies weight adjustments and creates dynamic evaluators
- `DynamicEvaluatorCreator`: Creates new evaluator classes from feedback

**Flow:**
```
Teacher Feedback
    ↓
Orchestrator (interprets semantically)
    ↓
Router (creates FeedbackMessage)
    ↓
EvaluatorManager (applies changes)
    ↓
- Adjust weights (delta-based, renormalized)
- Create new evaluator (registered + persisted)
- Reinforce existing weights
```

### Question Generation & Optimization (`backend/services/question_generator.py`)

**Components:**
- `QuestionGeneratorModule`: Main question generation with optional DSPy optimization
- `SuitabilityProgram`: Aggregates evaluator scores using manager-adjusted weights
- `MoralGenerator`: Extracts moral lessons and segments stories

**Optimization:**
- Uses `BootstrapFewShot` to optimize question generation prompts
- **Weights are NOT tuned by optimizer** - managed by orchestrator/manager system
- Optimizer uses manager-adjusted weights in metric evaluation
- Only optimizes prompts, not evaluator weights

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
  - **Orchestrator** (`FeedbackOrchestrator`): Interprets feedback semantically using LLM
  - **Router** (`OrchestratorRouter`): Creates `FeedbackMessage` objects and routes to manager
  - **Manager** (`EvaluatorManager`): Applies weight adjustments or creates new evaluators
  - **Action Types**:
    - **Reinforce existing**: Slightly increase all evaluator weights, then renormalize
    - **Adjust evaluator**: Apply deltas (small/medium/large) to specific evaluator weights, then renormalize
    - **Add new evaluator**: Create dynamic evaluator with LLM-backed evaluation logic
- **Dynamic Evaluator Creation**:
  - Orchestrator's reformulated instruction becomes the evaluator's prompt/criteria
  - `DynamicEvaluatorCreator` creates evaluator class using `make_dynamic_evaluator_class`
  - Evaluator registered in `DYNAMIC_EVALUATOR_REGISTRY` and persisted to `data/dynamic_evaluators.json`
  - Future evaluations must pass this new evaluator in addition to core rubric

### 6) Optimization with DSPy (Learning from History)
- The system uses DSPy's `BootstrapFewShot` to optimize question generation prompts over time.
- **Important**: The optimizer optimizes **prompts only**, not evaluator weights
- **Weight Management**: Weights are adjusted by the orchestrator/manager system, not learned by the optimizer
- **Training Examples** are built by correlating:
  - Question-level outcomes and evaluation diagnostics from `backend/storage/question_evaluations.json`
  - Teacher feedback and orchestrator decisions from `backend/storage/teacher_feedback_records.json`
  - Rubric scores (suitability scores) and dynamic evaluator scores
  - Individual question good/bad marks
- **Optimizer Initialization** (only when `optimize_with_feedback=True`):
  - Refreshes `EvaluatorManager` to load latest weights and dynamic evaluators
  - Uses manager-adjusted weights in `SuitabilityProgram` for metric evaluation
  - Optimizes question generation prompts to maximize quality given current weights
- This builds a memory of what failed, why, and how to improve. The generator incorporates this guidance in subsequent attempts.

### 7) Regeneration Based on Action
- When the teacher clicks Regenerate after feedback, the backend:
  1. Records feedback (if provided) → Orchestrator → Manager adjusts weights/creates evaluators
  2. Calls `call_objective_question_generation_script(optimize_with_feedback=True)`
  3. Initializes optimizer with manager-adjusted weights and dynamic evaluators
  4. Generates new questions using optimized prompts
  5. Evaluates using manager-adjusted weights and dynamic evaluators
- **Manager Actions Applied**:
  - **Reinforce**: Slightly increases all weights, then renormalizes
  - **Adjust**: Applies deltas to targeted evaluator weights, then renormalizes
  - **Add evaluator**: New dynamic evaluator is loaded and used in evaluation
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
  - `dynamic_evaluations`: map of dynamic evaluator results, e.g. `{ "complexity": { "score": 3, "decision": "pass", "reasoning": "..." } }`
  - `regenerated`: `true` if question came from regeneration batch
  - `batch_number`: Batch number (1 = initial, 2+ = regenerated)
  - `regeneration_history`: 1:1 mapping showing original failed question → regenerated question, with avoidance instructions and metadata
- **Regeneration History Format**:
  ```json
  "regeneration_history": {
    "original_failed_question": "What is the name of...",
    "original_failure_reason": "The question asks for factual information...",
    "original_batch": 1,
    "regenerated_question": "Why do you think...",
    "regenerated_batch": 2,
    "avoidance_instructions": "Avoid previous issues: ...",
    "regeneration_metadata": {
      "suitability_score": 4.5,
      "question_type": "Open-Ended",
      "total_failed_in_previous_batch": 5,
      "current_batch_performance": {...}
    }
  }
  ```
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

- **Creation**: Created when the orchestrator decides `add_new_evaluator`.
- **Definition**: Defined by a concise description and an instruction (prompt) derived from the feedback.
- **Implementation**: 
  - Created using `make_dynamic_evaluator_class` template
  - Uses LLM-backed evaluation via `dspy.Predict` with dynamic signature
  - Registered in `DYNAMIC_EVALUATOR_REGISTRY` at runtime
- **Persistence**:
  - Metadata stored in `backend/data/dynamic_evaluators.json`
  - Registered in `DYNAMIC_EVALUATOR_REGISTRY` (loaded on `EvaluatorManager.refresh()`)
  - Results appear in `backend/storage/question_evaluations.json` under `dynamic_evaluations`
- **Evaluation**: 
  - Evaluated as a separate check in `contextq_evaluators.py`
  - Pass criteria are thresholded (e.g., score ≥ 3) and contribute to the final decision
  - All dynamic evaluators must pass for question to pass (if any fails, question is regenerated)

## How the Optimizer Learns

- **Training Data**: The optimizer builds examples from:
  - Evaluation failures and reasoning (what went wrong and why)
  - Teacher feedback and orchestrator instructions (what to fix and how)
  - Rubric scores (suitability scores from core evaluators)
  - Dynamic evaluator scores (from manager-created evaluators)
  - Individual question good/bad marks
- **Optimization Process**:
  - Uses DSPy's `BootstrapFewShot` to optimize question generation prompts
  - **Does NOT optimize weights** - weights are managed by orchestrator/manager
  - Uses manager-adjusted weights in `SuitabilityProgram` for metric evaluation
  - Metric evaluates generated questions using current weights
- **Weight Management**:
  - Weights adjusted by `EvaluatorManager` based on orchestrator feedback
  - Optimizer uses these weights but doesn't change them
  - Weights are renormalized to sum to ~1.0 after adjustments
- **Result**: Over time, repeated feedback and outcomes make the generator more likely to produce passing questions on the first attempt, using prompts optimized for the current weight configuration.

## Frontend User Experience

- Select storybook(s) and an objective
- Generate moral and segments, then approve
- Review the generated questions and their summaries
- Provide question-level good/bad and overall set feedback
- Click Regenerate to apply the orchestrator’s action and produce a refined set

## File Structure

```
teacher_interface/
├── backend/
│   ├── evaluators/
│   │   ├── base_evaluator.py          # Abstract base class for all evaluators
│   │   ├── registry.py                 # Dual registry (core + dynamic)
│   │   ├── manager.py                  # EvaluatorManager - weight management
│   │   ├── suitability_evaluators.py   # Core evaluators + dynamic template
│   │   └── dynamic_template.py        # Dynamic evaluator creation helpers
│   ├── models/
│   │   └── schemas.py                  # Pydantic models for validation
│   ├── services/
│   │   ├── question_generator.py      # Question generation + DSPy optimization
│   │   ├── contextq_evaluators.py     # Evaluation pipeline
│   │   └── teacher_feedback_system.py # Feedback processing (orchestrator/router/manager)
│   ├── data/
│   │   ├── evaluators.json             # Core evaluator metadata
│   │   └── dynamic_evaluators.json     # Dynamic evaluator metadata
│   ├── storage/
│   │   ├── question_evaluations.json   # Question evaluations with regeneration history
│   │   └── teacher_feedback_records.json # Feedback records with orchestrator decisions
│   └── server.py                       # Flask API server
└── frontend/
    └── src/                            # React TypeScript frontend
```

## Notes on Configuration and Paths

- Environment variables live in `teacher_interface/.env` (template in `.env.example`).
- `ASSETS_PATH` points to local storybook assets and must include `qna_json/` and `image/` subfolders.
- Output paths are configurable; for the primary flow, JSON files are stored in `teacher_interface/backend/storage/`.
- Evaluator metadata stored in `backend/data/` (separate from evaluation results in `backend/storage/`).

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

## Key Design Decisions

### 1. Modular Evaluator System
- **Base Class**: `BaseEvaluator` ensures unified interface across all evaluators
- **Registry Pattern**: Dynamic discovery without hardcoding evaluator names
- **JSON Metadata**: Separates configuration (weights, descriptions) from implementation
- **Dual Registry**: Core (built-in) vs Dynamic (runtime-created) evaluators

### 2. Orchestrator → Manager Flow
- **Orchestrator**: Semantic interpretation (LLM-based, handles ambiguity)
- **Router**: Message routing (decouples interpretation from action)
- **Manager**: Centralized state management (weights, evaluators, persistence)
- **Separation**: Clear boundaries - orchestrator interprets, manager executes

### 3. Weight Management
- **Manager Controls Weights**: Orchestrator feedback → Manager adjusts weights
- **Optimizer Uses Weights**: Optimizer uses manager-adjusted weights in metric
- **No Weight Tuning**: `BootstrapFewShot` optimizes prompts, not weights
- **Renormalization**: Weights always sum to ~1.0 after adjustments

### 4. Dynamic Evaluator Creation
- **Template-Based**: Uses `make_dynamic_evaluator_class` to create evaluator classes
- **LLM-Backed**: Dynamic evaluators use LLM for evaluation (not rule-based)
- **Persistent**: Stored in `dynamic_evaluators.json` and registered in runtime
- **First-Class**: Dynamic evaluators treated same as core evaluators

### 5. Regeneration History
- **Per-Question Mapping**: Each regenerated question shows 1:1 mapping to failed question
- **Complete Metadata**: Includes original question, failure reason, avoidance instructions
- **Performance Tracking**: Batch-level performance metrics in metadata
- **No Duplication**: Failed questions shown once per regenerated question

## Glossary

- **ContextQ**: The pedagogical rubric used to score question suitability by type
- **Dynamic Evaluator**: A new, feedback-driven criterion added to future evaluations (created at runtime)
- **Orchestrator**: LLM-based component that interprets teacher feedback into actionable changes
- **EvaluatorManager**: Centralized service for managing evaluator instances, weights, and dynamic registration
- **SuitabilityProgram**: DSPy module that aggregates evaluator scores using manager-adjusted weights
- **DSPy**: Framework used for programmatic prompting and few-shot optimization
- **BootstrapFewShot**: DSPy teleprompter that optimizes prompts using few-shot examples

---

This README focuses on the conceptual flow. Refer to `ARCHITECTURE.md` for a deeper module breakdown and `SETUP.md` to install and run the project.
