# Teacher Interface Architecture

This document describes the architecture, modules, data flow, and operational details of the `teacher_interface` application. It is written to onboard developers and serve as a reference for future maintenance and extension.

## Goals
- Generate high-quality, age-appropriate questions from storybooks aligned with a teacher’s objective
- Automatically evaluate and regenerate questions using a pedagogical rubric (ContextQ)
- Incorporate teacher-in-the-loop feedback to improve future generations
- Support dynamic evaluation criteria created from feedback

## Repository Structure (teacher_interface)

```
teacher_interface/
  backend/
    server.py
    storage/
      question_evaluations.json
      teacher_feedback_records.json
    services/
      __init__.py
      contextq_evaluators.py
      gpt-moral-generation-structured.py
      gpt-objective-questions-structured-DSpy.py
      optimized_question_pipeline.py
      question_generator.py
      teacher_feedback_system.py
  firebase/
    __init__.py
    firebase_client.py
    manifest.json
  frontend/
    package.json
    src/ ...
  __init__.py
  config.py
  requirements.txt
  q_testing.json
```

Notes:
- JSON files under `backend/storage/` are the application’s local data stores (suitable for dev/testing and CI). In production, back a real database/cloud store.
- The `firebase/` folder integrates with a remote MIT proxy service using JWT; it does not use the Firebase Admin SDK directly by default.


## Backend

### Flask Server (`backend/server.py`)
The Flask application provides API endpoints to the frontend and orchestrates the end-to-end pipeline.

Key responsibilities:
- Load storybook content from `ASSETS_PATH` (see `.env`)
- Generate morals and segments (calls `gpt-moral-generation-structured.py`)
- Generate questions (calls `question_generator.py` / `gpt-objective-questions-structured-DSpy.py`)
- Evaluate and regenerate questions until a target set of passing questions is reached
- Store evaluation results into `backend/storage/question_evaluations.json`
- Record teacher feedback via `/api/teacher-feedback` into `backend/storage/teacher_feedback_records.json`
- Regenerate questions with updates from teacher feedback, including dynamic evaluators

Important endpoints:
- `GET /api/storybooks`: List available storybooks from local assets
- `GET /api/image/<storybook_id>`: Serve story cover image
- `POST /api/generate-moral`: Generate moral + segments and questions for selected stories/objective
- `POST /api/teacher-feedback`: Record teacher feedback; orchestrate action (reinforce/adjust/add evaluator)
- `POST /api/regenerate-questions`: Generate new questions using updated action/evaluators
- `GET /api/firebase/*`: Proxy calls to the MIT server via `firebase_client`

### Services

#### `services/gpt-moral-generation-structured.py`
- Uses OpenAI Responses API to produce a structured moral and segmented story representation.
- Loads prompt from `prompts/moralQ_prompts/story_moral_prompt.txt`. If not available, a default prompt is used.
- Outputs a `moral` string and a list of segments `{START, END, SUMMARY, REASONING}`.

#### `services/question_generator.py`
- Encapsulates generation logic using DSPy:
  - Signatures for question generation
  - Optional optimization with DSPy’s BootstrapFewShot
- Uses prior evaluation/feedback to build training examples and improve question quality.
- Emits batches of questions per objective/story with variation prompts.

#### `services/contextq_evaluators.py`
- Implements the ContextQ Suitability Evaluation Pipeline.
- Responsibilities:
  - Classify question type (Completion, Recall, Open-Ended, Wh, Distancing)
  - Route each question to type-specific suitability agents
  - Apply dynamic evaluators created from teacher feedback (e.g., complexity)
  - Decide pass/regenerate based on type-specific thresholds and dynamic evaluator outcomes
- Persist evaluation results to `backend/storage/question_evaluations.json` with set metadata

#### `services/teacher_feedback_system.py`
- The teacher-in-the-loop engine: collects, interprets, and stores feedback for future runs.
- Components:
  - FeedbackOrchestrator (DSPy-based): interprets feedback text into structured actions
  - Action decision: `reinforce_existing`, `adjust_evaluator`, or `add_new_evaluator`
  - Weight updates: adjusts type weights (e.g., `recall_suitability`) or reinforces all weights
  - Creates dynamic evaluators when a new attribute is identified
- Persists records to `backend/storage/teacher_feedback_records.json` in a school/teacher hierarchy.

#### `services/gpt-objective-questions-structured-DSpy.py`
- Alternative/companion generator for objective-based questions with DSPy.
- Supports batch mode for offline experiments and dataset creation.

#### `services/optimized_question_pipeline.py`
- Optional pipeline scaffolding for advanced optimization strategies (few-shot, heuristics, selection). Not required for the primary flow but available for experiments.

### Evaluation Storage

#### `backend/storage/question_evaluations.json`
- Append-only log of all evaluated questions with decisions and metadata:
  - `set_metadata`: `{storybook_id, objective, set_number, timestamp}`
  - Records per question: `{question, question_type, type_confidence, suitability_score, decision, evaluation_reasoning, dynamic_evaluations, regenerated, regeneration_history}`

#### `backend/storage/teacher_feedback_records.json`
- Structured records of feedback sessions and orchestration results
  - Top-level hierarchy: `{school_id: {teacher_id: [records...]}}`
  - Each record includes: story title, objective, teacher_feedback, interpretation, `action_taken`, `course_of_action`, evaluator scores, question-level feedback, and timestamps

### Firebase Integration (`firebase/firebase_client.py`)
- Authenticates against a remote MIT proxy via JWT using `SERVER_URL`, `USERNAME`, and `PASSWORD` from `.env`
- Exposes helper methods to fetch students, storybooks, and node data
- Not using Firebase Admin SDK by default; can be extended to direct Firestore/RTDB access with a service account

## Frontend

The React/TypeScript frontend provides the teacher UI for:
- Selecting storybooks and objectives
- Generating and reviewing questions
- Providing per-question (good/bad) and overall feedback
- Triggering regeneration using orchestrated actions

Key files:
- `frontend/package.json`: Dev server and build scripts
- `frontend/src/App.tsx`: Main shell
- `frontend/src/components/*`: Views (e.g., moral approval, results display) and controls
- `frontend/src/services/api.ts`: Axios client for `/api/*` endpoints
- `frontend/src/types/*`: Shared types between UI views and API DTOs

## Configuration & Environment

- `.env` (placed under `teacher_interface/.env`):
  - `OPENAI_API_KEY`: OpenAI API key
  - `SERVER_URL`, `USERNAME`, `PASSWORD`: Firebase MIT proxy credentials
  - `ASSETS_PATH`: Base path to storybook assets (expects subfolders `qna_json/` and `image/`)
  - Optional:
    - `OUTPUT_PATH`: Output folder for moral segments (if running batch scripts)
    - `OBJECTIVE_QUESTIONS_OUTPUT_PATH`: Output folder for objective question batches

- `requirements.txt` (under `teacher_interface/`): All backend/service dependencies.

## End-to-End Flow (Detailed)

1. The frontend calls `POST /api/generate-moral` with selected storybooks and an objective.
2. The backend loads story text from `ASSETS_PATH/qna_json/<storybook_id>`, invokes the moral generator to produce `moral` and `segments`.
3. The backend calls question generation to produce a diverse batch of candidate questions.
4. Each question is evaluated by ContextQ and any dynamic evaluators. Questions that fail are regenerated until a set of passing questions is collected or attempts are exhausted.
5. Results are displayed in the frontend; evaluation details are stored in `backend/storage/question_evaluations.json`.
6. The teacher reviews the set, marks questions good/bad and provides overall feedback.
7. `POST /api/teacher-feedback` records the feedback. The orchestrator interprets it into one of three actions:
   - Reinforce: Slightly stabilizes current evaluator weights
   - Adjust: Applies deltas to specific suitability weights and renormalizes
   - Add evaluator: Creates a new dynamic evaluator with criteria derived from reformulated instruction
8. `POST /api/regenerate-questions` generates a new set using the updated configuration and applies the new evaluator(s).


---


