"""
Teacher Interface Flask Application
Serves the storybook library dashboard for educators
"""

from flask import Flask, render_template, jsonify, send_file, request, send_from_directory
from flask_cors import CORS
import os
import json
import subprocess
import sys
from pathlib import Path

# Path to moral generation script
MORAL_GEN_SCRIPT = os.path.join(os.path.dirname(__file__), 'services', 'gpt-moral-generation-structured.py')
from datetime import datetime
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from firebase.firebase_client import FirebaseClientAuth
from dotenv import load_dotenv

app = Flask(__name__, static_folder='../frontend/build/static', template_folder='../frontend/build')
CORS(app)  # Enable CORS for all routes

# Load environment variables from teacher_interface/.env
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

# Directory for persisted feedback/evaluation artifacts
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

# Configuration
# Read assets path from .env; fallback to previous default if not set
ASSETS_PATH = os.getenv('ASSETS_PATH', "/Users/mariyamohiuddin/Desktop/interactive-storybook-assets")
QNA_JSON_PATH = os.path.join(ASSETS_PATH, "qna_json")
IMAGES_PATH = os.path.join(ASSETS_PATH, "image")

# Firebase client initialization
firebase_client = None
try:
    server_url = os.getenv('SERVER_URL')
    username = os.getenv('USERNAME')
    password = os.getenv('PASSWORD')
    
    if all([server_url, username, password]):
        firebase_client = FirebaseClientAuth(server_url, username, password)
        if firebase_client.authenticate():
            print("Firebase client authenticated successfully")
        else:
            print("Firebase client authentication failed")
            firebase_client = None
    else:
        print("Firebase credentials not found in environment variables")
except Exception as e:
    print(f"Firebase client initialization error: {e}")
    firebase_client = None

def get_storybook_data():
    """Load storybook data from the assets directory"""
    storybooks = []
    
    if not os.path.exists(QNA_JSON_PATH):
        return storybooks
    
    try:
        for story_folder in os.listdir(QNA_JSON_PATH):
            story_path = os.path.join(QNA_JSON_PATH, story_folder)
            if os.path.isdir(story_path):
                # Count JSON files (pages) in the folder
                json_files = [f for f in os.listdir(story_path) if f.endswith('.json')]
                page_count = len(json_files)
                
                # Get cover image path (structure: story_folder/story_folder_00/Background.png)
                cover_image_path = os.path.join(IMAGES_PATH, story_folder, f"{story_folder}_00", "Background.png")
                cover_image = f"/api/image/{story_folder}" if os.path.exists(cover_image_path) else None
                
                # Create storybook data
                storybook = {
                    "id": story_folder,
                    "title": story_folder.replace("_", " ").title(),
                    "pageCount": page_count,
                    "ageRange": "4-6",
                    "tags": ["education", "children", "storybook"],
                    "coverImage": cover_image
                }
                storybooks.append(storybook)
                
    except Exception as e:
        print(f"Error loading storybooks: {e}")
    
    return sorted(storybooks, key=lambda x: x["title"])

@app.route("/")
def index():
    """Serve the React app"""
    return send_from_directory('../frontend/build', 'index.html')

@app.route("/api/storybooks")
def api_storybooks():
    """API endpoint to get all storybooks"""
    storybooks = get_storybook_data()
    return jsonify({
        "storybooks": storybooks,
        "count": len(storybooks)
    })

@app.route("/api/storybook/<storybook_id>")
def api_storybook(storybook_id):
    """API endpoint to get a specific storybook"""
    storybooks = get_storybook_data()
    storybook = next((sb for sb in storybooks if sb["id"] == storybook_id), None)
    
    if storybook:
        return jsonify(storybook)
    else:
        return jsonify({"error": "Storybook not found"}), 404

@app.route("/api/image/<storybook_id>")
def api_image(storybook_id):
    """API endpoint to serve storybook cover images"""
    image_path = os.path.join(IMAGES_PATH, storybook_id, f"{storybook_id}_00", "Background.png")
    
    if os.path.exists(image_path):
        return send_file(image_path)
    else:
        return jsonify({"error": "Image not found"}), 404

# Firebase Data Endpoints
@app.route("/api/firebase/status")
def api_firebase_status():
    """Check Firebase connection status"""
    if firebase_client:
        return jsonify({
            "connected": True,
            "server_url": firebase_client.server_url,
            "authenticated": firebase_client.jwt_token is not None
        })
    else:
        return jsonify({
            "connected": False,
            "error": "Firebase client not initialized"
        })

@app.route("/api/firebase/students")
def api_firebase_students():
    """Get student data from Firebase"""
    if not firebase_client:
        return jsonify({"error": "Firebase client not available"}), 500
    
    try:
        students = firebase_client.make_authenticated_request("/get_students")
        if students:
            return jsonify({"students": students})
        else:
            return jsonify({"error": "Failed to fetch students"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/firebase/storybooks")
def api_firebase_storybooks():
    """Get storybook data from Firebase"""
    if not firebase_client:
        return jsonify({"error": "Firebase client not available"}), 500
    
    try:
        storybooks = firebase_client.make_authenticated_request("/get_storybooks")
        if storybooks:
            return jsonify({"storybooks": storybooks})
        else:
            return jsonify({"error": "Failed to fetch storybooks"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/firebase/top-level-node")
def api_firebase_top_level_node():
    """Get data from top-level node"""
    if not firebase_client:
        return jsonify({"error": "Firebase client not available"}), 500
    
    try:
        top_level_node = os.getenv('TOP_LEVEL_NODE')
        if not top_level_node:
            return jsonify({"error": "Top-level node not configured"}), 500
        
        params = {
            "subject_id": top_level_node,
            "path": "/"
        }
        
        data = firebase_client.make_authenticated_request("/get_nodes", params=params)
        if data:
            return jsonify({"data": data, "node": top_level_node})
        else:
            return jsonify({"error": "Failed to fetch top-level node data"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/generate-moral", methods=["POST"])
def api_generate_moral():
    """API endpoint to generate moral lesson and questions directly using structured script"""
    try:
        data = request.get_json()
        selected_storybooks = data.get('selectedStorybooks', [])
        objective = data.get('objective', 'moral')
        
        if not selected_storybooks:
            return jsonify({"error": "No storybooks selected"}), 400
        
        # Generate moral and questions for each selected storybook
        results = []
        for storybook in selected_storybooks:
            storybook_id = storybook['id']
            story_title = storybook['title']
            
            
            # Get story content from JSON files
            story_content = get_story_content(storybook_id)
            if not story_content:
                print(f"[WARN]  No story content found for {storybook_id}")
                continue
            
            # Use structured moral generation script directly
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "gpt-moral-generation-structured",
                    MORAL_GEN_SCRIPT
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                StoryMoralGeneratorStructured = module.StoryMoralGeneratorStructured
                
                # Initialize structured generator
                moral_generator = StoryMoralGeneratorStructured()
                
                # Generate moral and segments
                moral_result = moral_generator.generate_story_moral(story_content)
                
                if moral_result:
                    moral_text = moral_result.moral
                    segments_data = [
                        {
                            "name": seg.name,
                            "start": seg.START,
                            "end": seg.END,
                            "summary": seg.SUMMARY,
                            "reasoning": seg.REASONING
                        }
                        for seg in moral_result.segments
                    ]
                    
                    # Generate questions using the moral and segments as context
                    questions_data = call_objective_question_generation_script(
                        story_content, segments_data, objective, story_title, moral_text
                    )
                    
                    print(f"[INFO] Generated questions using structured moral: {moral_text[:50]}...")
                    print(f"   Segments: {len(segments_data)}")
                    
                    
                    if questions_data and questions_data.get("questions"):
                        questions = questions_data.get("questions", [])
                        print(f"[INFO] Retrieved {len(questions)} questions from generation script")
                        
                        # Evaluation and storage is handled internally in call_objective_question_generation_script
                        # Format result for frontend
                        result = {
                            "storybook": storybook,
                            "objective": objective,
                            "moral": {
                                "generated": moral_text,
                                "status": "approved",
                                "feedback": "positive",
                                "timestamp": datetime.now().isoformat()
                            },
                            "segments": segments_data,
                            "questions": questions,
                            "learning_objectives": questions_data.get("learning_objectives", [])
                        }
                        results.append(result)
                    else:
                        print(f"[WARN]  Failed to generate questions for {story_title}")
                else:
                    print(f"[WARN]  Structured moral generation failed for {story_title}")
                    
            except Exception as e:
                print(f"[WARN]  Error generating moral/questions for {story_title}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return jsonify({
            "results": results,
            "stage": "results",
            "success": True
        })
        
    except Exception as e:
        print(f"Error in generate-moral endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/generate-questions", methods=["POST"])
def api_generate_questions():
    """API endpoint to generate questions using DSPy"""
    try:
        data = request.get_json()
        selected_storybooks = data.get('selectedStorybooks', [])
        objective = data.get('objective', 'moral')
        question_frequency = data.get('questionFrequency', 'per-segment')
        
        # Only support segment-based questions for now
        if question_frequency != 'per-segment':
            return jsonify({
                "error": "Only 'per-segment' frequency is currently supported"
            }), 400
        
        if not selected_storybooks:
            return jsonify({"error": "No storybooks selected"}), 400
        
        # Generate questions for each selected storybook
        results = []
        for storybook in selected_storybooks:
            storybook_id = storybook['id']
            story_title = storybook['title']
            
            # Get story content from JSON files
            story_content = get_story_content(storybook_id)
            if not story_content:
                continue
            
            # First, get moral and segments using DSPy script
            moral_data = call_dspy_script(story_content)
            if not moral_data:
                continue
            
            moral = moral_data.get("moral", "")
            segments = moral_data.get("segments", [])
            
            # Then generate objective-based questions using the DSPy script
            questions_data = call_objective_question_generation_script(story_content, segments, objective, story_title)
            if questions_data:
                results.append({
                    "storybook": storybook,
                    "moral": moral,
                    "segments": segments,
                    "questions": questions_data.get("questions", []),
                    "learning_objectives": questions_data.get("learning_objectives", []),
                    "objective": objective,
                    "frequency": question_frequency
                })
            else:
                # Fallback if question generation fails
                results.append({
                    "storybook": storybook,
                    "moral": moral,
                    "segments": segments,
                    "questions": [],
                    "learning_objectives": [],
                    "objective": objective,
                    "frequency": question_frequency
                })
        
        return jsonify({
            "success": True,
            "results": results
        })
        
    except Exception as e:
        print(f"Error generating questions: {e}")
        return jsonify({"error": str(e)}), 500

def get_story_content(storybook_id):
    """Get story content from JSON files"""
    story_path = os.path.join(QNA_JSON_PATH, storybook_id)
    if not os.path.exists(story_path):
        print(f"Story path not found: {story_path}")
        return None
    
    story_content = ""
    json_files = sorted([f for f in os.listdir(story_path) if f.endswith('.json')])
    
    for json_file in json_files:
        file_path = os.path.join(story_path, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                page_data = json.load(f)
                # Extract text content (text is an array in this JSON structure)
                if 'text' in page_data:
                    if isinstance(page_data['text'], list):
                        # Join array elements with spaces
                        story_content += " ".join(page_data['text']) + "\n"
                    else:
                        story_content += str(page_data['text']) + "\n"
                elif 'content' in page_data:
                    if isinstance(page_data['content'], list):
                        story_content += " ".join(page_data['content']) + "\n"
                    else:
                        story_content += str(page_data['content']) + "\n"
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
    
    return story_content.strip() if story_content else None

def call_dspy_script(story_content):
    """Call the DSPy moral generation script"""
    try:
        # Path to the DSPy script
        dspy_script = os.path.join(os.path.dirname(__file__), "services", "gpt-moral-generation-DSpy.py")
        
        if not os.path.exists(dspy_script):
            print(f"DSPy script not found at: {dspy_script}")
            return None
        
        # Run the DSPy script with the story content
        result = subprocess.run([
            sys.executable, dspy_script, "--story", story_content
        ], capture_output=True, text=True, cwd=os.path.dirname(dspy_script))
        
        if result.returncode == 0:
            # Parse the JSON output
            return json.loads(result.stdout)
        else:
            print(f"DSPy script error: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Error calling DSPy script: {e}")
        return None

def get_main_system():
    """Get or create the main question generation system instance."""
    global _question_system
    if '_question_system' not in globals():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "question_generator",
            os.path.join(os.path.dirname(__file__), "services", "question_generator.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        feedback_path = os.path.join(os.path.dirname(__file__), "feedback_dataset.json")
        _question_system = module.TeacherInterfaceQuestionGenerator(feedback_path)
    
    return _question_system

def call_optimized_moral_script(story_content, story_title):
    """Generate moral candidates using the centralized system."""
    try:
        system = get_main_system()
        result = system.generate_moral_candidates(story_content, story_title)
        
        if result["success"] and result["candidates"]:
            # Return the best candidate (first one after sorting by quality)
            best_candidate = result["candidates"][0]
            return {
                "moral": best_candidate["moral"],
                "segments": best_candidate["segments"],
                "candidates": result["candidates"],  # Include all candidates for frontend
                "optimization_applied": result["optimization_applied"]
            }
        else:
            print(f"Error in optimized moral generation: {result.get('error', 'Unknown error')}")
            return None
        
    except Exception as e:
        print(f"Error calling moral generation: {e}")
        import traceback
        traceback.print_exc()
        return None

# Removed unused moral feedback placeholder

def log_to_testing_file(story_title, objective, set_number, questions, feedback=None):
    """Log questions and feedback to q_testing.json for testing and comparison."""
    try:
        import json
        testing_file = os.path.join(os.path.dirname(__file__), "..", "q_testing.json")
        
        # Load existing data
        if os.path.exists(testing_file):
            with open(testing_file, 'r') as f:
                data = json.load(f)
        else:
            data = {"test_logs": []}
        
        # Find or create entry for this story
        story_entry = None
        for entry in data["test_logs"]:
            if entry.get("story_title") == story_title and entry.get("objective") == objective:
                story_entry = entry
                break
        
        if not story_entry:
            story_entry = {
                "story_title": story_title,
                "objective": objective,
                "sets": []
            }
            data["test_logs"].append(story_entry)
        
        # Extract question texts
        question_texts = []
        for q in questions:
            if isinstance(q, dict):
                question_texts.append(q.get("question", ""))
            else:
                question_texts.append(str(q))
        
        # Add or update set entry
        set_entry = {
            "set_number": set_number,
            "questions": question_texts,
            "feedback": feedback or "",
            "timestamp": datetime.now().isoformat()
        }
        
        # Remove old set entry if it exists
        story_entry["sets"] = [s for s in story_entry["sets"] if s.get("set_number") != set_number]
        story_entry["sets"].append(set_entry)
        
        # Save to file
        with open(testing_file, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Logged {len(question_texts)} questions for {story_title} - {set_number}")
        
    except Exception as e:
        print(f"[WARN]  Error logging to testing file: {e}")

def call_objective_question_generation_script(story_content, segments, objective, story_title, moral_text=None, dynamic_evaluators=None):
    """Generate questions with automatic suitability checking and regeneration."""
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))
        
        # Import the new evaluation-based generator
        from contextq_evaluators import ContextQEvaluationPipeline
        from question_generator import QuestionGeneratorModule
        
        # Load dynamic evaluators from teacher feedback if not provided
        if dynamic_evaluators is None:
            dynamic_evaluators = []
            try:
                import json
                feedback_file = os.path.join(STORAGE_DIR, "teacher_feedback_records.json")
                if os.path.exists(feedback_file):
                    with open(feedback_file, 'r') as f:
                        feedback_data = json.load(f)
                        # Extract dynamic evaluators from feedback records
                        # Records are nested under school -> teacher (check all combinations)
                        records = []
                        for school_key in feedback_data:
                            if school_key == "records":
                                records.extend(feedback_data.get("records", []))
                            elif isinstance(feedback_data[school_key], dict):
                                for teacher_key in feedback_data[school_key]:
                                    if isinstance(feedback_data[school_key][teacher_key], list):
                                        records.extend(feedback_data[school_key][teacher_key])
                        
                        
                        for record in records:
                            details = record.get("action_taken", {}).get("details", {})
                            if details.get("status") == "active":
                                evaluator_name = details.get("evaluator_name", "unknown")
                                # Prefer explicit instruction text for clear criteria; fall back to description
                                evaluator_criteria = (
                                    record.get("course_of_action", {}).get("instruction")
                                    or record.get("action_taken", {}).get("reformulated_instruction")
                                    or record.get("interpretation", {}).get("reformulated_instruction")
                                    or details.get("evaluator_description", "")
                                )
                                evaluator_dict = {
                                    "name": evaluator_name,
                                    "criteria": evaluator_criteria
                                }
                                dynamic_evaluators.append(evaluator_dict)
                            else:
                                print(f"[WARN]  DEBUG: Skipping record with status: {details.get('status')}")
                        
                        if dynamic_evaluators:
                            print(f"[INFO] Loaded {len(dynamic_evaluators)} dynamic evaluators in call_objective_question_generation_script")
            except Exception as e:
                print(f"[WARN]  Error loading dynamic evaluators: {e}")
        
        # Initialize evaluator pipeline (handles generation + evaluation + regeneration)
        if dynamic_evaluators:
            for ev in dynamic_evaluators:
                print(f"   Evaluator: {ev}")
        
        evaluator = ContextQEvaluationPipeline(
            storage_file=os.path.join(STORAGE_DIR, "question_evaluations.json"),
            feedback_records_file=None,
            dynamic_evaluators=dynamic_evaluators if dynamic_evaluators else None
        )
        
        # Initialize question generator module for the actual question generation
        question_generator = QuestionGeneratorModule()
        
        # Load the most recent feedback's reformulated instruction
        reformulated_instruction = ""
        try:
            import json
            feedback_file = os.path.join(STORAGE_DIR, "teacher_feedback_records.json")
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    feedback_data = json.load(f)
                    records = feedback_data.get("records", [])
                    if records:
                        # Get the most recent feedback
                        latest_record = records[-1]
                        course_of_action = latest_record.get("course_of_action", {})
                        reformulated_instruction = course_of_action.get("reformulated_instruction", "")
                        if reformulated_instruction:
                            pass
        except Exception as e:
            print(f"[WARN]  Error loading feedback instruction: {e}")
        
        print(f"\n[INFO] Generating questions for: {story_title}")
        print(f"Objective: {objective}")
        
        # Track set number for metadata - load existing sets for this storybook
        set_number = "set_001"  # Default for initial generation
        try:
            import json
            eval_file = os.path.join(STORAGE_DIR, "question_evaluations.json")
            if os.path.exists(eval_file):
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                    # Find the latest set_number for this storybook and objective
                    evaluations = eval_data.get("evaluations", [])
                    max_set_num = 0
                    for item in evaluations:
                        set_meta = item.get("set_metadata", {})
                        if (set_meta.get("storybook_id") == story_title and 
                            set_meta.get("objective") == objective):
                            set_num_str = set_meta.get("set_number", "set_001")
                            # Extract number from "set_XXX" format
                            try:
                                set_num = int(set_num_str.split("_")[1])
                                max_set_num = max(max_set_num, set_num)
                            except:
                                pass
                    
                    if max_set_num > 0:
                        # Use next set number
                        set_number = f"set_{max_set_num + 1:03d}"
                        print(f"[INFO] Found {max_set_num} existing set(s), using {set_number}")
                    else:
                        print(f"[INFO] No existing sets found, using {set_number}")
        except Exception as e:
            print(f"[WARN]  Error determining set number: {e}")
        
        # Helper functions for regeneration guidance
        def _collect_failure_reasons(eval_results, questions):
            failure_notes = []
            for eval_result, q_text in zip(eval_results, questions):
                if not eval_result or eval_result.get("decision") == "pass":
                    continue
                question_type = eval_result.get("question_type", "")
                base_reason = (eval_result.get("evaluation_reasoning") or "").strip()
                if base_reason:
                    label = f"{question_type}: {base_reason}" if question_type else base_reason
                    failure_notes.append(label)
                dynamic_evals = eval_result.get("dynamic_evaluations", {}) or {}
                for dyn_name, dyn_info in dynamic_evals.items():
                    if dyn_info.get("decision") != "regenerate":
                        continue
                    dyn_reason = (dyn_info.get("reasoning") or "").strip()
                    score = dyn_info.get("score")
                    readable_name = dyn_name.replace("_", " ")
                    if dyn_reason and score is not None:
                        failure_notes.append(f"{readable_name}: {dyn_reason} (score={score})")
                    elif dyn_reason:
                        failure_notes.append(f"{readable_name}: {dyn_reason}")
                    elif score is not None:
                        failure_notes.append(f"{readable_name}: score={score} below threshold")
            return failure_notes

        def _build_failure_guidance(reasons_set):
            if not reasons_set:
                return ""
            top_reasons = list(reasons_set)[:5]
            joined = "; ".join(top_reasons)
            return f"Avoid previous issues: {joined}"

        def _compose_avoidance_instruction(base_instruction, guidance_text):
            default_prompt = "Generate at least 12 diverse, unique questions that avoid previous issues and vary types/difficulties."
            instructions = []
            if base_instruction:
                instructions.append(base_instruction.strip())
            if guidance_text:
                instructions.append(guidance_text)
            instructions.append(default_prompt)
            return " ".join(instructions).strip()

        failure_reason_set = set()

        # Step 1: Generate an initial batch of 15 questions for diversity (to ensure we get 10 passing)
        result = question_generator.generate(
            story=story_content,
            segments=segments,
            objective=objective,
            story_title=story_title,
            moral_text=moral_text,
            avoidance_instructions=reformulated_instruction
        )
        
        initial_questions = result.get("questions", [])
        print(f"[INFO] Generated {len(initial_questions)} questions in initial batch")
        
        if not initial_questions:
            print("[WARN]  No questions generated")
            suitable_questions = []
        else:
            # Step 2: Evaluate all questions in the batch
            question_texts = [q.get("question", "") if isinstance(q, dict) else str(q) for q in initial_questions]
            
            # Evaluate all questions but don't store to JSON yet
            evaluations_list = []
            for q_text in question_texts:
                eval_result = evaluator.suitability_program(question=q_text, story_context=story_content)
                evaluations_list.append(eval_result)
            
            # Log evaluation summary
            passing_count = sum(1 for e in evaluations_list if e.get("decision") == "pass")
            regenerate_count = sum(1 for e in evaluations_list if e.get("decision") == "regenerate")
            print(f"[INFO] Evaluation Summary: {passing_count} passing, {regenerate_count} need regeneration out of {len(evaluations_list)} total")
            
            # Step 3: Filter for passing questions with uniqueness tracking
            passing_questions = []
            unique_questions = set()  # Track unique question texts

            failure_reason_set.update(_collect_failure_reasons(evaluations_list, question_texts))
            
            for i, eval_result in enumerate(evaluations_list):
                decision = eval_result.get("decision")
                question_text = question_texts[i]
                
                if decision == "pass" and question_text not in unique_questions:
                    # Add to unique set
                    unique_questions.add(question_text)
                    
                    # Create question object with evaluation metadata
                    q_obj = initial_questions[i] if i < len(initial_questions) else {"question": question_text}
                    if not isinstance(q_obj, dict):
                        q_obj = {"question": question_text}
                    
                    q_obj["suitability_evaluation"] = {
                        "decision": eval_result.get("decision"),
                        "suitability_score": eval_result.get("suitability_score"),
                        "question_type": eval_result.get("question_type"),
                        "evaluation_reasoning": eval_result.get("evaluation_reasoning"),
                        "type_confidence": eval_result.get("type_confidence"),
                        "type_reasoning": eval_result.get("type_reasoning"),
                        "regenerated": False,  # Mark as initial generation
                        "regeneration_history": [],  # Will track failed attempts
                        "dynamic_evaluations": eval_result.get("dynamic_evaluations", {})  # Include dynamic evaluator results
                    }
                    passing_questions.append(q_obj)
            
            print(f"[INFO] Found {len(passing_questions)} unique passing questions")
            
            # Step 4: If we don't have enough (10), keep regenerating until we have 10
            attempts_remaining = 20  # Increased attempts to improve chances of reaching 10
            regeneration_attempt = 1
            while len(passing_questions) < 10 and attempts_remaining > 0:
                attempts_remaining -= 1
                print(f"[WARN]  Only {len(passing_questions)}/10 passing questions. Generating more... (Attempts remaining: {attempts_remaining})")
                
                avoidance_instruction = _compose_avoidance_instruction(reformulated_instruction, _build_failure_guidance(failure_reason_set))

                # Generate additional questions
                print(f"Generating {10 - len(passing_questions)} more questions...")
                additional_result = question_generator.generate(
                    story=story_content,
                    segments=segments,
                    objective=objective,
                    story_title=story_title,
                    moral_text=moral_text,
                    avoidance_instructions=avoidance_instruction
                )
                
                new_questions = additional_result.get("questions", [])
                if not new_questions:
                    break
                
                # Do not log intermediate regeneration attempts to q_testing.json; only final set is logged
                    
                # Evaluate new questions
                new_evaluations = []
                regeneration_attempt += 1
                failed_questions = []  # Track failed questions for history
                
                for q_text in [q.get("question", "") if isinstance(q, dict) else str(q) for q in new_questions]:
                    if q_text and q_text not in unique_questions:
                        eval_result = evaluator.suitability_program(question=q_text, story_context=story_content)
                        new_evaluations.append((q_text, eval_result))
                        
                        # Track failed questions
                        if eval_result.get("decision") != "pass":
                            failed_questions.append({
                                "question": q_text,
                                "decision": eval_result.get("decision"),
                                "reasoning": eval_result.get("evaluation_reasoning")
                            })
                
                # Log new evaluation summary
                new_passing = sum(1 for (_, e) in new_evaluations if e.get("decision") == "pass")
                new_regenerate = sum(1 for (_, e) in new_evaluations if e.get("decision") == "regenerate")
                print(f"[INFO] New batch: {new_passing} passing, {new_regenerate} need regeneration out of {len(new_evaluations)} total")

                failure_reason_set.update(
                    _collect_failure_reasons(
                        [e for (_, e) in new_evaluations],
                        [q for (q, _) in new_evaluations]
                    )
                )
                
                # Add passing ones
                for q_text, eval_result in new_evaluations:
                    if eval_result.get("decision") == "pass" and len(passing_questions) < 10:
                        unique_questions.add(q_text)
                        q_obj = {"question": q_text}
                        q_obj["suitability_evaluation"] = {
                            "decision": eval_result.get("decision"),
                            "suitability_score": eval_result.get("suitability_score"),
                            "question_type": eval_result.get("question_type"),
                            "evaluation_reasoning": eval_result.get("evaluation_reasoning"),
                            "type_confidence": eval_result.get("type_confidence"),
                            "type_reasoning": eval_result.get("type_reasoning"),
                            "regenerated": True,  # This came from regeneration
                            "regeneration_history": [{
                                "attempt": regeneration_attempt - 1,
                                "failed_questions_in_batch": len(failed_questions),
                                "total_attempts": regeneration_attempt
                            }],
                            "dynamic_evaluations": eval_result.get("dynamic_evaluations", {})  # Include dynamic evaluator results
                        }
                        passing_questions.append(q_obj)
                
                # Continue looping until we have 10 questions or attempts exhausted
                if len(passing_questions) >= 10:
                    print(f"[INFO] Reached target of 10 passing questions!")
                    break
            
            # Take exactly 10 unique passing questions (or as many as we have)
            suitable_questions = passing_questions[:10]
            
            # Now store ONLY the final questions to JSON with regeneration history
            print(f"[INFO] Storing {len(suitable_questions)} final questions to JSON...")
            for q in suitable_questions:
                q_text = q.get("question", "")
                if q_text:
                    suitability_eval = q.get("suitability_evaluation", {})
                    # Create proper eval_result dict for storage
                    eval_result = {
                        "decision": suitability_eval.get("decision", "pass"),
                        "suitability_score": suitability_eval.get("suitability_score", 1.0),
                        "question_type": suitability_eval.get("question_type", ""),
                        "evaluation_reasoning": suitability_eval.get("evaluation_reasoning", ""),
                        "type_confidence": suitability_eval.get("type_confidence", ""),
                        "type_reasoning": suitability_eval.get("type_reasoning", ""),
                        "regenerated": suitability_eval.get("regenerated", False),
                        "regeneration_history": suitability_eval.get("regeneration_history", []),
                        "dynamic_evaluations": suitability_eval.get("dynamic_evaluations", {})  # Include dynamic evaluator results
                    }
                    # Store with metadata: storybook_id, objective, and set_number
                    evaluator._store_evaluation_record(q_text, eval_result, story_content, 
                                                      storybook_id=story_title, 
                                                      objective=objective,
                                                      set_number=set_number)
        
        print(f"[INFO] Final set: {len(suitable_questions)} suitable questions")
        
        # Extract learning objectives from the generation (if needed)
        # For now, return a default list
        learning_objectives = [
            "Comprehension of story events",
            "Emotional understanding",
            "Personal connection to story"
        ]
        
        if suitable_questions:
            print(f"[INFO] Generated {len(suitable_questions)} suitable questions")
            
            # Log to testing file
            log_to_testing_file(story_title, objective, set_number, suitable_questions, feedback=None)
            
            return {
                "questions": suitable_questions,
                "learning_objectives": learning_objectives
            }
        else:
            print("Failed to generate suitable questions")
            return None
            
    except Exception as e:
        print(f"Error generating questions with suitability checks: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """API endpoint to record teacher feedback"""
    try:
        data = request.get_json()
        feedback_type = data.get('type', 'positive')
        inputs = data.get('inputs', {})
        outputs = data.get('outputs', {})
        reason = data.get('reason', '')
        
        # Import the objective question generator class
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "objective_questions", 
            os.path.join(os.path.dirname(__file__), "services/gpt-objective-questions-structured-DSpy.py")
        )
        objective_questions_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(objective_questions_module)
        
        ObjectiveQuestionGeneratorDSPy = objective_questions_module.ObjectiveQuestionGeneratorDSPy
        
        # Create generator instance
        generator = ObjectiveQuestionGeneratorDSPy()
        
        # Record feedback
        if feedback_type == 'positive':
            generator.record_positive_feedback(inputs, outputs)
        else:
            generator.record_negative_feedback(inputs, outputs, reason)
        
        return jsonify({"success": True, "message": "Feedback recorded successfully"})
        
    except Exception as e:
        print(f"Error recording feedback: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/teacher-feedback', methods=['POST'])
def api_teacher_feedback():
    """Record teacher feedback for question sets"""
    try:
        data = request.get_json()
        story_title = data.get('storyTitle', '')
        objective = data.get('objective', '')
        teacher_feedback = data.get('teacherFeedback', '')
        feedback_type = data.get('feedbackType', 'negative')  # 'positive' or 'negative'
        question_evaluations = data.get('questionEvaluations', [])
        story_context = data.get('storyContext', '')
        generated_questions = data.get('generatedQuestions', [])
        teacher_id = data.get('teacherId', 'default_teacher')
        school_id = data.get('schoolId', 'default_school')
        
        if not story_title or not teacher_feedback:
            return jsonify({"success": False, "error": "Missing required fields"})
        
        # Import and use the teacher feedback system
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))
        from teacher_feedback_system import TeacherFeedbackSystem
        
        # Initialize feedback system
        feedback_system = TeacherFeedbackSystem(
            storage_file=os.path.join(STORAGE_DIR, "teacher_feedback_records.json")
        )
        
        # Process feedback
        result = feedback_system.process_feedback(
            story_title=story_title,
            objective=objective,
            teacher_feedback=teacher_feedback,
            feedback_type=feedback_type,
            question_evaluations=question_evaluations,
            story_context=story_context,
            generated_questions=generated_questions,
            teacher_id=teacher_id,
            school_id=school_id
        )
        
        return jsonify({
            "success": True, 
            "message": "Teacher feedback processed successfully",
            "action_taken": result["action_taken"],
            "evaluator_weights": result["evaluator_weights"]
        })
        
    except Exception as e:
        print(f"Error processing teacher feedback: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/regenerate-questions", methods=["POST"])
def api_regenerate_questions():
    """API endpoint to regenerate questions using DSPy feedback learning"""
    try:
        data = request.get_json()
        selected_storybooks = data.get('selectedStorybooks', [])
        objective = data.get('objective', 'moral')
        general_feedback = data.get('generalFeedback', '')
        question_feedbacks = data.get('questionFeedbacks', {})
        original_questions = data.get('originalQuestions', [])
        
        if not selected_storybooks:
            return jsonify({"error": "No storybooks selected"}), 400
        
        # Use original questions from frontend instead of regenerating
        print(f"Debug: selected_storybooks count: {len(selected_storybooks)}")
        print(f"Debug: original_questions count: {len(original_questions)}")
        
        # If no original_questions provided, fall back to generating them
        if not original_questions:
            print("Warning: No original_questions provided, generating questions as fallback")
            current_results = []
            for storybook in selected_storybooks:
                storybook_id = storybook['id']
                story_title = storybook['title']
                story_content = get_story_content(storybook_id)
                if story_content:
                    # Use structured moral generation script directly
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "gpt-moral-generation-structured",
                        MORAL_GEN_SCRIPT
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    StoryMoralGeneratorStructured = module.StoryMoralGeneratorStructured
                    
                    # Initialize structured generator
                    moral_generator = StoryMoralGeneratorStructured()
                    
                    # Generate moral and segments
                    moral_result = moral_generator.generate_story_moral(story_content)
                    
                    if moral_result:
                        moral = moral_result.moral
                        segments_data = [
                            {
                                "name": seg.name,
                                "start": seg.START,
                                "end": seg.END,
                                "summary": seg.SUMMARY,
                                "reasoning": seg.REASONING
                            }
                            for seg in moral_result.segments
                        ]
                        questions_data = call_objective_question_generation_script(story_content, segments_data, objective, story_title, moral)
                        if questions_data:
                            current_results.append({
                                "storybook": storybook,
                                "questions": questions_data.get("questions", [])
                            })
        else:
            # Use original questions from frontend
            current_results = []
            for i, storybook in enumerate(selected_storybooks):
                if i < len(original_questions):
                    current_results.append({
                        "storybook": storybook,
                        "questions": original_questions[i]
                    })
                else:
                    print(f"Warning: No original questions for storybook {i} ({storybook['title']})")
                    # Fallback: generate questions if original_questions is missing
                    storybook_id = storybook['id']
                    story_title = storybook['title']
                    story_content = get_story_content(storybook_id)
                    if story_content:
                        # Use structured moral generation script directly
                        import importlib.util
                        spec = importlib.util.spec_from_file_location(
                            "gpt-moral-generation-structured",
                            MORAL_GEN_SCRIPT
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        StoryMoralGeneratorStructured = module.StoryMoralGeneratorStructured
                        
                        # Initialize structured generator
                        moral_generator = StoryMoralGeneratorStructured()
                        
                        # Generate moral and segments
                        moral_result = moral_generator.generate_story_moral(story_content)
                        
                        if moral_result:
                            moral = moral_result.moral
                            segments_data = [
                                {
                                    "name": seg.name,
                                    "start": seg.START,
                                    "end": seg.END,
                                    "summary": seg.SUMMARY,
                                    "reasoning": seg.REASONING
                                }
                                for seg in moral_result.segments
                            ]
                            questions_data = call_objective_question_generation_script(story_content, segments_data, objective, story_title, moral)
                            if questions_data:
                                current_results.append({
                                    "storybook": storybook,
                                    "questions": questions_data.get("questions", [])
                                })
        
        # Record feedback FIRST before generating new questions
        if general_feedback and current_results:
            try:
                import sys
                sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))
                from teacher_feedback_system import TeacherFeedbackSystem
                
                # Initialize teacher feedback system
                feedback_system = TeacherFeedbackSystem(
                    storage_file=os.path.join(STORAGE_DIR, "teacher_feedback_records.json")
                )
                
                # Record feedback for each storybook
                for i, result in enumerate(current_results):
                    if i >= len(selected_storybooks):
                        continue
                        
                    storybook = result.get("storybook")
                    storybook_id = storybook['id']
                    story_title = storybook['title']
                    story_content = get_story_content(storybook_id)
                    
                    if not story_content:
                        continue
                    
                    # Calculate start index for this storybook
                    storybook_start_index = sum(len(r.get("questions", [])) for r in current_results[:i])
                    
                    # Map global question_feedbacks to local indices
                    dspy_question_feedbacks = {}
                    original_qs = result.get("questions", [])
                    
                    for global_key, value in question_feedbacks.items():
                        global_index = int(global_key)
                        local_index = global_index - storybook_start_index
                        
                        if 0 <= local_index < len(original_qs):
                            dspy_question_feedbacks[local_index] = value
                    
                    # Extract generated question texts
                    generated_questions_list = []
                    for q in original_qs:
                        if isinstance(q, dict):
                            generated_questions_list.append(q.get("question", ""))
                        else:
                            generated_questions_list.append(str(q))
                    
                    # Get moral and segments (generate if needed)
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "gpt-moral-generation-structured",
                        MORAL_GEN_SCRIPT
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    StoryMoralGeneratorStructured = module.StoryMoralGeneratorStructured
                    
                    moral_generator = StoryMoralGeneratorStructured()
                    moral_result = moral_generator.generate_story_moral(story_content)
                    
                    if moral_result:
                        moral_text = moral_result.moral
                        segments_data = [
                            {"name": seg.name, "start": seg.START, "end": seg.END, 
                             "summary": seg.SUMMARY, "reasoning": seg.REASONING}
                            for seg in moral_result.segments
                        ]
                        
                        # Load actual question evaluations from JSON for this storybook
                        question_evaluations = []
                        try:
                            import json as json_module
                            eval_file = os.path.join(STORAGE_DIR, "question_evaluations.json")
                            if os.path.exists(eval_file):
                                with open(eval_file, 'r') as f:
                                    eval_data = json_module.load(f)
                                    evaluations = eval_data.get("evaluations", [])
                                    
                                    # Find evaluations for this storybook and objective
                                    for eval_item in evaluations:
                                        # Skip metadata entries
                                        if "set_metadata" in eval_item:
                                            continue
                                        
                                        # Match by story context (full story text)
                                        eval_story = eval_item.get("story_context", "")
                                        if eval_story and len(eval_story) > 100 and eval_story[:100] in story_content:
                                            question_evaluations.append(eval_item)
                                    
                                    print(f"[INFO] Loaded {len(question_evaluations)} evaluation records for feedback")
                        except Exception as eval_load_error:
                            print(f"[WARN]  Error loading evaluations: {eval_load_error}")
                        
                        # Process feedback with original questions AND evaluations
                        try:
                            feedback_system.process_feedback(
                                story_title=story_title,
                                objective=objective,
                                teacher_feedback=general_feedback,
                                feedback_type='negative',
                                question_evaluations=question_evaluations,  # NOW PASSING ACTUAL EVALUATIONS
                                story_context=story_content,
                                generated_questions=generated_questions_list,
                                question_feedbacks=dspy_question_feedbacks,
                                teacher_id='default_teacher',
                                school_id='default_school'
                            )
                            print(f"[INFO] Recorded feedback for {story_title} BEFORE regeneration")
                        except Exception as e:
                            print(f"[WARN]  Error recording feedback for {story_title}: {e}")
                            
            except Exception as e:
                print(f"[WARN]  Error in feedback recording: {e}")
                import traceback
                traceback.print_exc()
        
        # Now generate new questions
        # Load dynamic evaluators from feedback records BEFORE regeneration
        dynamic_evaluators = []
        try:
            import json
            feedback_file = os.path.join(STORAGE_DIR, "teacher_feedback_records.json")
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    feedback_data = json.load(f)
                    # Extract dynamic evaluators from feedback records
                    records = []
                    for school_key in feedback_data:
                        if school_key == "records":
                            records.extend(feedback_data.get("records", []))
                        elif isinstance(feedback_data[school_key], dict):
                            for teacher_key in feedback_data[school_key]:
                                if isinstance(feedback_data[school_key][teacher_key], list):
                                    records.extend(feedback_data[school_key][teacher_key])
                    
                    
                    for record in records:
                        details = record.get("action_taken", {}).get("details", {})
                        if details.get("status") == "active":
                            evaluator_name = details.get("evaluator_name", "unknown")
                            # Prefer explicit instruction text for clear criteria; fall back to description
                            evaluator_criteria = (
                                record.get("course_of_action", {}).get("instruction")
                                or record.get("action_taken", {}).get("reformulated_instruction")
                                or record.get("interpretation", {}).get("reformulated_instruction")
                                or details.get("evaluator_description", "")
                            )
                            evaluator_dict = {
                                "name": evaluator_name,
                                "criteria": evaluator_criteria
                            }
                            dynamic_evaluators.append(evaluator_dict)
        except Exception as e:
            print(f"[WARN]  Error loading dynamic evaluators for regeneration: {e}")
        
        
        results = []
        for i, storybook in enumerate(selected_storybooks):
            storybook_id = storybook['id']
            story_title = storybook['title']
            
            # Initialize next_set_num to avoid scope issues
            next_set_num = None
            
            # Get story content from JSON files
            story_content = get_story_content(storybook_id)
            if not story_content:
                continue
            
            # Use structured moral generation script directly
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "gpt-moral-generation-structured",
                MORAL_GEN_SCRIPT
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            StoryMoralGeneratorStructured = module.StoryMoralGeneratorStructured
            
            # Initialize structured generator
            moral_generator = StoryMoralGeneratorStructured()
            
            # Generate moral and segments
            moral_result = moral_generator.generate_story_moral(story_content)
            
            if not moral_result:
                continue
            
            moral = moral_result.moral
            segments_data = [
                {
                    "name": seg.name,
                    "start": seg.START,
                    "end": seg.END,
                    "summary": seg.SUMMARY,
                    "reasoning": seg.REASONING
                }
                for seg in moral_result.segments
            ]
            
            # Generate new questions using the structured script (with feedback learning)
            # Pass dynamic evaluators to ensure they are applied during evaluation
            questions_data = call_objective_question_generation_script(story_content, segments_data, objective, story_title, moral, dynamic_evaluators=dynamic_evaluators)
            
            # Extract questions from the returned data
            # NOTE: Evaluation is already handled inside call_objective_question_generation_script
            # Removed duplicate evaluation block to prevent creating set_003
            if questions_data and questions_data.get("questions"):
                questions = questions_data.get("questions", [])
            
            # Log regenerated questions to testing file
            if questions_data and questions:
                try:
                    # Determine set number from evaluation data or use default
                    import json
                    eval_file = os.path.join(STORAGE_DIR, "question_evaluations.json")
                    if os.path.exists(eval_file):
                        with open(eval_file, 'r') as f:
                            eval_data = json.load(f)
                            evaluations = eval_data.get("evaluations", [])
                            max_set_num = 0
                            for item in evaluations:
                                if "set_metadata" in item:
                                    meta = item["set_metadata"]
                                    if meta.get("storybook_id") == story_title and meta.get("objective") == objective:
                                        try:
                                            set_num = int(meta.get("set_number", "set_001").replace("set_", ""))
                                            max_set_num = max(max_set_num, set_num)
                                        except:
                                            pass
                            set_num_for_logging = f"set_{max_set_num:03d}"
                    else:
                        set_num_for_logging = "set_002"
                    
                    log_to_testing_file(story_title, objective, set_num_for_logging, questions, feedback=general_feedback)
                    print(f"[INFO] Logged {len(questions)} regenerated questions for {story_title} as {set_num_for_logging}")
                except Exception as log_error:
                    print(f"[WARN]  Failed to log regenerated questions: {log_error}")
            
            # Feedback was already recorded BEFORE generating questions (see api_regenerate_questions section)
            # No need to record again here
            
            if questions_data:
                results.append({
                    "storybook": storybook,
                    "moral": {
                        "generated": moral,
                        "status": "approved",
                        "feedback": "positive",
                        "regenerations": 0,
                        "timestamp": datetime.now().isoformat()
                    },
                    "segments": segments_data,
                    "questions": questions_data.get("questions", []),
                    "learning_objectives": questions_data.get("learning_objectives", []),
                    "objective": objective,
                    "frequency": "per-segment"
                })
            else:
                # Fallback if question generation fails
                results.append({
                    "storybook": storybook,
                    "moral": {
                        "generated": moral,
                        "status": "approved",
                        "feedback": "positive",
                        "regenerations": 0,
                        "timestamp": datetime.now().isoformat()
                    },
                    "segments": segments_data,
                    "questions": [],
                    "learning_objectives": [],
                    "objective": objective,
                    "frequency": "per-segment"
                })
        
        return jsonify({
            "success": True,
            "results": results,
            "message": "Questions regenerated using your feedback!"
        })
        
    except Exception as e:
        print(f"Error regenerating questions: {e}")
        return jsonify({"error": str(e)}), 500

def call_feedback_recording(inputs, outputs, feedback_type, question_feedbacks=None, overall_feedback=""):
    """Record feedback using the centralized system."""
    try:
        system = get_main_system()
        
        # Prepare inputs with original_questions
        story = inputs.get("story", "")
        segments = inputs.get("segments", [])
        objective = inputs.get("objective", "moral")
        story_title = inputs.get("story_title", "Unknown")
        
        # Original questions should be wrapped in a list of lists
        original_questions = [[q for q in outputs.get("questions", [])]]
        
        # Generated questions
        generated_questions = outputs.get("questions", [])
        learning_objectives = outputs.get("learning_objectives", [])
        
        # Convert question_feedbacks indices to the format the system expects
        formatted_feedbacks = {}
        if question_feedbacks:
            for idx, feedback in question_feedbacks.items():
                # Format as global index (0000XXXX where XXXX is the question index)
                global_idx = f"0000{int(idx):04d}"
                formatted_feedbacks[global_idx] = feedback
        
        # Record feedback
        system.record_feedback(
            story=story,
            segments=segments,
            objective=objective,
            story_title=story_title,
            original_questions=original_questions,
            generated_questions=generated_questions,
            learning_objectives=learning_objectives,
            question_feedbacks=formatted_feedbacks,
            overall_feedback=feedback_type
        )
        
    except Exception as e:
        print(f"Error recording feedback: {e}")
        import traceback
        traceback.print_exc()

@app.route("/api/approve-moral", methods=["POST"])
def api_approve_moral():
    """API endpoint to approve moral and generate questions"""
    try:
        data = request.get_json()
        results = data.get('results', [])
        
        # Update moral status to approved
        for result in results:
            result['moral']['status'] = 'approved'
            result['moral']['feedback'] = 'positive'
            result['moral']['timestamp'] = datetime.now().isoformat()
            
            # Get story content for question generation
            storybook = result['storybook']
            storybook_id = storybook['id']
            story_title = storybook['title']
            objective = result['objective']
            segments = result['segments']
            
            # Get story content
            story_content = get_story_content(storybook_id)
            if story_content:
                # Record positive feedback for moral approval
                moral_text = result['moral']['generated']
                inputs = {
                    "story": story_content,
                    "segments": segments_data,
                    "objective": objective,
                    "story_title": story_title,
                    "moral": moral_text
                }
                outputs = {
                    "questions": [],
                    "learning_objectives": []
                }
                
                # Generate questions using structured moral generation script directly
                questions_data = None
                try:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "gpt_moral_generation_structured",
                        MORAL_GEN_SCRIPT
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    StoryMoralGeneratorStructured = module.StoryMoralGeneratorStructured
                    
                    # Initialize structured generator
                    moral_generator = StoryMoralGeneratorStructured()
                    
                    # Generate moral and segments
                    moral_result = moral_generator.generate_story_moral(story_content)
                    
                    if moral_result:
                        moral_text = moral_result.moral
                        segments_data = [
                            {
                                "name": seg.name,
                                "start": seg.START,
                                "end": seg.END,
                                "summary": seg.SUMMARY,
                                "reasoning": seg.REASONING
                            }
                            for seg in moral_result.segments
                        ]
                        
                        # Now generate questions using the moral and segments as context
                        questions_data = call_objective_question_generation_script(
                            story_content, objective, segments_data, story_title, moral_text
                        )
                        
                        print(f"[INFO] Generated questions using structured moral: {moral_text[:50]}...")
                        print(f"   Segments: {len(segments_data)}")
                        
                    else:
                        print(f"[WARN]  Structured moral generation failed, using fallback")
                        # Fallback to simple generation
                        questions_data = {
                            "questions": [
                                {
                                    "question": f"How does the story of {story_title} relate to {objective}?",
                                    "type": "comprehension",
                                    "difficulty": "medium",
                                    "explanation": f"Generated question focusing on {objective}",
                                    "page_number": 1
                                }
                            ],
                            "learning_objectives": []
                        }
                except Exception as e:
                    print(f"[WARN]  Error generating questions: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback to simple generation
                    questions_data = {
                        "questions": [
                            {
                                "question": f"How does the story of {story_title} relate to {objective}?",
                                "type": "comprehension",
                                "difficulty": "medium",
                                "explanation": f"Generated question focusing on {objective}",
                                "page_number": 1
                            }
                        ],
                        "learning_objectives": []
                    }
                
                # Debug output
                
                if questions_data:
                    questions = questions_data.get("questions", [])
                    
                    # Evaluate questions using ContextQ evaluators
                    try:
                        import sys
                        import os
                        sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))
                        from contextq_evaluators import ContextQEvaluationPipeline
                        
                        # Initialize evaluation pipeline
                        evaluation_pipeline = ContextQEvaluationPipeline(
                            storage_file=os.path.join(STORAGE_DIR, "question_evaluations.json"),
                            feedback_records_file=os.path.join(STORAGE_DIR, "teacher_feedback_records.json")
                        )
                        
                        # Extract question texts for evaluation
                        question_texts = [q.get("question", "") for q in questions if q.get("question")]
                        
                        if question_texts:
                            print(f"\nEvaluating {len(question_texts)} questions with ContextQ rubric...")
                            
                            # Evaluate questions
                            evaluations = evaluation_pipeline.evaluate_and_store(
                                questions=question_texts,
                                story_context=story_content,
                                moral_or_objective=objective,
                                clear_existing=False  # Don't clear existing evaluations
                            )
                            
                            # Add evaluation scores to questions
                            for i, question in enumerate(questions):
                                if i < len(evaluations):
                                    question["evaluation"] = {
                                        "average_score": evaluations[i]["average_score"],
                                        "scores": evaluations[i]["scores"],
                                        "question_id": evaluations[i]["question_id"]
                                    }
                            
                            print(f"[INFO] Questions evaluated and scored")
                        else:
                            print("[WARN]  No questions to evaluate")
                            
                    except Exception as e:
                        print(f"[WARN]  Error evaluating questions: {e}")
                        # Continue without evaluation if it fails
                    
                    result['questions'] = {
                        "generated": questions,
                        "status": "generated",
                        "feedback_summary": None,
                        "regenerations": 0,
                        "timestamp": datetime.now().isoformat()
                    }
                    result['learning_objectives'] = questions_data.get("learning_objectives", [])
                else:
                    result['questions']['status'] = 'failed'
        
        return jsonify({
            "success": True,
            "results": results,
            "stage": "questions_generated"
        })
        
    except Exception as e:
        print(f"Error approving moral and generating questions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/regenerate-moral", methods=["POST"])
def api_regenerate_moral():
    """API endpoint to regenerate moral using feedback"""
    try:
        data = request.get_json()
        results = data.get('results', [])
        selected_indices = data.get('selectedIndices', [])
        feedback = data.get('feedback', 'negative')
        
        # Only regenerate moral for selected storybooks
        for index in selected_indices:
            if 0 <= index < len(results):
                result = results[index]
                storybook = result['storybook']
                storybook_id = storybook['id']
                story_title = storybook['title']
                
                # Get story content
                story_content = get_story_content(storybook_id)
                if story_content:
                    # Regenerate moral using structured script
                    try:
                        import importlib.util
                        spec = importlib.util.spec_from_file_location(
                            "gpt_moral_generation_structured",
                            MORAL_GEN_SCRIPT
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        StoryMoralGeneratorStructured = module.StoryMoralGeneratorStructured
                        
                        # Initialize structured generator
                        moral_generator = StoryMoralGeneratorStructured()
                        
                        # Generate moral and segments
                        moral_result = moral_generator.generate_story_moral(story_content)
                        
                        if moral_result:
                            result['moral']['generated'] = moral_result.moral
                        result['moral']['regenerations'] += 1
                        result['moral']['feedback'] = feedback
                        result['moral']['timestamp'] = datetime.now().isoformat()
                        result['segments'] = [
                                {
                                    "name": seg.name,
                                    "start": seg.START,
                                    "end": seg.END,
                                    "summary": seg.SUMMARY,
                                    "reasoning": seg.REASONING
                                }
                                for seg in moral_result.segments
                            ]
                    except Exception as e:
                        print(f"Error regenerating moral: {e}")
        
        return jsonify({
            "success": True,
            "results": results,
            "stage": "moral_approval"
        })
        
    except Exception as e:
        print(f"Error regenerating moral: {e}")
        return jsonify({"error": str(e)}), 500

# Catch-all route for React Router
@app.route('/<path:path>')
def serve_react_app(path):
    """Serve the React app for all non-API routes"""
    if path.startswith('api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
    
    # Serve React app for all other routes
    return send_from_directory('../frontend/build', 'index.html')

if __name__ == "__main__":
    print("Starting Teacher Interface Server...")
    print("Storybook Library Dashboard")
    print("Access at: http://localhost:5001")
    print("Assets path:", ASSETS_PATH)
    
    app.run(debug=True, host="0.0.0.0", port=5001)
