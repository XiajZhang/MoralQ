"""
Teacher Interface Flask Application
Serves the storybook library dashboard for educators
"""

from flask import Flask, render_template, jsonify, send_file, request, send_from_directory
import os
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from firebase_client import FirebaseClientAuth
from dotenv import load_dotenv

app = Flask(__name__, static_folder='../frontend/build/static', template_folder='../frontend/build')

# Load environment variables from .env file in root MoralQ directory
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

# Configuration
ASSETS_PATH = "/Users/mariyamohiuddin/Desktop/interactive-storybook-assets"
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
    """API endpoint to generate moral lesson only"""
    try:
        data = request.get_json()
        selected_storybooks = data.get('selectedStorybooks', [])
        objective = data.get('objective', 'moral')
        
        if not selected_storybooks:
            return jsonify({"error": "No storybooks selected"}), 400
        
        # Generate moral for each selected storybook
        results = []
        for storybook in selected_storybooks:
            storybook_id = storybook['id']
            story_title = storybook['title']
            
            # Get story content from JSON files
            story_content = get_story_content(storybook_id)
            if not story_content:
                continue
            
            # Get moral and segments using DSPy script
            moral_data = call_simple_moral_script(story_content, story_title)
            if not moral_data:
                continue
            
            moral = moral_data.get("moral", "")
            segments = moral_data.get("segments", [])
            
            # Create the new structure
            result = {
                "storybook": storybook,
                "objective": objective,
                "moral": {
                    "generated": moral,
                    "status": "pending",
                    "feedback": None,
                    "regenerations": 0,
                    "timestamp": datetime.now().isoformat()
                },
                "segments": segments,
                "questions": {
                    "generated": [],
                    "status": "not_generated",
                    "feedback_summary": None,
                    "regenerations": 0,
                    "timestamp": None
                }
            }
            
            results.append(result)
        
        return jsonify({
            "success": True,
            "results": results,
            "stage": "moral_approval"
        })
        
    except Exception as e:
        print(f"Error generating moral: {e}")
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
    
    return story_content.strip()

def call_dspy_script(story_content):
    """Call the DSPy moral generation script"""
    try:
        # Path to the DSPy script
        dspy_script = os.path.join(os.path.dirname(__file__), "..", "services", "gpt-moral-generation-DSpy.py")
        
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

def call_simple_moral_script(story_content, story_title):
    """Call the DSPy moral generation script"""
    try:
        # Import the moral generator directly
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "moral_generator", 
            os.path.join(os.path.dirname(__file__), "..", "services", "gpt-moral-generation-DSpy.py")
        )
        moral_generator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(moral_generator_module)
        
        MoralGenerator = moral_generator_module.MoralGenerator
        generator = MoralGenerator()
        result = generator.generate_moral(story_content, story_title)
        
        return result
        
    except Exception as e:
        print(f"Error calling moral script: {e}")
        print("Falling back to original moral generation...")
        # Fallback to original script
        return call_dspy_script(story_content)

def record_moral_feedback(story_title, story_content, moral, segments, is_positive, reason=""):
    """Record moral feedback (placeholder - no optimization for morals yet)"""
    # For now, we're not using optimization for moral generation
    # This is just a placeholder for future implementation
    pass

def call_objective_question_generation_script(story_content, segments, objective, story_title):
    """Call the DSPy objective questions generation script directly"""
    try:
        # Import the objective question generator class directly
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        # Import the module with hyphens in filename
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "objective_questions", 
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "services", "gpt-objective-questions-structured-DSpy.py")
        )
        objective_questions_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(objective_questions_module)
        
        ObjectiveQuestionGeneratorDSPy = objective_questions_module.ObjectiveQuestionGeneratorDSPy
        
        # Create generator instance
        generator = ObjectiveQuestionGeneratorDSPy()
        
        # Generate questions
        questions_result = generator.generate_objective_questions(
            story=story_content,
            segments=segments,
            objective=objective,
            story_title=story_title
        )
        
        if questions_result:
            return questions_result
        else:
            print("Failed to generate objective questions")
            return None
            
    except Exception as e:
        print(f"Error calling objective question generation script: {e}")
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
            os.path.join(os.path.dirname(__file__), "../services/gpt-objective-questions-structured-DSpy.py")
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
                    moral_data = call_dspy_script(story_content)
                    if moral_data:
                        moral = moral_data.get("moral", "")
                        segments = moral_data.get("segments", [])
                        questions_data = call_objective_question_generation_script(story_content, segments, objective, story_title)
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
                        moral_data = call_dspy_script(story_content)
                        if moral_data:
                            moral = moral_data.get("moral", "")
                            segments = moral_data.get("segments", [])
                            questions_data = call_objective_question_generation_script(story_content, segments, objective, story_title)
                            if questions_data:
                                current_results.append({
                                    "storybook": storybook,
                                    "questions": questions_data.get("questions", [])
                                })
        
        # Now generate new questions and record feedback
        results = []
        for i, storybook in enumerate(selected_storybooks):
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
            
            # Generate new questions using the DSPy script (with feedback learning)
            questions_data = call_objective_question_generation_script(story_content, segments, objective, story_title)
            
            # Record feedback for learning (only if we have feedback)
            if general_feedback and questions_data:
                inputs = {
                    "story": story_content,
                    "segments": segments,
                    "objective": objective,
                    "story_title": story_title,
                    "moral": moral  # Add moral data for the new format
                }
                outputs = {
                    "questions": current_results[i]["questions"] if i < len(current_results) else [],
                    "learning_objectives": questions_data.get("learning_objectives", []) if questions_data else []
                }
                
                # Map global question indices to local indices for this storybook
                # Calculate the start index for this storybook based on current results
                storybook_start_index = sum(len(current_results[j]["questions"]) for j in range(i))
                
                # Convert global question_feedbacks to local indices for this storybook
                dspy_question_feedbacks = {}
                print(f"Debug: Processing storybook {i}: {storybook['title']}")
                print(f"Debug: storybook_start_index = {storybook_start_index}")
                print(f"Debug: outputs['questions'] length = {len(outputs['questions'])}")
                
                for global_key, value in question_feedbacks.items():
                    global_index = int(global_key)
                    local_index = global_index - storybook_start_index
                    print(f"Debug: global_index {global_index} -> local_index {local_index}")
                    
                    # Only include feedback for questions that belong to this storybook
                    if 0 <= local_index < len(outputs["questions"]):
                        dspy_question_feedbacks[local_index] = value
                        print(f"Debug: Added feedback for local_index {local_index}")
                    else:
                        print(f"Debug: Skipped feedback for local_index {local_index} (out of range)")
                
                print(f"Debug: Final dspy_question_feedbacks = {dspy_question_feedbacks}")
                
                # Use general feedback and individual question feedbacks for learning
                call_feedback_recording(inputs, outputs, 'negative', dspy_question_feedbacks, general_feedback)
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
                    "segments": segments,
                    "questions": {
                        "generated": questions_data.get("questions", []),
                        "status": "generated",
                        "feedback_summary": None,
                        "regenerations": 0,
                        "timestamp": datetime.now().isoformat()
                    },
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
                    "segments": segments,
                    "questions": {
                        "generated": [],
                        "status": "failed",
                        "feedback_summary": None,
                        "regenerations": 0,
                        "timestamp": datetime.now().isoformat()
                    },
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
    """Record feedback using the DSPy script"""
    try:
        # Import the objective question generator class
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "objective_questions", 
            os.path.join(os.path.dirname(__file__), "../services/gpt-objective-questions-structured-DSpy.py")
        )
        objective_questions_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(objective_questions_module)
        
        ObjectiveQuestionGeneratorDSPy = objective_questions_module.ObjectiveQuestionGeneratorDSPy
        
        # Create generator instance
        generator = ObjectiveQuestionGeneratorDSPy()
        
        # Record feedback
        if feedback_type == 'positive':
            generator.record_positive_feedback(inputs, outputs, question_feedbacks, overall_feedback)
        else:
            generator.record_negative_feedback(inputs, outputs, question_feedbacks, overall_feedback)
        
    except Exception as e:
        print(f"Error recording feedback: {e}")

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
                    "segments": segments,
                    "objective": objective,
                    "story_title": story_title,
                    "moral": moral_text
                }
                outputs = {
                    "questions": [],
                    "learning_objectives": []
                }
                call_feedback_recording(inputs, outputs, 'positive')
                
                # Generate questions using DSPy script
                questions_data = call_objective_question_generation_script(
                    story_content, segments, objective, story_title
                )
                
                if questions_data:
                    result['questions'] = {
                        "generated": questions_data.get("questions", []),
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
                    # Regenerate moral using DSPy script
                    moral_data = call_simple_moral_script(story_content, story_title)
                    if moral_data:
                        result['moral']['generated'] = moral_data.get("moral", "")
                        result['moral']['regenerations'] += 1
                        result['moral']['feedback'] = feedback
                        result['moral']['timestamp'] = datetime.now().isoformat()
                        result['segments'] = moral_data.get("segments", [])
        
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
