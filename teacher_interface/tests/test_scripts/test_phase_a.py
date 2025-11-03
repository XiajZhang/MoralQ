"""
Phase A: Moral and Question Generation Quality Test Suite

Purpose: Ensure story-based moral and question generation produces age-appropriate,
interpretable, and thematically correct outputs.

Test Cases:
- A1: A Letter to Amy - Empathy
- A2: Grumpy Monkey - Emotional regulation
- A3: Last Stop on Market Street - Gratitude
- A4: If You Give a Mouse a Cookie - Consequence awareness
- A5: Ada Twist, Scientist - Curiosity

Evaluation Metrics: Relevance (to story moral), Appropriateness (4-6 years), Clarity (simple syntax)
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add backend to path (test_scripts is one level deeper now)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'services'))

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(env_path)

# Import required modules
from server import get_story_content, ASSETS_PATH, QNA_JSON_PATH
import importlib.util

# Test cases definition
TEST_CASES = {
    "A1": {
        "test_id": "A1",
        "story_title": "A Letter to Amy",
        "story_id": "a_letter_to_amy",  # Assuming this is the folder name
        "objective": "Empathy",
        "expected_moral": "Be proud of your friends and be kind",
        "unexpected_moral": "Have fun at your birthday",
        "expected_outcome": {
            "moral_focus": "Should emphasize being proud of friends and kindness, NOT about having fun at birthday",
            "question_focus": "Should encourage reflection on friendship, empathy, and being proud of others"
        }
    },
    "A2": {
        "test_id": "A2",
        "story_title": "Grumpy Monkey",
        "story_id": "grumpy_monkey",
        "objective": "Emotional regulation",
        "expected_moral": "Understanding and accepting emotions",
        "unexpected_moral": "Don't be grumpy",
        "expected_outcome": {
            "moral_focus": "Should focus on understanding emotions and emotional regulation, NOT simply avoiding grumpiness",
            "question_focus": "Should explore emotional awareness, acceptance of feelings, and healthy ways to express emotions"
        }
    },
    "A3": {
        "test_id": "A3",
        "story_title": "Last Stop on Market Street",
        "story_id": "last_stop_on_market_street",  # May need to adjust
        "objective": "Gratitude",
        "expected_moral": "Finding beauty and gratitude in everyday life",
        "unexpected_moral": None,
        "expected_outcome": {
            "moral_focus": "Should encourage seeing beauty in everyday experiences and being grateful",
            "question_focus": "Should encourage reflection on seeing beauty in everyday life and appreciating what we have"
        }
    },
    "A4": {
        "test_id": "A4",
        "story_title": "If You Give a Mouse a Cookie",
        "story_id": "if_you_give_a_mouse_a_cookie",
        "objective": "Consequence awareness",
        "expected_moral": "Understanding cause and effect relationships",
        "unexpected_moral": None,
        "expected_outcome": {
            "moral_focus": "Should emphasize understanding consequences and cause-effect relationships",
            "question_focus": "Should connect actions to reactions and avoid abstract reasoning, using concrete examples"
        }
    },
    "A5": {
        "test_id": "A5",
        "story_title": "Ada Twist, Scientist",
        "story_id": "ada_twist_scientist",
        "objective": "Curiosity",
        "expected_moral": "Perseverance and curiosity in learning",
        "unexpected_moral": None,
        "expected_outcome": {
            "moral_focus": "Should stress perseverance and curiosity, avoiding moral drift to unrelated themes",
            "question_focus": "Should encourage scientific curiosity, asking questions, and persistence in learning"
        }
    }
}


class PhaseATester:
    """Test suite for Phase A: Moral and Question Generation Quality"""
    
    def __init__(self):
        self.results = []
        self.timestamp = datetime.now().isoformat()
        
    def load_story_content(self, story_id: str) -> Optional[str]:
        """Load story content from assets directory"""
        story_path = os.path.join(QNA_JSON_PATH, story_id)
        if not os.path.exists(story_path):
            print(f"[ERROR] Story path not found: {story_path}")
            print(f"[INFO] Available stories in {QNA_JSON_PATH}:")
            if os.path.exists(QNA_JSON_PATH):
                for item in os.listdir(QNA_JSON_PATH):
                    print(f"  - {item}")
            return None
        
        story_content = ""
        json_files = sorted([f for f in os.listdir(story_path) if f.endswith('.json')])
        
        for json_file in json_files:
            file_path = os.path.join(story_path, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    page_data = json.load(f)
                    if 'text' in page_data:
                        if isinstance(page_data['text'], list):
                            story_content += " ".join(page_data['text']) + "\n"
                        else:
                            story_content += str(page_data['text']) + "\n"
            except Exception as e:
                print(f"[WARN] Error reading {file_path}: {e}")
        
        return story_content.strip() if story_content else None
    
    def generate_moral(self, story_content: str) -> Optional[Dict[str, Any]]:
        """Generate moral and segments using the moral generation script"""
        try:
            # Import moral generation script
            moral_script_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'backend', 'services', 
                'gpt-moral-generation-structured.py'
            )
            
            spec = importlib.util.spec_from_file_location(
                "gpt-moral-generation-structured",
                moral_script_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            generator = module.StoryMoralGeneratorStructured()
            result = generator.generate_story_moral(story_content)
            
            # Convert Pydantic model to dict
            return {
                "moral": result.moral,
                "segments": [
                    {
                        "name": seg.name,
                        "start": seg.START,
                        "end": seg.END,
                        "summary": seg.SUMMARY,
                        "reasoning": seg.REASONING
                    }
                    for seg in result.segments
                ]
            }
        except Exception as e:
            print(f"[ERROR] Moral generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_questions(self, story_content: str, segments: List[Dict], objective: str, story_title: str) -> Optional[Dict[str, Any]]:
        """Generate questions using the question generator module"""
        try:
            # Import question generator module (same as server.py uses)
            from question_generator import QuestionGeneratorModule
            
            # Initialize generator
            generator = QuestionGeneratorModule()
            
            # Format segments for the generator (convert our format to expected format)
            formatted_segments = [
                {
                    "START": seg.get("start", 1),
                    "END": seg.get("end", 1),
                    "SUMMARY": seg.get("summary", ""),
                    "REASONING": seg.get("reasoning", "")
                }
                for seg in segments
            ]
            
            # Generate questions (use 'generate' method)
            result = generator.generate(
                story=story_content,
                segments=formatted_segments,
                objective=objective,
                story_title=story_title,
                moral_text=None,  # We can add moral text if needed
                avoidance_instructions=""
            )
            
            # Parse JSON output if needed
            if isinstance(result, dict):
                questions = result.get("questions", [])
                learning_objectives = result.get("learning_objectives", [])
            else:
                # If it's a string, parse it
                import json
                if isinstance(result, str):
                    result = json.loads(result)
                questions = result.get("questions", [])
                learning_objectives = result.get("learning_objectives", [])
            
            return {
                "questions": questions if isinstance(questions, list) else [questions],
                "learning_objectives": learning_objectives if isinstance(learning_objectives, list) else [learning_objectives]
            }
        except Exception as e:
            print(f"[ERROR] Question generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def run_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case"""
        test_id = test_case["test_id"]
        story_title = test_case["story_title"]
        story_id = test_case["story_id"]
        objective = test_case["objective"]
        
        print(f"\n{'='*80}")
        print(f"Running Test {test_id}: {story_title} - {objective}")
        print(f"{'='*80}")
        
        result = {
            "test_id": test_id,
            "story_title": story_title,
            "story_id": story_id,
            "objective": objective,
            "timestamp": datetime.now().isoformat(),
            "expected_outcomes": test_case["expected_outcome"],
            "generated_results": {},
            "status": "pending"
        }
        
        # Load story content
        print(f"[STEP 1] Loading story content for: {story_id}")
        story_content = self.load_story_content(story_id)
        if not story_content:
            result["status"] = "failed"
            result["error"] = f"Could not load story content for {story_id}"
            print(f"[ERROR] {result['error']}")
            return result
        
        result["generated_results"]["story_length"] = len(story_content)
        print(f"[SUCCESS] Loaded story content ({len(story_content)} characters)")
        
        # Generate moral
        print(f"[STEP 2] Generating moral for objective: {objective}")
        moral_result = self.generate_moral(story_content)
        if not moral_result:
            result["status"] = "failed"
            result["error"] = "Moral generation failed"
            print(f"[ERROR] Moral generation failed")
            return result
        
        generated_moral = moral_result["moral"]
        segments = moral_result["segments"]
        result["generated_results"]["moral"] = generated_moral
        result["generated_results"]["segments"] = segments
        print(f"[SUCCESS] Generated moral: {generated_moral[:100]}...")
        print(f"[SUCCESS] Generated {len(segments)} segments")
        
        # Generate questions
        print(f"[STEP 3] Generating questions")
        questions_result = self.generate_questions(story_content, segments, objective, story_title)
        if not questions_result:
            result["status"] = "partial"
            result["error"] = "Question generation failed"
            print(f"[ERROR] Question generation failed")
        else:
            generated_questions = questions_result["questions"]
            learning_objectives = questions_result.get("learning_objectives", [])
            result["generated_results"]["questions"] = generated_questions
            result["generated_results"]["learning_objectives"] = learning_objectives
            print(f"[SUCCESS] Generated {len(generated_questions)} questions")
            result["status"] = "completed"
        
        return result
    
    def run_all_tests(self):
        """Run all test cases"""
        print(f"\n{'='*80}")
        print(f"Phase A Test Suite: Moral and Question Generation Quality")
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
            "test_suite": "Phase A: Moral and Question Generation Quality",
            "timestamp": self.timestamp,
            "total_tests": len(self.results),
            "completed": sum(1 for r in self.results if r["status"] == "completed"),
            "partial": sum(1 for r in self.results if r["status"] == "partial"),
            "failed": sum(1 for r in self.results if r["status"] == "failed"),
            "test_results": self.results
        }
        
        # Save to JSON file in test_results folder
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'test_results')
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, "phase_a_results.json")
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
    tester = PhaseATester()
    tester.run_all_tests()

