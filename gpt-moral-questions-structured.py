"""
Generate moral questions using OpenAI's structured output API.
"""

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
import os
import json

class MoralQuestion(BaseModel):
    question: str
    type: str = Field(..., description="Type of question: comprehension, reflection, application, or extension")
    difficulty: str = Field(..., description="Difficulty level: easy, medium, or hard")
    explanation: str = Field(..., description="Explanation of why this question helps understand the moral")

class MoralQuestionsResponse(BaseModel):
    questions: List[MoralQuestion] = Field(..., min_items=5, max_items=10)
    moral: str
    story_title: str
    learning_objectives: List[str] = Field(..., min_items=3)
    discussion_points: List[str] = Field(..., min_items=3)

class MoralQuestionGenerator:
    def __init__(self):
        self.client = OpenAI()
        self.client.api_key = os.environ.get('OPENAI_API_KEY')
        self.model = "gpt-4.1-2025-04-14"

    def generate_moral_questions(self, story: str, moral: str, story_title: str, segments: List[dict]) -> MoralQuestionsResponse:
        """
        Generate questions that help learners understand and reflect on the story's moral.
        
        Args:
            story: The full story text
            moral: The moral of the story
            story_title: The title of the story
            segments: List of story segments with their summaries and reasoning
            
        Returns:
            MoralQuestionsResponse: Contains questions, learning objectives, and discussion points
        """
        # Format segments for the prompt
        formatted_segments = "\n".join([
            f"Segment {i+1}: {segment.get('SUMMARY', '')}"
            f"\nReasoning: {segment.get('REASONING', '')}"
            for i, segment in enumerate(segments)
        ])
        
        questions_prompt = f"""You are an expert in creating educational questions for children aged 4-6 years old that help them understand moral lessons.

When creating questions, keep in mind:
- Use simple, clear language appropriate for young children
- Focus on concrete examples and situations they can relate to
- Avoid complex concepts or abstract thinking
- Encourage discussion and sharing of personal experiences

Story Title: {story_title}

Story Moral: {moral}

Story Segments:
{formatted_segments}

Full Story:
{story}

Generate 5-10 questions that help young children (4-6 years old) understand and reflect on this moral. Use the story segments to create questions that:

1. Test understanding of specific story segments and their connection to the moral
2. Encourage reflection on how different parts of the story contribute to the moral lesson
3. Apply the moral to real-life situations that are relevant to young children
4. Challenge learners to think about the story in ways they can understand

For each question, provide:
- Question type (comprehension, reflection, application, extension)
- Difficulty level (easy, medium, hard) - keep in mind the age group
- Explanation of how this question helps young children understand the moral

Also provide:
- 3-4 learning objectives for this lesson
- 3-5 discussion points to guide classroom discussion with young children

"""

        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": "You are an expert in creating educational questions that help learners understand moral lessons."},
                {"role": "user", "content": questions_prompt}
            ],
            text_format=MoralQuestionsResponse
        )
        return response.output_parsed

    def load_story_content(self, story_dir: str) -> str:
        """Load story content from JSON files, maintaining page order."""
        story_content = []
        json_files = [f for f in os.listdir(story_dir) if f.endswith('.json')]
        
        # Sort files by page number (format: storyname_00.json, storyname_01.json, etc.)
        json_files.sort(key=lambda f: int(f.split('.')[0].split('_')[-1]))
        
        for filename in json_files:
            with open(os.path.join(story_dir, filename), 'r') as f:
                page_content = json.load(f)
                text = page_content.get('text', '')
                if isinstance(text, list):
                    text = ' '.join(text)
                story_content.append(text)
        
        return ' '.join(f"[Page {i+1}] {text}" for i, text in enumerate(story_content))

def main():
    generator = MoralQuestionGenerator()
    
    asset_path = "/Users/mariyamohiuddin/Desktop/interactive-storybook-assets/qna_json/"
    moral_path = "/Users/mariyamohiuddin/Desktop/Outputs/"
    output_path = "/Users/mariyamohiuddin/Desktop/MoralQuestions/"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Get list of moral files
    moral_files = [f for f in os.listdir(moral_path) if f.endswith('_moral_segments.json')]
    
    # Process each moral file
    for moral_file in moral_files:
        story_name = moral_file.replace('_moral_segments.json', '')
        print(f"\nProcessing story: {story_name}")
        
        # Load story content
        story_dir = os.path.join(asset_path, story_name)
        if not os.path.exists(story_dir):
            print(f"Story directory not found: {story_dir}")
            continue
            
        story_content = generator.load_story_content(story_dir)
        
        # Load existing moral
        moral_file_path = os.path.join(moral_path, moral_file)
        if not os.path.exists(moral_file_path):
            print(f"Moral file not found: {moral_file_path}")
            continue
            
        with open(moral_file_path, 'r', encoding='utf-8') as f:
            moral_data = json.load(f)
            moral = moral_data.get('moral', '')
            segments = moral_data.get('segments', [])
            
        # Generate moral questions
        questions_file = os.path.join(output_path, f"{story_name}_moral_questions.json")
        if os.path.exists(questions_file):
            print(f"Questions file already exists: {questions_file}")
            continue
            
        try:
            questions_result = generator.generate_moral_questions(
                story=story_content,
                moral=moral,
                story_title=story_name,
                segments=segments
            )
            
            if questions_result:
                with open(questions_file, 'w', encoding='utf-8') as f:
                    json.dump(questions_result.dict(), f, indent=4, ensure_ascii=False)
                print(f"Saved moral questions to: {questions_file}")
            else:
                print(f"Failed to generate moral questions for story: {story_name}")
                
        except Exception as e:
            print(f"Error generating moral questions for {story_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
