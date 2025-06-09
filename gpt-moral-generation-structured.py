"""
This script generates the moral lessons and segments of a story.
"""

import json
from openai import OpenAI
import os

class StoryMoralGeneratorStructured:
    def __init__(self):
        self.client = OpenAI()
        self.client.api_key = os.environ.get('OPENAI_API_KEY')
        self.model = "gpt-4.1-2025-04-14"
        with open("./prompts/moralQ_prompts/story_moral_prompt.txt", "r") as f:
            self.prompt = f.read()

        # Define the expected structured output format 
        self.moral_tool_schema = {
            "type": "object",
            "properties": {
                "moral": {"type": "string"},
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "pattern": "^segment_\\d+$"},
                            "START": {"type": "integer"},
                            "END": {"type": "integer"},
                            "SUMMARY": {"type": "string"},
                            "REASONING": {"type": "string"}
                        },
                        "required": ["name", "START", "END", "SUMMARY", "REASONING"],
                        "additionalProperties": False
                    },
                    "minItems": 2,
                    "maxItems": 7
                }
            },
            "required": ["moral", "segments"],
            "additionalProperties": False
        }
    

    def load_story_content(self, story_dir: str) -> str:
        """Load story content from JSON files in a directory."""
        story_content = []
        # Get all JSON files and sort them by filename
        json_files = [f for f in os.listdir(story_dir) if f.endswith('.json')]
        json_files.sort()  
        
        for filename in json_files:
            with open(os.path.join(story_dir, filename), 'r') as f:
                page_content = json.load(f)
                # Handle if text is nested in a list
                text = page_content.get('text', '')
                if isinstance(text, list):
                    text = ' '.join(text)
                story_content.append(text)
        return ' '.join(story_content)

    def generate_story_moral(self, story: str):
        """Use OpenAI structured output (function calling) to generate moral."""
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": story}
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "story_moral",
                        "schema": self.moral_tool_schema,
                        "strict": True
                    }
                }
            )

            # Extract the structured response directly from output_text
            try:
                return json.loads(response.output_text)
            except json.JSONDecodeError:
                print(f"Error parsing JSON response: {response.output_text}")
                return None

        except Exception as e:
            print(f"Error using structured output API: {e}")
            return None

def main():
    generator = StoryMoralGeneratorStructured()
    
    asset_path = "/Users/mariyamohiuddin/Desktop/interactive-storybook-assets/qna_json/"
    output_path = "/Users/mariyamohiuddin/Desktop/Outputs/"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    

    educare_storybooks = ["ada_twist_scientist", "a_letter_to_amy", "boxitects", "grumpy_monkey", 
                         "if_you_give_a_mouse_a_cookie", "jabari_jumps", "last_stop_on_market_street", 
                         "mango_abuela_and_me", "peters_chair", "stand_tall_molly_lou_melon", 
                         "the_proudest_blue"]

    kipp_storybooks = ["chicka_chicka_boom_boom", "the_little_red_hen-muldrow", 
                       "rap_a_tap_tap_heres_bojangles", "three_little_pigs-marshall", 
                       "the_three_billy_goats_gruff-finch", "ganeshas_sweet_tooth",
                       "moon_rope", "the_story_of_ferdinand", "helpers_in_my_community", 
                       "a_day_in_the_life_of_a_firefighter",
                      "my_five_senses-aliki"]

    stories = sorted(kipp_storybooks + educare_storybooks)
    
    # Process each story
    for story in stories:
        print(f"\nProcessing story: {story}")
        output_file = os.path.join(output_path, f"{story}_moral_segments.json")
        
        # Skip if output file already exists
        if os.path.exists(output_file):
            print(f"Output file already exists: {output_file}")
            continue
            
        story_dir = os.path.join(asset_path, story)
        
        if not os.path.exists(story_dir):
            print(f"Story directory not found: {story_dir}")
            continue
            
        # Load story content from JSON files
        story_content = generator.load_story_content(story_dir)
        
        # Generate moral and segments
        result = generator.generate_story_moral(story_content)
        
        if result:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"Saved moral and segments to: {output_file}")
        else:
            print(f"Failed to generate moral for story: {story}")

if __name__ == "__main__":
    main()
