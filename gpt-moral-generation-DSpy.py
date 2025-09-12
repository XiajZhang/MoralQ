"""
Generate moral lessons and story segments using DSPy framework.
This refactored version uses DSPy's declarative approach for better modularity and automatic optimization.
"""

import dspy
import os
import json
from typing import List
from pathlib import Path

# Configure DSPy with OpenAI
dspy.configure(lm=dspy.LM("openai/gpt-4.1-2025-04-14"))

class MoralExtractor(dspy.Signature):
    """Extract moral lessons and segment stories for children aged 4-6 using the original prompt format."""
    
    story = dspy.InputField(desc="Children's storybook content with page markers")
    moral = dspy.OutputField(desc="Age-appropriate moral lesson for children aged 4-6, simplified to their language level")
    segments = dspy.OutputField(desc="""Return a JSON array of 3-7 story segments (maximum 7). Each segment must be a JSON object with exactly these fields:
    {
        "START": <page_number>,
        "END": <page_number>,
        "SUMMARY": "<brief summary>",
        "REASONING": "<explanation>"
    }
    
    Ensure the output is valid JSON with proper quotes and structure. Limit to maximum 7 segments.""")

class StoryMoralGeneratorDSPy:
    def __init__(self):
        """Initialize the DSPy-based moral generator."""
        # Create the moral extraction module
        self.moral_extractor = dspy.Predict(MoralExtractor)
        
       
    def generate_story_moral(self, story: str) -> dict:
        """Generate moral lesson and story segments using DSPy."""
        try:
            # DSPy handles the prompting and model interaction automatically
            result = self.moral_extractor(story=story)
            
            segments = result.segments
            if isinstance(segments, str):
                # Try to extract JSON from the string
                try:
                    # Look for JSON array pattern in the string
                    import re
                    json_match = re.search(r'\[.*\]', segments, re.DOTALL)
                    if json_match:
                        import json
                        segments = json.loads(json_match.group())
                    else:
                        # If no JSON found, create a simple structure
                        segments = self._parse_segments_from_text(segments)
                except:
                    segments = self._parse_segments_from_text(segments)
            
            # Add name field to each segment and limit to 7 segments
            if isinstance(segments, list):
                for i, segment in enumerate(segments[:7], 1):  # Limit to 7 segments
                    segment['name'] = f"segment_{i}"
            
            # Convert DSPy result to dictionary format
            return {
                "moral": result.moral,
                "segments": segments[:7] if isinstance(segments, list) else segments  # Ensure max 7 segments
            }
        except Exception as e:
            print(f"Error generating moral: {e}")
            return None
    
    def _parse_segments_from_text(self, text: str) -> list:
        """Parse segments from text output if JSON parsing fails."""
        segments = []
        lines = text.split('\n')
        current_segment = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('- START:'):
                if current_segment:
                    segments.append(current_segment)
                current_segment = {}
                # Extract START and END
                start_match = re.search(r'START: Page (\d+)', line)
                end_match = re.search(r'END: Page (\d+)', line)
                if start_match:
                    current_segment['START'] = int(start_match.group(1))
                if end_match:
                    current_segment['END'] = int(end_match.group(1))
            elif line.startswith('SUMMARY:'):
                current_segment['SUMMARY'] = line.replace('SUMMARY:', '').strip()
            elif line.startswith('REASONING:'):
                current_segment['REASONING'] = line.replace('REASONING:', '').strip()
        
        # Add the last segment
        if current_segment:
            segments.append(current_segment)
        
        # Add name field to each segment and limit to 7
        for i, segment in enumerate(segments[:7], 1):
            segment['name'] = f"segment_{i}"
        
        return segments[:7]  # Return maximum 7 segments
    
    def load_story_content(self, story_dir: str) -> str:
        """Load story content from JSON files, maintaining page order."""
        story_content = []
        story_path = Path(story_dir)
        
        if not story_path.exists():
            raise FileNotFoundError(f"Story directory not found: {story_dir}")
        
        # Get all JSON files and sort by page number
        json_files = [f for f in story_path.iterdir() if f.suffix == '.json']
        json_files.sort(key=lambda f: int(f.stem.split('_')[-1]))
        
        for file_path in json_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                page_content = json.load(f)
                text = page_content.get('text', '')
                
                # Handle both string and list text formats
                if isinstance(text, list):
                    text = ' '.join(text)
                
                story_content.append(text)
        
        # Combine with page markers
        return ' '.join(f"[Page {i+1}] {text}" for i, text in enumerate(story_content))
    
    def evaluate_moral_quality(self, prediction, target=None):
        """Optional: Evaluation metric for optimization."""
        # This could be used with DSPy optimizers to improve quality
        # For now, return a simple score based on output completeness
        if prediction.moral and prediction.segments:
            return 1.0
        return 0.0

def main():
    """Main function to process multiple storybooks."""
    
    generator = StoryMoralGeneratorDSPy()
    
    # Configuration paths
    asset_path = "/Users/mariyamohiuddin/Desktop/interactive-storybook-assets/qna_json/"
    output_path = "/Users/mariyamohiuddin/Desktop/Outputs/Dspy Outputs"
    
    # Create output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Storybook collections
    educare_storybooks = [
        "ada_twist_scientist", "a_letter_to_amy", "boxitects", "grumpy_monkey", 
        "if_you_give_a_mouse_a_cookie", "jabari_jumps", "last_stop_on_market_street", 
        "mango_abuela_and_me", "peters_chair", "stand_tall_molly_lou_melon", 
        "the_proudest_blue"
    ]
    
    kipp_storybooks = [
        "chicka_chicka_boom_boom", "the_little_red_hen-muldrow", 
        "rap_a_tap_tap_heres_bojangles", "three_little_pigs-marshall", 
        "the_three_billy_goats_gruff-finch", "ganeshas_sweet_tooth",
        "moon_rope", "the_story_of_ferdinand", "helpers_in_my_community", 
        "a_day_in_the_life_of_a_firefighter", "my_five_senses-aliki"
    ]
    
    stories = sorted(kipp_storybooks + educare_storybooks)
    
    # Process each story
    for story in stories:
        print(f"\nProcessing story: {story}")
        output_file = Path(output_path) / f"{story}_moral_segments_dspy.json"
        
        # Skip if output file already exists
        if output_file.exists():
            print(f"Output file already exists: {output_file}")
            continue
        
        story_dir = Path(asset_path) / story
        
        if not story_dir.exists():
            print(f"Story directory not found: {story_dir}")
            continue
        
        try:
            # Load story content
            story_content = generator.load_story_content(str(story_dir))
            
            # Generate moral and segments using DSPy
            result = generator.generate_story_moral(story_content)
            
            if result:
                # Save results
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                print(f"Saved moral and segments to: {output_file}")
            else:
                print(f"Failed to generate moral for story: {story}")
                
        except Exception as e:
            print(f"Error processing {story}: {e}")
            continue

if __name__ == "__main__":
    main()
