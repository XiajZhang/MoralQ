"""
Helper script to list available stories in the assets directory.
Use this to find the correct story IDs for test cases.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(env_path)

ASSETS_PATH = os.getenv('ASSETS_PATH', '/path/to/interactive-storybook-assets')
QNA_JSON_PATH = os.path.join(ASSETS_PATH, "qna_json")

print(f"\nLooking for stories in: {QNA_JSON_PATH}\n")

if not os.path.exists(QNA_JSON_PATH):
    print(f"ERROR: Directory not found: {QNA_JSON_PATH}")
    print(f"Please check your ASSETS_PATH in .env file")
    sys.exit(1)

stories = []
for item in os.listdir(QNA_JSON_PATH):
    item_path = os.path.join(QNA_JSON_PATH, item)
    if os.path.isdir(item_path):
        json_files = [f for f in os.listdir(item_path) if f.endswith('.json')]
        stories.append({
            "id": item,
            "title": item.replace("_", " ").title(),
            "pages": len(json_files)
        })

if stories:
    print(f"Found {len(stories)} stories:\n")
    print(f"{'Story ID':<40} {'Title':<40} {'Pages':<10}")
    print("-" * 90)
    for story in sorted(stories, key=lambda x: x["title"]):
        print(f"{story['id']:<40} {story['title']:<40} {story['pages']:<10}")
else:
    print("No stories found in the directory")

print()

