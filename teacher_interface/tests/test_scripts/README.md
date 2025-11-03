# Test Suite for Teacher Interface

This directory contains comprehensive test suites organized by phases.

## Test Phases

- **Phase A**: Moral and Question Generation Quality
- Phase B: (To be implemented)
- Phase C: (To be implemented)
- Phase D: (To be implemented)
- Phase E: (To be implemented)

## Running Tests

### Phase A: Moral and Question Generation Quality

```bash
cd teacher_interface/tests/test_scripts
python3 test_phase_a.py
```

**Requirements:**
- `.env` file must be configured with `OPENAI_API_KEY` and `ASSETS_PATH`
- Story assets must be available in the `ASSETS_PATH/qna_json/` directory

**Test Cases:**
- A1: A Letter to Amy - Empathy
- A2: Grumpy Monkey - Emotional regulation
- A3: Last Stop on Market Street - Gratitude
- A4: If You Give a Mouse a Cookie - Consequence awareness
- A5: Ada Twist, Scientist - Curiosity

**Output:**
- Results are saved to `../test_results/phase_a_results.json`
- Comprehensive JSON report includes:
  - Generated morals and questions
  - Evaluation scores (Relevance, Appropriateness, Clarity)
  - Comparison with expected outcomes
  - Overall test statistics

## Story ID Mapping

The test script expects story IDs to match folder names in `ASSETS_PATH/qna_json/`. Common formats:
- `a_letter_to_amy`
- `grumpy_monkey`
- `ada_twist_scientist`
- `if_you_give_a_mouse_a_cookie`

If story IDs don't match, update the `TEST_CASES` dictionary in `test_phase_a.py` with the correct folder names.

## Evaluation Metrics

1. **Relevance**: How well the output aligns with expected themes
2. **Appropriateness**: Age-appropriate language and concepts for 4-6 year olds
3. **Clarity**: Simple syntax and clear communication

## Test Results Format

Each test produces a comprehensive JSON report with:
- Test metadata (ID, story, objective, timestamp)
- Expected outcomes
- Generated results (moral, segments, questions)
- Evaluation scores and analysis
- Pass/fail status
- Overall statistics across all tests

