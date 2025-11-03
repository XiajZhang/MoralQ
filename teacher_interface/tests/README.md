# Test Suite Directory Structure

This directory contains comprehensive test suites for the Teacher Interface system.

## Directory Structure

```
tests/
├── test_scripts/          # All test scripts
│   ├── test_phase_a.py    # Phase A: Moral and Question Generation Quality
│   ├── list_stories.py     # Helper script to list available stories
│   └── README.md           # Detailed test documentation
└── test_results/           # Test results and reports
    └── phase_a_results.json # Phase A test results (generated after running)
```

## Quick Start

1. **List available stories** (to verify story IDs):
   ```bash
   cd teacher_interface/tests/test_scripts
   python3 list_stories.py
   ```

2. **Run Phase A tests**:
   ```bash
   cd teacher_interface/tests/test_scripts
   python3 test_phase_a.py
   ```

3. **View results**:
   - Results are saved to `../test_results/phase_a_results.json`

## Test Phases

- **Phase A**: Moral and Question Generation Quality ✓
- Phase B: (To be implemented)
- Phase C: (To be implemented)
- Phase D: (To be implemented)
- Phase E: (To be implemented)

For detailed documentation on each phase, see `test_scripts/README.md`.
