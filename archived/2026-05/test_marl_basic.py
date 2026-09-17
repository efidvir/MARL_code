#!/usr/bin/env python3
"""
Basic test for MARL agent creation and imports.
"""

import sys
from pathlib import Path

# Add the project root to Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def test_marl_imports():
    """Test that MARL imports work."""
    try:
        from sixg_sim.agent import RLAgent, CentralizedMARLTrainer, create_marl_agent_for_node
        from sixg_sim.traffic import SliceDictionary
        print("SUCCESS: MARL imports successful")
        return True
    except Exception as e:
        print(f"FAILED: MARL import failed: {e}")
        return False

def test_agent_creation():
    """Test that RL agents can be created."""
    try:
        from sixg_sim.agent import create_marl_agent_for_node
        from sixg_sim.traffic import SliceDictionary

        # Create slice dictionary
        slice_dict = SliceDictionary()

        # Create RL agent
        agent = create_marl_agent_for_node("test_node", slice_dict, use_rl=True)

        if agent.__class__.__name__ == "RLAgent":
            print("SUCCESS: RL agent creation successful")
            return True
        else:
            print(f"FAILED: Wrong agent type: {agent.__class__.__name__}")
            return False

    except Exception as e:
        print(f"FAILED: RL agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_heuristic_fallback():
    """Test that heuristic agents still work."""
    try:
        from sixg_sim.agent import create_marl_agent_for_node
        from sixg_sim.traffic import SliceDictionary

        # Create slice dictionary
        slice_dict = SliceDictionary()

        # Create heuristic agent
        agent = create_marl_agent_for_node("test_node", slice_dict, use_rl=False)

        if agent.__class__.__name__ == "HeuristicAgent":
            print("SUCCESS: Heuristic agent creation successful")
            return True
        else:
            print(f"FAILED: Wrong agent type: {agent.__class__.__name__}")
            return False

    except Exception as e:
        print(f"FAILED: Heuristic agent creation failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing MARL implementation...")

    tests = [
        test_marl_imports,
        test_agent_creation,
        test_heuristic_fallback,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1

    print(f"\nResults: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("SUCCESS: All MARL tests PASSED")
        sys.exit(0)
    else:
        print("FAILED: Some MARL tests FAILED")
        sys.exit(1)