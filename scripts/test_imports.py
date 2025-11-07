#!/usr/bin/env python3
"""
Simple test to check if basic imports work.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic imports one by one."""
    print("🧪 Testing basic imports...")
    
    try:
        print("  Testing FSM imports...")
        from src.fsm.moore_machine import MooreMachine, MooreMachineGenerator
        print("  ✅ FSM imports successful")
        
        # Test FSM generation
        generator = MooreMachineGenerator(seed=42)
        fsm = generator.generate()
        print(f"  ✅ FSM generation successful: {fsm.num_states} states, {fsm.num_actions} actions")
        
    except Exception as e:
        print(f"  ❌ FSM import failed: {e}")
        return False
    
    try:
        print("  Testing model imports...")
        from src.training.models import SimpleTransformer, TransformerConfig
        print("  ✅ Model imports successful")
        
        # Test model creation
        config = TransformerConfig(vocab_size=50, num_layers=2)
        model = SimpleTransformer(config)
        print(f"  ✅ Model creation successful: {model.count_parameters():,} parameters")
        
    except Exception as e:
        print(f"  ❌ Model import failed: {e}")
        return False
    
    try:
        print("  Testing dataset imports...")
        from src.training.dataset import MooreMachineDataset
        print("  ✅ Dataset imports successful")
        
    except Exception as e:
        print(f"  ❌ Dataset import failed: {e}")
        return False
    
    print("\n🎉 All basic imports successful!")
    return True

if __name__ == '__main__':
    success = test_basic_imports()
    sys.exit(0 if success else 1)