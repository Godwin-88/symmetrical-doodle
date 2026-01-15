import sys
sys.path.insert(0, 'src')

try:
    from intelligence_layer.experiment_config import ExperimentConfig, ExperimentManager
    print("Import successful!")
    
    # Test basic functionality
    config = ExperimentConfig(name="test")
    print(f"Created config: {config.name}")
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Other error: {e}")
    import traceback
    traceback.print_exc()