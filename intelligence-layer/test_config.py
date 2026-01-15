import sys
sys.path.insert(0, 'src')

try:
    from intelligence_layer.config import load_config
    print("Config import successful!")
    config = load_config()
    print("Config loaded successfully!")
except Exception as e:
    print(f"Config error: {e}")
    import traceback
    traceback.print_exc()