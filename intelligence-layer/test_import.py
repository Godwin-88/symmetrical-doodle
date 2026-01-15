try:
    import intelligence_layer.experiment_config
    print("Module imported successfully")
    print(dir(intelligence_layer.experiment_config))
except Exception as e:
    print(f"Error importing module: {e}")
    import traceback
    traceback.print_exc()