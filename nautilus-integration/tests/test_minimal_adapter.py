"""
Minimal test to verify basic adapter structure.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_basic_import():
    """Test basic import without full functionality."""
    try:
        # Test individual components first
        from nautilus_integration.services.data_catalog_adapter import (
            DataMigrationType,
            DataQualityStatus
        )
        
        print("✓ Enums imported successfully")
        
        from nautilus_integration.services.data_catalog_adapter import (
            DataMigrationConfig,
            DataValidationResult,
            MigrationProgress
        )
        
        print("✓ Data classes imported successfully")
        
        # Test the main class
        from nautilus_integration.services.data_catalog_adapter import DataCatalogAdapter
        
        print("✓ DataCatalogAdapter imported successfully")
        
        # Test basic instantiation
        config = DataMigrationConfig(
            source_schema="test",
            target_path="./test"
        )
        
        adapter = DataCatalogAdapter(config)
        print("✓ DataCatalogAdapter instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_import()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)