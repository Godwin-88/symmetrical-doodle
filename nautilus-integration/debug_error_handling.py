#!/usr/bin/env python3

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("Debugging error_handling module...")

try:
    # Try to execute the module step by step
    print("1. Importing required modules...")
    
    import asyncio
    import time
    import random
    import json
    import traceback
    from pathlib import Path
    from typing import Dict, Any, Optional, List, Callable, Union, TypeVar, Generic, Type
    from dataclasses import dataclass, field, asdict
    from enum import Enum
    from datetime import datetime, timedelta
    from contextlib import asynccontextmanager
    import pickle
    import hashlib
    import uuid
    
    print("   Basic imports: OK")
    
    import structlog
    print("   Structlog: OK")
    
    from nautilus_trader.core.nautilus_pyo3 import LogLevel
    print("   NautilusTrader: OK")
    
    from nautilus_integration.core.logging import get_logger, set_correlation_id, get_correlation_id
    print("   Logging imports: OK")
    
    print("2. Testing logger creation...")
    logger = get_logger(__name__)
    print("   Logger created: OK")
    
    print("3. Testing TypeVar...")
    T = TypeVar('T')
    print("   TypeVar: OK")
    
    print("4. Now importing the actual module...")
    exec(open('src/nautilus_integration/core/error_handling.py').read())
    print("   Module executed successfully")
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()