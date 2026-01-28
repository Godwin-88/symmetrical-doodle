#!/usr/bin/env python3

# Test minimal error handling file
from enum import Enum

class NautilusErrorType(Enum):
    """Types of errors specific to NautilusTrader integration"""
    NAUTILUS_ENGINE_ERROR = "nautilus_engine_error"
    NETWORK_ERROR = "network_error"

print("Minimal test successful")
print(f"NautilusErrorType: {NautilusErrorType.NAUTILUS_ENGINE_ERROR}")