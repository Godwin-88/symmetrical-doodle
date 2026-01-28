#!/usr/bin/env python3
"""
Test enhanced monitoring and diagnostic system
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nautilus-integration', 'src'))

async def test_monitoring_system():
    try:
        from nautilus_integration.core.monitoring import (
            setup_monitoring,
            get_system_diagnostics,
            DiagnosticLevel,
            get_resolution_guidance,
            execute_automated_fix
        )
        
        print("✓ Successfully imported enhanced monitoring components")
        
        # Setup monitoring system
        monitor = setup_monitoring()
        print("✓ Monitoring system setup completed")
        
        # Test diagnostic system
        diagnostics = await get_system_diagnostics(DiagnosticLevel.BASIC)
        print(f"✓ Retrieved {len(diagnostics)} diagnostic results")
        
        # Test resolution guidance
        guidance = await get_resolution_guidance("nautilus_engine_high_latency")
        if guidance:
            print(f"✓ Retrieved resolution guidance: {guidance.title}")
        else:
            print("✓ No resolution guidance found (expected for clean system)")
        
        print("✓ Enhanced monitoring and diagnostic system is working!")
        
    except Exception as e:
        print(f"✗ Error testing monitoring system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_monitoring_system())