#!/usr/bin/env python3
"""
Quick test script to verify Markets API endpoints.
Run this after starting the Intelligence Layer backend.
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_endpoint(name, url, params=None):
    """Test an API endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    if params:
        print(f"Params: {params}")
    print('='*60)
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"✅ SUCCESS - Status: {response.status_code}")
        print(f"Response preview:")
        print(json.dumps(data, indent=2)[:500] + "...")
        return True
    except requests.exceptions.ConnectionError:
        print(f"❌ FAILED - Connection refused. Is the backend running?")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ FAILED - Request timeout")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"❌ FAILED - HTTP Error: {e}")
        print(f"Response: {response.text[:200]}")
        return False
    except Exception as e:
        print(f"❌ FAILED - Error: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Markets API Test Suite")
    print("="*60)
    
    # Test health check first
    if not test_endpoint("Health Check", f"{BASE_URL}/health"):
        print("\n⚠️  Backend is not running. Please start it with:")
        print("   cd intelligence-layer")
        print("   python -m intelligence_layer.main")
        return
    
    # Test Markets endpoints
    assets = "EURUSD,GBPUSD,USDJPY"
    
    tests = [
        ("Live Market Data", f"{BASE_URL}/markets/live-data", {"assets": assets}),
        ("Live Market Data with Depth", f"{BASE_URL}/markets/live-data", 
         {"assets": assets, "include_depth": "true"}),
        ("Correlation Matrix", f"{BASE_URL}/markets/correlations", 
         {"assets": assets, "window": "24H", "method": "pearson"}),
        ("Microstructure", f"{BASE_URL}/markets/microstructure", 
         {"asset_id": "EURUSD"}),
        ("Liquidity Analysis", f"{BASE_URL}/markets/liquidity", 
         {"assets": assets}),
        ("Market Events", f"{BASE_URL}/markets/events", 
         {"severity_min": "0.5"}),
    ]
    
    results = []
    for name, url, params in tests:
        results.append(test_endpoint(name, url, params))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

if __name__ == "__main__":
    main()
