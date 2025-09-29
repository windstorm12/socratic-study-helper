#!/usr/bin/env python3
"""
Test script to verify CORS and 404 fixes
"""

import requests
import json

def test_cors_and_endpoints():
    # Update this URL to match your deployment
    BASE_URL = "https://socratic-study-helper-copy-production.up.railway.app"
    
    print("=== Testing CORS and Endpoint Fixes ===")
    print(f"Testing against: {BASE_URL}")
    print()
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Test 2: Test endpoint (new)
    print("2. Testing new test endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/test")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Test 3: OPTIONS preflight for set_subject
    print("3. Testing OPTIONS preflight for set_subject...")
    try:
        response = requests.options(f"{BASE_URL}/set_subject", 
                                  headers={'Origin': 'http://localhost:3000'})
        print(f"   Status: {response.status_code}")
        print(f"   CORS Headers: {dict(response.headers)}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Test 4: POST to set_subject (should fail with 401 - no auth)
    print("4. Testing POST to set_subject (expecting 401 - no auth)...")
    try:
        response = requests.post(f"{BASE_URL}/set_subject",
                               json={"subject": "Math"},
                               headers={'Content-Type': 'application/json'})
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Test 5: Login test
    print("5. Testing login endpoint...")
    try:
        response = requests.post(f"{BASE_URL}/login",
                               json={"username": "Avnish", "password": "Nerd"},
                               headers={'Content-Type': 'application/json'})
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
        
        if response.status_code == 200:
            print("   ✅ Login successful!")
            # Test set_subject with session
            session = requests.Session()
            login_resp = session.post(f"{BASE_URL}/login",
                                    json={"username": "Avnish", "password": "Nerd"},
                                    headers={'Content-Type': 'application/json'})
            if login_resp.status_code == 200:
                print("6. Testing set_subject with authentication...")
                subject_resp = session.post(f"{BASE_URL}/set_subject",
                                          json={"subject": "Math"},
                                          headers={'Content-Type': 'application/json'})
                print(f"   Status: {subject_resp.status_code}")
                print(f"   Response: {subject_resp.text}")
        else:
            print("   ❌ Login failed")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    print("=== Test Complete ===")

if __name__ == "__main__":
    test_cors_and_endpoints()
