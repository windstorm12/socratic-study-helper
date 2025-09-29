#!/usr/bin/env python3
"""
Script to verify Railway deployment is working correctly
"""

import requests
import json
import time

def verify_deployment():
    BASE_URL = "https://socratic-study-helper-copy-production.up.railway.app"
    
    print("=== Railway Deployment Verification ===")
    print(f"Testing: {BASE_URL}")
    print()
    
    # Test 1: Health check
    print("1. Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
        if response.status_code == 200:
            print("   ✅ Health check passed")
        else:
            print("   ❌ Health check failed")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()
    
    # Test 2: Debug endpoint
    print("2. Debug Info...")
    try:
        response = requests.get(f"{BASE_URL}/debug", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Port: {data.get('port')}")
            print(f"   Gemini API Key Present: {data.get('gemini_api_key_present')}")
            print(f"   Secret Key Present: {data.get('secret_key_present')}")
            print(f"   Frontend Origins: {data.get('frontend_origins')}")
            print("   ✅ Debug info retrieved")
        else:
            print("   ❌ Debug endpoint failed")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()
    
    # Test 3: Test endpoint
    print("3. Test Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/test", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   CORS Working: {data.get('cors_working')}")
            print("   ✅ Test endpoint working")
        else:
            print("   ❌ Test endpoint failed")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()
    
    # Test 4: OPTIONS preflight
    print("4. CORS Preflight Test...")
    try:
        response = requests.options(f"{BASE_URL}/set_subject", 
                                  headers={'Origin': 'http://localhost:3000'},
                                  timeout=10)
        print(f"   Status: {response.status_code}")
        cors_headers = {k: v for k, v in response.headers.items() if 'access-control' in k.lower()}
        print(f"   CORS Headers: {cors_headers}")
        if response.status_code == 200:
            print("   ✅ CORS preflight working")
        else:
            print("   ❌ CORS preflight failed")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()
    
    # Test 5: Login flow
    print("5. Login Flow Test...")
    try:
        session = requests.Session()
        login_response = session.post(f"{BASE_URL}/login",
                                    json={"username": "Avnish", "password": "Nerd"},
                                    headers={'Content-Type': 'application/json'},
                                    timeout=10)
        print(f"   Login Status: {login_response.status_code}")
        
        if login_response.status_code == 200:
            print("   ✅ Login successful")
            
            # Test set_subject
            subject_response = session.post(f"{BASE_URL}/set_subject",
                                          json={"subject": "Math"},
                                          headers={'Content-Type': 'application/json'},
                                          timeout=10)
            print(f"   Set Subject Status: {subject_response.status_code}")
            if subject_response.status_code == 200:
                print("   ✅ Set subject working")
            else:
                print(f"   ❌ Set subject failed: {subject_response.text}")
        else:
            print(f"   ❌ Login failed: {login_response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()
    
    print("=== Verification Complete ===")

if __name__ == "__main__":
    verify_deployment()
