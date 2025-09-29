#!/usr/bin/env python3
"""
Test script for Google Gemini API
This script allows you to test if your Gemini API key is working
and if you have remaining credits.
"""

import os
from google import genai

def test_gemini_api():
    print("=== Google Gemini API Test ===")
    print()
    
    # Get API key from user input
    api_key = input("Enter your Gemini API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return
    
    try:
        # Initialize the client
        print("🔧 Initializing Gemini client...")
        client = genai.Client(api_key=api_key)
        print("✅ Client initialized successfully")
        
        # List available models
        print("\n🔍 Listing available models...")
        try:
            models = client.models.list()
            print("📋 Available models:")
            for model in models:
                print(f"  - {model.name}")
        except Exception as e:
            print(f"⚠️  Could not list models: {e}")
        
        # Test with a simple prompt
        print("\n📝 Testing with a simple prompt...")
        test_prompt = "Hello! Can you respond with 'API is working' if you can read this message?"
        
        print(f"Prompt: {test_prompt}")
        print("⏳ Sending request to Gemini...")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=test_prompt
        )
        
        print("✅ API request successful!")
        print(f"📤 Response: {response.text}")
        
        # Test with a more complex prompt
        print("\n" + "="*50)
        print("🧪 Testing with a more complex prompt...")
        complex_prompt = input("\nEnter a test prompt (or press Enter for default): ").strip()
        
        if not complex_prompt:
            complex_prompt = "Explain what Python is in 2 sentences."
        
        print(f"\nPrompt: {complex_prompt}")
        print("⏳ Sending request to Gemini...")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=complex_prompt
        )
        
        print("✅ Complex request successful!")
        print(f"📤 Response: {response.text}")
        
        # Interactive mode
        print("\n" + "="*50)
        print("🎯 Interactive mode - Enter prompts to test (type 'quit' to exit)")
        print()
        
        while True:
            user_prompt = input("You: ").strip()
            
            if user_prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_prompt:
                print("Please enter a prompt.")
                continue
            
            try:
                print("⏳ Thinking...")
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_prompt
                )
                print(f"Gemini: {response.text}")
                print()
                
            except Exception as e:
                print(f"❌ Error: {e}")
                print()
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("\nThis could indicate:")
        print("- Invalid API key")
        print("- API quota exceeded")
        print("- Network connectivity issues")
        print("- API service unavailable")
        
        # Try to provide more specific error information
        error_str = str(e).lower()
        if "quota" in error_str or "limit" in error_str:
            print("\n🚨 QUOTA ISSUE DETECTED: You may have exceeded your free tier limits")
        elif "api" in error_str and "key" in error_str:
            print("\n🔑 API KEY ISSUE: Check if your API key is correct")
        elif "network" in error_str or "connection" in error_str:
            print("\n🌐 NETWORK ISSUE: Check your internet connection")

if __name__ == "__main__":
    test_gemini_api()
