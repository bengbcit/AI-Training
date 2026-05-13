# prompt_optimizer.py
# A tool to optimize prompts using various free AI APIs
# Usage: python prompt_optimizer.py

import os
import requests
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# ==================== DeepSeek Version ====================


def test_deepseek_api() -> bool:
    """Test if DeepSeek API key is working"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ DEEPSEEK_API_KEY not found in .env")
        return False

    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Say 'API works'"}],
        "max_tokens": 20
    }

    try:
        response = requests.post(url, headers=headers,
                                 json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ DeepSeek API key is valid")
            return True
        else:
            print(f"❌ DeepSeek API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ DeepSeek connection error: {e}")
        return False


def optimize_prompt_deepseek(bad_prompt: str) -> str:
    """Optimize a prompt using DeepSeek API"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "Error: Please set DEEPSEEK_API_KEY in .env file"

    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}

    system_prompt = """You are a Prompt Engineering expert.
Improve the user's prompt following these rules:
1. Make it more specific and clear
2. Add necessary context
3. Specify output format
4. Remove ambiguity

Return format:
Original: [original prompt]
Improved: [improved prompt]
Explanation: [why these changes help]"""

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please optimize this prompt:\n{bad_prompt}"}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()

        # Error handling for non-200 responses
        if "choices" not in result:
            return f"DeepSeek API Error: {result.get('error', {}).get('message', 'Unknown error')}"

        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"DeepSeek Exception: {e}"


# ==================== Gemini Version (UPDATED - Latest Model) ====================

def test_gemini_api() -> bool:
    """Test if Gemini API key is working with latest model"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env")
        return False

    # Latest Gemini model as of 2026
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "parts": [{"text": "Say 'API works'"}]
        }]
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ Gemini API key is valid (gemini-2.5-flash)")
            return True
        else:
            print(
                f"❌ Gemini API error ({response.status_code}): {response.text[:100]}")
            return False
    except Exception as e:
        print(f"❌ Gemini connection error: {e}")
        return False


def optimize_prompt_gemini(bad_prompt: str) -> str:
    """Optimize a prompt using Google Gemini 2.5 Flash API"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: Please set GEMINI_API_KEY in .env file"

    # Latest Gemini 2.5 Flash model (as of 2026)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    system_prompt = """You are a Prompt Engineering expert.
Return format:
Original: [original prompt]
Improved: [improved prompt]
Explanation: [why these changes help]"""

    full_prompt = f"{system_prompt}\n\nPlease optimize this prompt:\n{bad_prompt}"

    payload = {
        "contents": [{
            "parts": [{"text": full_prompt}]
        }]
    }

    try:
        response = requests.post(url, json=payload)
        result = response.json()

        # Error handling
        if "candidates" not in result:
            error_msg = result.get('error', {}).get('message', 'Unknown error')
            return f"Gemini API Error: {error_msg}"

        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Gemini Exception: {e}"


def list_gemini_models() -> list:
    """List all available Gemini models for your API key"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return []

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [m["name"].replace("models/", "") for m in models]
        return []
    except Exception:
        return []


# ==================== Groq Version ====================

def test_groq_api() -> bool:
    """Test if Groq API key is working"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in .env")
        return False

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "Say 'API works'"}],
        "max_tokens": 20
    }

    try:
        response = requests.post(url, headers=headers,
                                 json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ Groq API key is valid")
            return True
        else:
            print(f"❌ Groq API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Groq connection error: {e}")
        return False


def optimize_prompt_groq(bad_prompt: str) -> str:
    """Optimize a prompt using Groq API (Llama 3.3)"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: Please set GROQ_API_KEY in .env file"

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}

    system_prompt = """You are a Prompt Engineering expert.
Return format:
Original: [original prompt]
Improved: [improved prompt]
Explanation: [why these changes help]"""

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please optimize this prompt:\n{bad_prompt}"}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()

        if "choices" not in result:
            return f"Groq API Error: {result.get('error', {}).get('message', 'Unknown error')}"

        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Groq Exception: {e}"


# ==================== NVIDIA NIM Version ====================

def test_nvidia_api() -> bool:
    """Test if NVIDIA API key is working"""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ NVIDIA_API_KEY not found in .env")
        return False

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": [{"role": "user", "content": "Say 'API works'"}],
        "max_tokens": 20
    }

    try:
        response = requests.post(url, headers=headers,
                                 json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ NVIDIA API key is valid")
            return True
        else:
            print(f"❌ NVIDIA API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ NVIDIA connection error: {e}")
        return False


def optimize_prompt_nvidia(bad_prompt: str) -> str:
    """Optimize a prompt using NVIDIA NIM API"""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return "Error: Please set NVIDIA_API_KEY in .env file"

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}

    system_prompt = """You are a Prompt Engineering expert.
Return format:
Original: [original prompt]
Improved: [improved prompt]
Explanation: [why these changes help]"""

    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please optimize this prompt:\n{bad_prompt}"}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()

        if "choices" not in result:
            return f"NVIDIA API Error: {result.get('error', {}).get('message', 'Unknown error')}"

        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"NVIDIA Exception: {e}"


# ==================== Main ====================

def test_all_apis():
    """Test all configured API keys"""
    print("\n" + "="*50)
    print("TESTING ALL API KEYS")
    print("="*50)

    test_deepseek_api()
    test_gemini_api()
    test_groq_api()
    test_nvidia_api()

    # Show available Gemini models if key exists
    if os.getenv("GEMINI_API_KEY"):
        print("\n🔍 Available Gemini models:")
        models = list_gemini_models()
        if models:
            for model in models[:10]:  # Show first 10
                print(f"   - {model}")
        else:
            print("   Could not fetch models (check API key)")

    print("="*50 + "\n")


if __name__ == "__main__":
    # Test all API keys first
    test_all_apis()

    # Get the bad prompt
    bad_prompt = "tell me about AI"

    print(f"Original prompt: '{bad_prompt}'")
    print("="*60)
    print("📝 RUNNING OPTIMIZATION WITH GROQ (most reliable):")
    print("-"*60)

    # Use Groq (it's working reliably)
    result = optimize_prompt_groq(bad_prompt)
    print(result)

    # Optional: Also try Gemini if you want to test it
    print("\n" + "="*60)
    print("📝 TRYING GEMINI 2.5 FLASH:")
    print("-"*60)
    gemini_result = optimize_prompt_gemini(bad_prompt)
    print(gemini_result)

    # Optional: Also try NVIDIA if you want to test it
    print("\n" + "="*60)
    print("📝 TRYING NVIDIA NIM:")
    print("-"*60)
    nvidia_result = optimize_prompt_nvidia(bad_prompt)
    print(nvidia_result)
