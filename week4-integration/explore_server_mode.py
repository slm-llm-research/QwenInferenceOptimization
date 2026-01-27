"""
Week 4: Explore vLLM Server Mode

This script demonstrates how to programmatically start and interact with
vLLM's OpenAI-compatible API server. This prepares you for Week 6 deployment.

Usage:
    python explore_server_mode.py
"""

import subprocess
import time
import requests
import sys
import signal
import os


def start_vllm_server():
    """Start vLLM server in background"""
    
    print("="*70)
    print("Week 4: Exploring vLLM Server Mode")
    print("="*70)
    print()
    print("🚀 Starting vLLM Server...")
    print()
    
    # Server configuration
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    port = 8000
    host = "127.0.0.1"
    
    # Start server process
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", host,
        "--port", str(port),
        "--trust-remote-code",
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    print("⏳ Server is starting (this takes 30-60 seconds)...")
    print("   Press Ctrl+C to stop the server and exit")
    print()
    
    try:
        # Start server process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Wait for server to be ready
        print("📝 Server logs:")
        print("-" * 70)
        
        server_ready = False
        for line in process.stdout:
            print(f"   {line}", end='')
            
            # Check if server is ready
            if "Uvicorn running on" in line or "Application startup complete" in line:
                server_ready = True
                break
        
        if not server_ready:
            print("\n⚠️  Server didn't start properly")
            return None, None
        
        print("-" * 70)
        print("\n✅ Server is running!")
        print()
        
        # Test the server
        print("🧪 Testing server with sample requests...")
        print()
        
        test_server(host, port)
        
        # Keep server running
        print("\n" + "="*70)
        print("✅ Server exploration complete!")
        print("="*70)
        print()
        print(f"📡 Server is accessible at: http://{host}:{port}")
        print()
        print("You can test it manually:")
        print(f'  curl http://{host}:{port}/v1/completions \\')
        print("    -H 'Content-Type: application/json' \\")
        print("    -d '{")
        print('      "model": "Qwen/Qwen2.5-7B-Instruct",')
        print('      "prompt": "Hello, world!",')
        print('      "max_tokens": 20')
        print("    }'")
        print()
        print("Press Ctrl+C to stop the server...")
        
        # Wait for user interrupt
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping server...")
        process.terminate()
        process.wait()
        print("✅ Server stopped")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if 'process' in locals():
            process.terminate()
    
    return process


def test_server(host, port):
    """Test the server with sample requests"""
    
    base_url = f"http://{host}:{port}"
    
    # Test 1: Health check (models endpoint)
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ Test 1: Server health check passed")
            models = response.json()
            print(f"   Available models: {[m['id'] for m in models['data']]}")
        else:
            print(f"⚠️  Test 1: Unexpected status {response.status_code}")
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    print()
    
    # Test 2: Completion request
    try:
        completion_data = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "prompt": "The capital of France is",
            "max_tokens": 10,
            "temperature": 0.0,
        }
        
        print("📝 Test 2: Sending completion request...")
        print(f"   Prompt: '{completion_data['prompt']}'")
        
        response = requests.post(
            f"{base_url}/v1/completions",
            json=completion_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result['choices'][0]['text']
            print(f"✅ Test 2: Completion successful")
            print(f"   Generated: '{generated_text}'")
        else:
            print(f"⚠️  Test 2: Status {response.status_code}")
            print(f"   Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
    
    print()
    
    # Test 3: Chat completion (if supported)
    try:
        chat_data = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [
                {"role": "user", "content": "What is AI?"}
            ],
            "max_tokens": 30,
        }
        
        print("💬 Test 3: Sending chat completion request...")
        
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=chat_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            reply = result['choices'][0]['message']['content']
            print(f"✅ Test 3: Chat completion successful")
            print(f"   Reply: '{reply[:100]}...'")
        else:
            print(f"⚠️  Test 3: Status {response.status_code}")
    
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")


def main():
    """Main function"""
    
    print("\n🔧 vLLM Server Mode")
    print()
    print("This demonstrates vLLM's OpenAI-compatible API server.")
    print("The server will:")
    print("  • Load the Qwen2.5-7B-Instruct model")
    print("  • Start an HTTP API on port 8000")
    print("  • Accept requests in OpenAI format")
    print()
    
    response = input("Ready to start server? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled")
        return
    
    print()
    start_vllm_server()


if __name__ == "__main__":
    main()

