"""
Week 7: Locust Load Testing Configuration

This file defines the user behavior for load testing the vLLM endpoint.
Use with Locust for interactive load testing.

Usage:
    locust -f locustfile.py --host http://YOUR_ENDPOINT
    
    Then open http://localhost:8089 in browser
"""

from locust import HttpUser, task, between
import json


class QwenUser(HttpUser):
    """Simulates a user sending inference requests to vLLM"""
    
    # Wait 0.5-3 seconds between requests (simulates user think time)
    wait_time = between(0.5, 3.0)
    
    # Sample prompts to use
    prompts = [
        "Explain machine learning in simple terms:",
        "What is quantum computing?",
        "Describe how neural networks work:",
        "Write a short poem about AI:",
        "What are the benefits of cloud computing?",
        "Explain blockchain technology:",
        "How does natural language processing work?",
        "What is the future of artificial intelligence?",
        "Describe the scientific method:",
        "Explain the concept of big data:",
    ]
    
    def on_start(self):
        """Called when a simulated user starts"""
        self.prompt_index = 0
    
    @task(weight=10)
    def generate_completion(self):
        """
        Main task: Send a completion request
        Weight=10 means this is the primary action
        """
        # Get next prompt (cycle through list)
        prompt = self.prompts[self.prompt_index % len(self.prompts)]
        self.prompt_index += 1
        
        # Request payload
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.7,
        }
        
        # Send request
        with self.client.post(
            "/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="completion_request"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    generated_text = result['choices'][0]['text']
                    # Verify we got a response
                    if len(generated_text) > 0:
                        response.success()
                    else:
                        response.failure("Empty response")
                except Exception as e:
                    response.failure(f"Parse error: {e}")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(weight=2)
    def generate_short_completion(self):
        """
        Secondary task: Short completion (faster)
        Weight=2 means this happens less frequently
        """
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "prompt": "The capital of France is",
            "max_tokens": 10,
            "temperature": 0.0,
        }
        
        with self.client.post(
            "/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="short_completion"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(weight=1)
    def check_health(self):
        """
        Occasional health check
        Weight=1 means this is rare
        """
        with self.client.get(
            "/health",
            catch_response=True,
            name="health_check"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class HeavyUser(HttpUser):
    """
    Heavy user - sends longer requests less frequently
    Use this to simulate power users
    """
    
    wait_time = between(5.0, 10.0)  # Longer wait between requests
    
    @task
    def generate_long_completion(self):
        """Generate longer text"""
        prompt = "Write a detailed explanation of how transformer models work in deep learning, including attention mechanisms:"
        
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "prompt": prompt,
            "max_tokens": 200,  # Much longer
            "temperature": 0.8,
        }
        
        with self.client.post(
            "/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="long_completion"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

