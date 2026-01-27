"""
Week 7: Automated Load Testing Script

This script runs automated load tests against your deployed vLLM endpoint.
It simulates concurrent users and reports performance metrics.

Usage:
    python run_load_test.py --endpoint http://YOUR_ENDPOINT --users 50 --duration 300
"""

import argparse
import time
import requests
import statistics
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


class LoadTester:
    """Load testing class for vLLM endpoint"""
    
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url.rstrip('/')
        self.results = []
        self.errors = []
    
    def send_request(self, request_id: int, prompt: str) -> dict:
        """Send a single inference request"""
        start_time = time.time()
        
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.7,
        }
        
        try:
            response = requests.post(
                f"{self.endpoint_url}/v1/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['choices'][0]['text']
                tokens = len(generated_text.split())
                
                return {
                    "request_id": request_id,
                    "success": True,
                    "latency": elapsed,
                    "tokens": tokens,
                    "status_code": 200,
                }
            else:
                return {
                    "request_id": request_id,
                    "success": False,
                    "latency": elapsed,
                    "status_code": response.status_code,
                    "error": f"HTTP {response.status_code}",
                }
        
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            return {
                "request_id": request_id,
                "success": False,
                "latency": elapsed,
                "error": "Timeout",
            }
        
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "request_id": request_id,
                "success": False,
                "latency": elapsed,
                "error": str(e),
            }
    
    def run_load_test(self, num_users: int, duration: int):
        """
        Run load test with specified users for given duration.
        
        Args:
            num_users: Number of concurrent users
            duration: Test duration in seconds
        """
        
        print("="*70)
        print(f"Load Test Starting")
        print("="*70)
        print(f"Endpoint: {self.endpoint_url}")
        print(f"Concurrent users: {num_users}")
        print(f"Duration: {duration} seconds")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        print()
        
        # Sample prompts
        prompts = [
            "Explain machine learning:",
            "What is quantum computing?",
            "Describe neural networks:",
            "How does AI work?",
            "What is cloud computing?",
            "Explain blockchain:",
            "What is deep learning?",
            "Describe data science:",
        ]
        
        start_time = time.time()
        request_id = 0
        
        print("🚀 Sending requests...")
        print()
        
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = []
            
            # Keep sending requests until duration expires
            while time.time() - start_time < duration:
                # Submit up to num_users requests
                while len(futures) < num_users and time.time() - start_time < duration:
                    prompt = prompts[request_id % len(prompts)]
                    future = executor.submit(self.send_request, request_id, prompt)
                    futures.append(future)
                    request_id += 1
                
                # Check for completed futures
                done_futures = [f for f in futures if f.done()]
                for future in done_futures:
                    result = future.result()
                    
                    if result["success"]:
                        self.results.append(result)
                        print(f"✅ Request {result['request_id']}: {result['latency']:.2f}s ({result['tokens']} tokens)")
                    else:
                        self.errors.append(result)
                        print(f"❌ Request {result['request_id']}: {result.get('error', 'Unknown error')}")
                    
                    futures.remove(future)
                
                # Brief sleep to avoid hammering
                time.sleep(0.1)
            
            # Wait for remaining futures
            print("\n⏳ Waiting for remaining requests to complete...")
            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    self.results.append(result)
                else:
                    self.errors.append(result)
        
        total_time = time.time() - start_time
        
        # Print results
        self.print_results(total_time)
    
    def print_results(self, total_time: float):
        """Print formatted results"""
        
        print("\n" + "="*70)
        print("Load Test Results")
        print("="*70)
        print()
        
        total_requests = len(self.results) + len(self.errors)
        successful_requests = len(self.results)
        failed_requests = len(self.errors)
        
        print(f"📊 Requests:")
        print(f"   Total:      {total_requests}")
        print(f"   Successful: {successful_requests}")
        print(f"   Failed:     {failed_requests}")
        
        if total_requests > 0:
            error_rate = (failed_requests / total_requests) * 100
            print(f"   Error rate: {error_rate:.2f}%")
        
        print()
        
        if self.results:
            latencies = [r["latency"] for r in self.results]
            tokens = [r["tokens"] for r in self.results]
            
            print(f"⏱️  Latency (seconds):")
            print(f"   Mean:   {statistics.mean(latencies):.2f}")
            print(f"   Median: {statistics.median(latencies):.2f}")
            print(f"   P95:    {sorted(latencies)[int(len(latencies) * 0.95)]:.2f}")
            print(f"   P99:    {sorted(latencies)[int(len(latencies) * 0.99)]:.2f}")
            print(f"   Max:    {max(latencies):.2f}")
            print(f"   Min:    {min(latencies):.2f}")
            print()
            
            print(f"📈 Throughput:")
            rps = successful_requests / total_time
            total_tokens = sum(tokens)
            tokens_per_sec = total_tokens / total_time
            print(f"   Requests/sec: {rps:.2f}")
            print(f"   Tokens/sec:   {tokens_per_sec:.1f}")
            print(f"   Total tokens: {total_tokens}")
            print()
        
        # Recommendations
        print("💡 Analysis:")
        
        if not self.results:
            print("   ❌ No successful requests - check endpoint health")
        else:
            error_rate = (failed_requests / total_requests) * 100
            p95 = sorted(latencies)[int(len(latencies) * 0.95)]
            
            if error_rate < 1:
                print("   ✅ Error rate is excellent (<1%)")
            elif error_rate < 5:
                print("   ⚠️  Error rate is acceptable but could be improved")
            else:
                print(f"   ❌ Error rate is high ({error_rate:.1f}%) - system is overloaded")
            
            if p95 < 3:
                print("   ✅ P95 latency is good (<3s)")
            elif p95 < 5:
                print("   ⚠️  P95 latency is acceptable")
            else:
                print(f"   ❌ P95 latency is high ({p95:.1f}s) - consider scaling")
        
        print()
        
        # Save results
        results_file = f"load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "test_config": {
                    "endpoint": self.endpoint_url,
                    "total_time": total_time,
                },
                "summary": {
                    "total_requests": total_requests,
                    "successful": successful_requests,
                    "failed": failed_requests,
                    "error_rate": error_rate if total_requests > 0 else 0,
                },
                "results": self.results,
                "errors": self.errors,
            }, f, indent=2)
        
        print(f"📁 Results saved to: {results_file}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Load test vLLM endpoint")
    parser.add_argument("--endpoint", required=True, help="Endpoint URL (e.g., http://example.com)")
    parser.add_argument("--users", type=int, default=50, help="Number of concurrent users (default: 50)")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds (default: 300)")
    
    args = parser.parse_args()
    
    tester = LoadTester(args.endpoint)
    tester.run_load_test(args.users, args.duration)


if __name__ == "__main__":
    main()

