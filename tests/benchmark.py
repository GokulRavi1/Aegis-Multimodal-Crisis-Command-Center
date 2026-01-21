"""
Benchmark Module for Aegis Crisis Command Center
Measures retrieval accuracy, latency, and end-to-end performance.
"""

import os
import sys
import time
import json
import datetime
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastembed import TextEmbedding
from config import get_qdrant_client

# Test queries with expected results
TEST_QUERIES = [
    {
        "query": "flood in Chennai",
        "expected_disaster": "flood",
        "expected_location": "chennai",
        "modality": "any"
    },
    {
        "query": "Bengaluru urban flooding",
        "expected_disaster": "flood",
        "expected_location": "bengaluru",
        "modality": "any"
    },
    {
        "query": "cyclone warning Kerala",
        "expected_disaster": "cyclone",
        "expected_location": "kerala",
        "modality": "any"
    },
    {
        "query": "bridge collapse emergency",
        "expected_disaster": "collapsed building",
        "expected_location": None,
        "modality": "visual"
    },
    {
        "query": "fire hazard alert",
        "expected_disaster": "fire",
        "expected_location": None,
        "modality": "any"
    }
]

# Collections with their embedding dimensions
TEXT_COLLECTIONS = ["audio_memory", "tactical_memory"]  # 384d BGE
VISUAL_COLLECTIONS = ["visual_memory"]  # 512d CLIP


class Benchmark:
    """Evaluation harness for Aegis retrieval system."""
    
    def __init__(self):
        self.client = get_qdrant_client()
        print("Loading embedding models...")
        self.text_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.clip_model = TextEmbedding(model_name="Qdrant/clip-ViT-B-32-text")
        self.results = []
        print("Models loaded.")
    
    def run_query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Run a single query and measure performance."""
        # Generate embeddings for both model types
        text_vector = list(self.text_model.embed([query]))[0].tolist()
        clip_vector = list(self.clip_model.embed([query]))[0].tolist()
        
        start_time = time.time()
        all_hits = []
        
        # Query text-based collections (384d)
        for collection in TEXT_COLLECTIONS:
            try:
                if self.client.collection_exists(collection):
                    res = self.client.query_points(
                        collection_name=collection,
                        query=text_vector,
                        limit=k,
                        with_payload=True
                    )
                    all_hits.extend(res.points)
            except Exception as e:
                print(f"Error querying {collection}: {e}")
        
        # Query visual collections (512d CLIP)
        for collection in VISUAL_COLLECTIONS:
            try:
                if self.client.collection_exists(collection):
                    res = self.client.query_points(
                        collection_name=collection,
                        query=clip_vector,
                        limit=k,
                        with_payload=True
                    )
                    all_hits.extend(res.points)
            except Exception as e:
                print(f"Error querying {collection}: {e}")
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Sort by score
        all_hits.sort(key=lambda x: x.score, reverse=True)
        top_k = all_hits[:k]
        
        return {
            "query": query,
            "latency_ms": round(latency_ms, 2),
            "total_hits": len(all_hits),
            "top_k": [
                {
                    "score": hit.score,
                    "disaster": hit.payload.get("detected_disaster"),
                    "location": hit.payload.get("location", {}).get("name", "") if isinstance(hit.payload.get("location"), dict) else "",
                    "source": hit.payload.get("source")
                }
                for hit in top_k
            ]
        }
    
    def evaluate_query(self, test_case: Dict, k: int = 5) -> Dict[str, Any]:
        """Evaluate a query against expected results."""
        result = self.run_query(test_case["query"], k)
        
        # Calculate metrics
        relevant_count = 0
        for hit in result["top_k"]:
            disaster_match = (
                test_case["expected_disaster"] is None or
                test_case["expected_disaster"].lower() in str(hit.get("disaster", "")).lower()
            )
            location_match = (
                test_case["expected_location"] is None or
                test_case["expected_location"].lower() in str(hit.get("location", "")).lower()
            )
            
            if disaster_match and location_match:
                relevant_count += 1
        
        precision_at_k = relevant_count / k if k > 0 else 0
        
        return {
            **result,
            "expected_disaster": test_case["expected_disaster"],
            "expected_location": test_case["expected_location"],
            "relevant_count": relevant_count,
            "precision_at_k": round(precision_at_k, 3)
        }
    
    def run_benchmark(self, k: int = 5) -> Dict[str, Any]:
        """Run full benchmark suite."""
        print("🧪 Running Aegis Benchmark Suite...")
        print(f"   Testing {len(TEST_QUERIES)} queries with k={k}")
        print("-" * 50)
        
        results = []
        total_latency = 0
        total_precision = 0
        
        for i, test in enumerate(TEST_QUERIES, 1):
            print(f"   [{i}/{len(TEST_QUERIES)}] {test['query'][:40]}...")
            result = self.evaluate_query(test, k)
            results.append(result)
            
            total_latency += result["latency_ms"]
            total_precision += result["precision_at_k"]
            
            print(f"         Latency: {result['latency_ms']:.1f}ms | P@{k}: {result['precision_at_k']:.2f}")
        
        # Aggregate metrics
        summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            "num_queries": len(TEST_QUERIES),
            "k": k,
            "avg_latency_ms": round(total_latency / len(TEST_QUERIES), 2),
            "avg_precision_at_k": round(total_precision / len(TEST_QUERIES), 3),
            "results": results
        }
        
        print("-" * 50)
        print(f"📊 Summary:")
        print(f"   Avg Latency: {summary['avg_latency_ms']:.1f} ms")
        print(f"   Avg P@{k}: {summary['avg_precision_at_k']:.3f}")
        
        # Save results
        with open("benchmark_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n✅ Results saved to benchmark_results.json")
        
        return summary
    
    def measure_throughput(self, num_queries: int = 20) -> Dict[str, Any]:
        """Measure query throughput."""
        print(f"\n⚡ Measuring throughput ({num_queries} queries)...")
        
        start = time.time()
        for i in range(num_queries):
            query = TEST_QUERIES[i % len(TEST_QUERIES)]["query"]
            self.run_query(query, k=3)
        
        elapsed = time.time() - start
        qps = num_queries / elapsed
        
        result = {
            "num_queries": num_queries,
            "elapsed_seconds": round(elapsed, 2),
            "queries_per_second": round(qps, 2)
        }
        
        print(f"   Queries/sec: {qps:.2f}")
        return result


if __name__ == "__main__":
    benchmark = Benchmark()
    
    # Run benchmark
    summary = benchmark.run_benchmark(k=5)
    
    # Measure throughput
    throughput = benchmark.measure_throughput(20)
    
    print("\n✅ Benchmark complete!")
