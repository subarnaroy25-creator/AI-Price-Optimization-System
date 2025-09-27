import requests
import json

def test_mcp_server():
    base_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health")
        print("Health check:", response.json())
        
        # Test products endpoint
        response = requests.get(f"{base_url}/products")
        print("Products:", response.json())
        
        # Test query endpoint
        test_queries = [
            {"query": "summary", "params": {}},
            {"query": "price_ranges", "params": {}},
            {"query": "elasticity", "params": {}},
        ]
        
        for query in test_queries:
            response = requests.post(f"{base_url}/query", json=query)
            print(f"Query {query['query']}:", response.json())
            
    except Exception as e:
        print(f"Error testing MCP server: {e}")

if __name__ == "__main__":
    test_mcp_server()