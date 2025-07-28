#!/usr/bin/env python3
"""
Test script for debugging the embedding search functionality
"""

import os
import sys
import json
from pathlib import Path

# Add the current directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from register import debug_search, search_similar_files, find_all_code_files
from embedding_functions import embed_query, embed_file

def test_search(repo_path: str, test_queries: list[str]):
    """Test the search functionality with various queries"""
    
    print(f"Testing search functionality on repo: {repo_path}")
    print("=" * 50)
    
    # First, let's see what files were found
    code_files = find_all_code_files(repo_path)
    print(f"Found {len(code_files)} code files:")
    for file_path in code_files[:10]:  # Show first 10
        print(f"  {os.path.relpath(file_path, repo_path)}")
    if len(code_files) > 10:
        print(f"  ... and {len(code_files) - 10} more")
    print()
    
    # Test each query
    for query in test_queries:
        print(f"Testing query: '{query}'")
        print("-" * 30)
        
        # Use debug search for detailed info
        debug_info = debug_search(query, top_k=5)
        
        print(f"Total files in index: {debug_info['total_files']}")
        print(f"Total chunks in index: {debug_info['total_chunks']}")
        print(f"Index size: {debug_info['index_size']}")
        print(f"Preprocessed query: '{debug_info['preprocessed_query']}'")
        
        if 'error' in debug_info:
            print(f"Error: {debug_info['error']}")
        else:
            for search_result in debug_info['search_results']:
                print(f"\nResults for {search_result['query_type']} query:")
                if 'error' in search_result:
                    print(f"  Error: {search_result['error']}")
                else:
                    for i, result in enumerate(search_result['results'], 1):
                        print(f"  {i}. {result['file']} (score: {result['score']:.4f}, type: {result['type']})")
        
        print("\nStandard search results:")
        standard_results = search_similar_files(query, top_k=5)
        for i, (file_path, score) in enumerate(standard_results, 1):
            print(f"  {i}. {os.path.basename(file_path)} (score: {score:.4f})")
        
        print("\n" + "=" * 50)

def test_embedding_similarity(text1: str, text2: str):
    """Test similarity between two pieces of text"""
    import numpy as np
    
    print(f"Testing similarity between:")
    print(f"Text 1: '{text1[:100]}...'")
    print(f"Text 2: '{text2[:100]}...'")
    
    embed1 = embed_query(text1)
    embed2 = embed_query(text2)
    
    # Calculate cosine similarity
    embed1 = np.array(embed1)
    embed2 = np.array(embed2)
    
    # Normalize
    embed1_norm = embed1 / np.linalg.norm(embed1)
    embed2_norm = embed2 / np.linalg.norm(embed2)
    
    similarity = np.dot(embed1_norm, embed2_norm)
    print(f"Cosine similarity: {similarity:.4f}")
    print()

if __name__ == "__main__":
    # Example usage
    repo_path = "cloned_repos/NeMo-Agent-Toolkit-UI"  # Adjust this path
    
    if not os.path.exists(repo_path):
        print(f"Repository path not found: {repo_path}")
        print("Please update the repo_path variable in this script")
        sys.exit(1)
    
    test_queries = [
        "React component",
        "login form",
        "authentication",
        "API endpoint",
        "navigation menu",
        "button component",
        "state management",
        "user interface",
        "database connection",
        "error handling"
    ]
    
    # Test text similarity first
    print("Testing text embedding similarity:")
    test_embedding_similarity(
        "React component for user authentication",
        "login form component in React"
    )
    test_embedding_similarity(
        "React component",
        "import React from 'react'; function MyComponent() { return <div>Hello</div>; }"
    )
    
    # Test search functionality
    test_search(repo_path, test_queries) 