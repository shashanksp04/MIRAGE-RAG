#!/usr/bin/env python3
"""
Standalone test script for RAG Agent.

This script allows you to test the RAG agent independently without running
the full pipeline. It accepts command-line arguments and returns the RAG
agent's output.

Usage:
    python test_standalone.py --query "Your question here"
    python test_standalone.py --query "Your question" --reset-collection
    python test_standalone.py --query "Your question" --test-model "Qwen2.5-VL-3B-Instruct" --api-base "http://127.0.0.1:11434/v1"
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_agent.main import MainAgent


def extract_rag_response(rag_response, verbose=False):
    """
    Extract the RAG agent's text response from the Event objects.
    This replicates the logic from generate.py's rag_worker_process.
    
    Args:
        rag_response: Response from runner.run_debug()
        verbose: Whether to print debug information
        
    Returns:
        tuple: (rag_answer, web_search_performed, tool_calls_found)
    """
    rag_answer = None
    web_search_performed = False
    tool_calls_found = []
    
    if rag_response is None:
        if verbose:
            print("[Response Extraction] WARNING - rag_response is None!")
        return None, False, []
    
    elif isinstance(rag_response, list):
        if verbose:
            print(f"[Response Extraction] Response is a list with {len(rag_response)} events")
        
        agent_texts = []
        for i, event in enumerate(rag_response):
            event_type = type(event).__name__
            author = getattr(event, 'author', 'unknown')
            
            if verbose:
                print(f"[Response Extraction] Event {i}: type={event_type}, author={author}")
            
            # Check for tool calls in the event
            function_calls = event.get_function_calls()
            if function_calls:
                if verbose:
                    print(f"[Response Extraction]   → Found {len(function_calls)} tool call(s) in event {i}")
                for tc in function_calls:
                    tool_name = getattr(tc, 'name', 'unknown')
                    tool_calls_found.append(tool_name)
                    if verbose:
                        print(f"[Response Extraction]     - Tool: {tool_name}")
            
            # Check if this is from the Rag_Agent
            if author == 'Rag_Agent':
                # Extract text from content.parts
                if hasattr(event, 'content') and hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            agent_texts.append(part.text)
        
        if tool_calls_found:
            if verbose:
                print(f"[Response Extraction] ✓ Found tool calls: {', '.join(tool_calls_found)}")
        
        # Check if web_search was performed
        web_search_performed = 'web_search' in tool_calls_found or '_tracked_web_search' in tool_calls_found
        if web_search_performed:
            if verbose:
                print(f"[Response Extraction] ✓ Web search was performed")
        
        if agent_texts:
            # Get the last agent response (most recent)
            rag_answer = agent_texts[-1].strip()
            if verbose:
                print(f"[Response Extraction] ✓ Extracted agent text from Event objects, length: {len(rag_answer)} chars")
        else:
            if verbose:
                print(f"[Response Extraction] WARNING - No agent text found in events!")
    
    else:
        # Fallback: try to convert to string and extract using regex
        rag_response_str = str(rag_response)
        if verbose:
            print(f"[Response Extraction] Response is not a list, trying string extraction...")
        
        # Check if web_search appears in string representation (fallback detection)
        if 'web_search' in rag_response_str.lower() or '_tracked_web_search' in rag_response_str:
            web_search_performed = True
            if verbose:
                print(f"[Response Extraction] ✓ Web search detected in string representation")
        
        import re
        # Try to extract text from the string representation (look for text="""...""")
        text_match = re.search(r'text="""(.*?)"""', rag_response_str, re.DOTALL)
        if text_match:
            rag_answer = text_match.group(1).strip()
            if verbose:
                print(f"[Response Extraction] ✓ Extracted text from string representation, length: {len(rag_answer)} chars")
    
    return rag_answer, web_search_performed, tool_calls_found


async def run_rag_query(runner, query, session_id=None, verbose=False):
    """
    Run a query through the RAG agent and return the response.
    
    Args:
        runner: The InMemoryRunner instance from MainAgent.main()
        query: The user query/question
        session_id: Optional session ID for the query
        verbose: Whether to print debug information
        
    Returns:
        Response from runner.run_debug()
    """
    try:
        if session_id:
            rag_response = await runner.run_debug(query, session_id=session_id)
            if verbose:
                print(f"[RAG Query] Used session_id={session_id}")
        else:
            rag_response = await runner.run_debug(query)
            if verbose:
                print(f"[RAG Query] Using default session")
        
        return rag_response
    except TypeError:
        # If session_id parameter doesn't exist, use default behavior
        rag_response = await runner.run_debug(query)
        if verbose:
            print(f"[RAG Query] Using default session (session_id not supported)")
        return rag_response


def main():
    parser = argparse.ArgumentParser(
        description="Standalone test script for RAG Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic query
  python test_standalone.py --query "What is Agent Development Kit from Google?"
  
  # Query with custom model and API base
  python test_standalone.py --query "Your question" --test-model "Qwen2.5-VL-3B-Instruct" --api-base "http://127.0.0.1:11434/v1"
  
  # Query with collection reset (starts fresh)
  python test_standalone.py --query "Your question" --reset-collection
  
  # Verbose output with debug information
  python test_standalone.py --query "Your question" --verbose
  
  # Custom database path
  python test_standalone.py --query "Your question" --db-path "/path/to/chroma_db"
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='The query/question to ask the RAG agent'
    )
    
    # Optional arguments
    parser.add_argument(
        '--test-model',
        type=str,
        default="Qwen2.5-VL-3B-Instruct",
        help='Model name for the RAG agent (default: Qwen2.5-VL-3B-Instruct)'
    )
    
    parser.add_argument(
        '--embed-model',
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help='Embedding model name (default: BAAI/bge-base-en-v1.5)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default="None",
        help='Device to use for embeddings (default: None, uses CPU)'
    )
    
    parser.add_argument(
        '--api-base',
        type=str,
        default="http://127.0.0.1:11434/v1",
        help='API base URL for the LLM (default: http://127.0.0.1:11434/v1)'
    )
    
    parser.add_argument(
        '--qdrant-url',
        type=str,
        default=None,
        help='Qdrant server URL (default: QDRANT_URL env or http://127.0.0.1:6333)'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='Deprecated alias for --qdrant-url (Chroma path no longer used)'
    )
    
    parser.add_argument(
        '--reset-collection',
        action='store_true',
        help='Reset the collection before running the query (drops all existing data)'
    )
    
    parser.add_argument(
        '--session-id',
        type=str,
        default=None,
        help='Session ID for the query (default: auto-generated)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose debug information'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Save the RAG response to a file (optional)'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 80)
    print("RAG Agent Standalone Test")
    print("=" * 80)
    print(f"Query: {args.query}")
    print(f"Test Model: {args.test_model}")
    print(f"Embed Model: {args.embed_model}")
    print(f"Device: {args.device}")
    print(f"API Base: {args.api_base}")
    qdrant_url = args.qdrant_url or args.db_path or os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    print(f"Qdrant URL: {qdrant_url}")
    print(f"Reset Collection: {args.reset_collection}")
    print(f"Verbose: {args.verbose}")
    print("=" * 80)
    print()
    
    try:
        # Initialize the RAG agent
        if args.verbose:
            print("[Init] Initializing MainAgent...")
        
        main_agent = MainAgent(
            test_model=args.test_model,
            embed_model_name=args.embed_model,
            device=args.device,
            api_base=args.api_base,
            qdrant_url=qdrant_url,
        )
        
        # Reset collection if requested
        if args.reset_collection:
            if args.verbose:
                print("[Init] Resetting collection...")
            main_agent.reset_collection()
            print("[Init] ✓ Collection reset complete")
        
        # Create the runner
        if args.verbose:
            print("[Init] Creating runner...")
        runner = main_agent.main()
        
        # Generate session ID if not provided
        session_id = args.session_id
        if session_id is None:
            import uuid
            session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        
        # Run the query
        print(f"\n[Query] Processing query...")
        print(f"[Query] Session ID: {session_id}")
        print()
        
        # Run async query
        rag_response = asyncio.run(run_rag_query(runner, args.query, session_id=session_id, verbose=args.verbose))
        
        # Extract the response
        print("\n[Extraction] Extracting response...")
        rag_answer, web_search_performed, tool_calls = extract_rag_response(rag_response, verbose=args.verbose)
        
        # Print results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        if tool_calls:
            print(f"\nTools Called: {', '.join(tool_calls)}")
        
        if web_search_performed:
            print("\n✓ Web search was performed during retrieval")
        
        print("\n" + "-" * 80)
        print("RAG Agent Response:")
        print("-" * 80)
        
        if rag_answer:
            print(rag_answer)
            print()
            print(f"Response length: {len(rag_answer)} characters")
            
            # Validate response
            tool_names = ['_tracked_retrieve_content', '_tracked_evaluate_confidence', 
                         '_tracked_web_search', '_tracked_add_web_content', '_tracked_add_pdf_content', 
                         '_tracked_extract_keywords']
            
            rag_answer_lower = rag_answer.lower()
            tool_name_count = sum(1 for tool_name in tool_names if tool_name.lower() in rag_answer_lower)
            
            if tool_name_count >= 1 and len(rag_answer.split('\n')) <= tool_name_count + 2:
                print("\n⚠ WARNING: Response appears to be just tool names, not actual results")
            else:
                print("\n✓ Response appears valid")
            
            # Save to file if requested
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write(rag_answer)
                print(f"\n✓ Response saved to: {args.output_file}")
        else:
            print("No response extracted from RAG agent.")
            print("\n⚠ WARNING: Failed to extract response. Check verbose output for details.")
            if args.verbose:
                print(f"\nRaw response type: {type(rag_response)}")
                print(f"Raw response (first 500 chars): {str(rag_response)[:500]}")
        
        print("=" * 80)
        
        # Return exit code
        return 0 if rag_answer else 1
        
    except KeyboardInterrupt:
        print("\n\n[ERROR] Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n[ERROR] Exception occurred: {str(e)}")
        if args.verbose:
            import traceback
            print("\nTraceback:")
            print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
