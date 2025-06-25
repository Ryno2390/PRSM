#!/usr/bin/env python3
"""
PRSM Python SDK - Streaming Examples

This example demonstrates streaming capabilities including real-time responses,
batch streaming, and advanced streaming patterns.
"""

import asyncio
import os
import time
from prsm_sdk import PRSMClient, PRSMError


async def basic_streaming_example():
    """Basic streaming response example"""
    print("=== Basic Streaming ===")
    
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    try:
        print("Prompt: 'Explain machine learning step by step'")
        print("Response: ", end="", flush=True)
        
        start_time = time.time()
        total_tokens = 0
        
        async for chunk in client.models.stream(
            model="gpt-4",
            prompt="Explain machine learning step by step in simple terms",
            max_tokens=500,
            temperature=0.7
        ):
            print(chunk.content, end="", flush=True)
            total_tokens += chunk.tokens if hasattr(chunk, 'tokens') else 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n\nStreaming completed:")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Tokens: {total_tokens}")
        print(f"Rate: {total_tokens / duration:.1f} tokens/second")
        
    except PRSMError as e:
        print(f"PRSM API Error: {e.message}")


async def multiple_stream_example():
    """Handle multiple concurrent streams"""
    print("\n=== Concurrent Streaming ===")
    
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    prompts = [
        "Write a haiku about technology",
        "Explain quantum physics briefly", 
        "List 5 benefits of renewable energy"
    ]
    
    async def stream_prompt(prompt, stream_id):
        """Stream a single prompt with ID"""
        print(f"\n[Stream {stream_id}] Starting: '{prompt[:30]}...'")
        response_parts = []
        
        try:
            async for chunk in client.models.stream(
                model="gpt-3.5-turbo",  # Faster for demo
                prompt=prompt,
                max_tokens=150
            ):
                response_parts.append(chunk.content)
            
            full_response = "".join(response_parts)
            print(f"\n[Stream {stream_id}] Complete:")
            print(f"{full_response}")
            
            return {"id": stream_id, "prompt": prompt, "response": full_response}
            
        except PRSMError as e:
            print(f"[Stream {stream_id}] Error: {e.message}")
            return {"id": stream_id, "error": str(e)}
    
    # Run multiple streams concurrently
    tasks = [
        stream_prompt(prompt, i + 1) 
        for i, prompt in enumerate(prompts)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    print(f"\nCompleted {len(results)} concurrent streams")


async def interactive_streaming_example():
    """Interactive streaming with user feedback"""
    print("\n=== Interactive Streaming ===")
    
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    conversation_history = []
    
    # Simulate a conversation
    user_inputs = [
        "Hello, can you help me with Python programming?",
        "How do I create a web API?",
        "What about authentication?"
    ]
    
    for user_input in user_inputs:
        print(f"\nUser: {user_input}")
        print("Assistant: ", end="", flush=True)
        
        # Build conversation context
        context = "\n".join([
            f"User: {msg['user']}\nAssistant: {msg['assistant']}"
            for msg in conversation_history
        ])
        
        if context:
            full_prompt = f"{context}\nUser: {user_input}\nAssistant:"
        else:
            full_prompt = f"User: {user_input}\nAssistant:"
        
        response_parts = []
        
        try:
            async for chunk in client.models.stream(
                model="gpt-4",
                prompt=full_prompt,
                max_tokens=200,
                temperature=0.7
            ):
                print(chunk.content, end="", flush=True)
                response_parts.append(chunk.content)
            
            full_response = "".join(response_parts)
            conversation_history.append({
                "user": user_input,
                "assistant": full_response
            })
            
            print()  # New line after streaming
            
        except PRSMError as e:
            print(f"\nError: {e.message}")
            break


async def streaming_with_cost_tracking():
    """Stream while tracking costs in real-time"""
    print("\n=== Streaming with Cost Tracking ===")
    
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    try:
        # Get initial budget
        initial_budget = await client.cost_optimization.get_budget()
        print(f"Starting budget: ${initial_budget.remaining:.4f}")
        
        prompt = "Write a detailed technical explanation of how neural networks work"
        print(f"\nPrompt: {prompt}")
        print("Response: ", end="", flush=True)
        
        total_cost = 0.0
        chunk_count = 0
        
        async for chunk in client.models.stream(
            model="gpt-4",
            prompt=prompt,
            max_tokens=800
        ):
            print(chunk.content, end="", flush=True)
            
            # Track cost if available
            if hasattr(chunk, 'cost'):
                total_cost += chunk.cost
                chunk_count += 1
                
                # Show cost update every 50 chunks
                if chunk_count % 50 == 0:
                    print(f"\n[Cost update: ${total_cost:.6f}]", end="", flush=True)
        
        # Final cost summary
        final_budget = await client.cost_optimization.get_budget()
        actual_cost = initial_budget.remaining - final_budget.remaining
        
        print(f"\n\nCost Summary:")
        print(f"Estimated cost: ${total_cost:.6f}")
        print(f"Actual cost: ${actual_cost:.6f}")
        print(f"Remaining budget: ${final_budget.remaining:.4f}")
        
    except PRSMError as e:
        print(f"PRSM API Error: {e.message}")


async def streaming_with_filters():
    """Demonstrate streaming with content filtering"""
    print("\n=== Streaming with Content Filtering ===")
    
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    def content_filter(text):
        """Simple content filter - remove certain words"""
        filtered_words = ["TODO", "FIXME", "XXX"]
        for word in filtered_words:
            text = text.replace(word, "[FILTERED]")
        return text
    
    try:
        print("Prompt: 'Write code comments with TODO items'")
        print("Filtered Response: ", end="", flush=True)
        
        buffer = ""
        
        async for chunk in client.models.stream(
            model="gpt-4",
            prompt="Write Python code with TODO comments for a web scraper",
            max_tokens=400
        ):
            buffer += chunk.content
            
            # Process complete sentences
            while ". " in buffer:
                sentence, buffer = buffer.split(". ", 1)
                sentence += ". "
                filtered_sentence = content_filter(sentence)
                print(filtered_sentence, end="", flush=True)
        
        # Process remaining buffer
        if buffer:
            filtered_buffer = content_filter(buffer)
            print(filtered_buffer, end="", flush=True)
        
        print("\n[Filtering applied during streaming]")
        
    except PRSMError as e:
        print(f"PRSM API Error: {e.message}")


async def main():
    """Run all streaming examples"""
    if not os.getenv("PRSM_API_KEY"):
        print("Please set PRSM_API_KEY environment variable")
        return
    
    print("PRSM Python SDK - Streaming Examples")
    print("=" * 50)
    
    await basic_streaming_example()
    await multiple_stream_example()
    await interactive_streaming_example()
    await streaming_with_cost_tracking()
    await streaming_with_filters()
    
    print("\n" + "=" * 50)
    print("Streaming examples completed!")


if __name__ == "__main__":
    asyncio.run(main())