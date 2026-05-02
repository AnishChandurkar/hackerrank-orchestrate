import os
import time
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from prompts import SYSTEM_PROMPT, build_user_prompt

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=dotenv_path)

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is missing. Please set it in the .env file at the repository root.")

# Initialize Groq client
client = Groq(api_key=groq_api_key)

def generate_response(ticket_issue, ticket_subject, domain, retrieved_chunks, force_escalate=False, escalate_reason="", retrieval_score=0.0):
    """
    Generate a response using the Groq API and Llama 3.3 70B model.
    """
    if force_escalate:
        return {
            "status": "escalated",
            "product_area": "General Support",
            "response": f"Your request has been escalated to our support team. {escalate_reason}",
            "justification": f"Escalated by safety layer: {escalate_reason}",
            "request_type": "product_issue"
        }

    model_name = "llama-3.3-70b-versatile"
    
    company = domain if domain and domain != 'unknown' else 'our company'
    system_prompt = SYSTEM_PROMPT.format(domain=company)
    
    # Cap total context to 2500 characters to stay within Groq TPM limits.
    # Chunks arrive ranked by BM25 relevance (best first), so we trim from
    # the tail — dropping the *least* relevant chunk until the budget fits.
    MAX_CONTEXT_CHARS = 2500
    capped_chunks = list(retrieved_chunks) if retrieved_chunks else []
    while capped_chunks and sum(len(c.get("text", "")) for c in capped_chunks) > MAX_CONTEXT_CHARS:
        capped_chunks.pop()  # drop least-relevant (last) chunk

    user_prompt = build_user_prompt(ticket_issue, ticket_subject, domain, capped_chunks, force_escalate, escalate_reason, retrieval_score=retrieval_score)
    
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    
    required_fields = {"status", "product_area", "response", "justification", "request_type"}
    
    # We allow a maximum of 2 attempts. Two attempts is the right limit because:
    # 1. It provides one chance to fix missing fields or format issues.
    # 2. More than two attempts wastes API tokens, inflates latency, and risks infinite loops on a fundamentally broken prompt.
    max_attempts = 2
    
    for attempt in range(max_attempts):
        try:
            # Log model and timestamp to stdout
            print(f"[{datetime.now().isoformat()}] Calling model: {model_name} (Attempt {attempt + 1})")
            
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=0,
                seed=42,
                response_format={"type": "json_object"}
            )
            
            response_content = chat_completion.choices[0].message.content
            result = json.loads(response_content)
            
            # Check if all required fields are present
            if required_fields.issubset(result.keys()):
                return result
            else:
                print(f"[{datetime.now().isoformat()}] Missing required fields in attempt {attempt + 1}: {result.keys()}")
                if attempt < max_attempts - 1:
                    messages.append({
                        "role": "assistant",
                        "content": response_content
                    })
                    messages.append({
                        "role": "user",
                        "content": "Your previous response was missing required fields. Return valid JSON with exactly these fields: status, product_area, response, justification, request_type"
                    })
        except Exception as e:
            print(f"[{datetime.now().isoformat()}] API Error on attempt {attempt + 1}: {str(e)}")
            # On exception, we could retry if it's not the last attempt
            # But the requirement specifically says "On API error, return a safe hardcoded escalation dict rather than crashing."
            # and "If the second attempt also fails, return the hardcoded escalation fallback."
            # We'll let it retry on API error as well or just loop.
            pass
            
        finally:
            # Add a 2 second sleep after each call as requested
            time.sleep(2)
            
    return {
        "status": "escalated",
        "product_area": "unknown",
        "response": "",
        "justification": "API Error: Unable to generate valid response after 2 attempts",
        "request_type": "unknown"
    }

if __name__ == "__main__":
    # Single ticket test - burns one API call
    test_chunks = [
        {
            "domain": "visa",
            "subfolder": "support",
            "title": "Dispute Resolution",
            "source_filename": "dispute-resolution.md",
            "text": "If you believe a charge is incorrect, contact your card issuer to initiate a dispute."
        }
    ]
    
    result = generate_response(
        ticket_issue="I was charged twice for the same transaction",
        ticket_subject="Double charge",
        domain="visa",
        retrieved_chunks=test_chunks,
        force_escalate=False,
        escalate_reason=""
    )
    
    print(json.dumps(result, indent=2))