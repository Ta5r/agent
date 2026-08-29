import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import database
import rag

load_dotenv()

# Define tools for the agent
def check_order_status(order_id: str, email: str) -> str:
    """Checks the status and shipping details of a customer's order in the SQL database.
    
    Args:
        order_id: The ID of the order to check (e.g. ORD-1001).
        email: The customer's registered email address.
    """
    order = database.get_order(order_id, email)
    if order:
        tracking = f", Tracking Number: {order['tracking_number']}" if order['tracking_number'] else ""
        return (f"Order Found: {order['product_name']}. Status: {order['status']}. "
                f"Purchase Date: {order['purchase_date']}{tracking}.")
    else:
        return f"No order found with ID '{order_id}' for email '{email}'. Please check details or verify with the user."

def file_support_ticket(email: str, subject: str, description: str) -> str:
    """Creates a new customer support ticket in the SQL database when an issue cannot be resolved.
    
    Args:
        email: The customer's email address.
        subject: A brief subject describing the issue.
        description: A detailed description of the customer's complaint or request.
    """
    ticket = database.create_ticket(email, subject, description)
    return (f"Support ticket successfully created! Ticket ID: {ticket['id']}. "
            f"Status: {ticket['status']}. Subject: '{ticket['subject']}'. "
            f"We will contact the user at {ticket['customer_email']} soon.")

def search_policies_and_faqs(query: str) -> str:
    """Searches the company's knowledge base (PDF documents) for return policies, shipping times, warranty information, etc.
    
    Args:
        query: The semantic search query (e.g. 'refund policy window' or 'domestic shipping times').
    """
    matches = rag.search_knowledge_base(query, top_k=2)
    if not matches:
        return "No policy matching this query was found in the knowledge base."
    
    response_texts = []
    for m in matches:
        response_texts.append(f"From policy document '{m['source']}' (Relevance: {m['score']:.2f}):\n\"{m['text']}\"")
        
    return "\n\n---\n\n".join(response_texts)

# Map function names to executable Python functions
TOOL_MAP = {
    "check_order_status": check_order_status,
    "file_support_ticket": file_support_ticket,
    "search_policies_and_faqs": search_policies_and_faqs
}

SYSTEM_INSTRUCTION = """You are a helpful, professional, and friendly Customer Support Assistant for Quantum Tech Co.
Your goal is to resolve customer inquiries regarding orders, returns, refunds, and general support.

Guardrails and Instructions:
1. Verify Order Status: When a customer asks about their order status, you MUST ask for both their Order ID and their email address. Use the `check_order_status` tool to look it up. Never fabricate or guess order details.
2. Search Policies (RAG): If the user asks about returns, refund windows, shipping times, or other company policies, use `search_policies_and_faqs` to find the exact details. Cite policies accurately and align your answers with them.
3. Escalation / Filing Tickets: If you cannot resolve an issue, or if the order is cancelled/missing, or if the customer requests a refund that violates policies (e.g., return period expired), offer to file a support ticket for them. Use the `file_support_ticket` tool to register their complaint.
4. Conversation Scope: You must ONLY handle customer support inquiries related to Quantum Tech Co. If the user asks general questions unrelated to support (e.g., writing code, math, history, jokes), politely decline and state that you are only here to help with customer support queries.
5. Privacy: Never share details of other customers' orders or tickets.
"""

def run_agent(message: str, history: list) -> dict:
    """Runs the agent reasoning loop for a message, executes tools if requested, and returns the final response and logs.
    
    History format: list of {"role": "user"|"model", "text": "..."}
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "response": "⚠️ **API Key Missing**: Please set the `GEMINI_API_KEY` environment variable in a `.env` file at the project root to activate the AI Agent.",
            "logs": []
        }
        
    try:
        client = genai.Client()
    except Exception as e:
        return {
            "response": f"⚠️ **Client Error**: Could not initialize Gemini client: {str(e)}",
            "logs": []
        }
        
    # Convert chat history to API contents list
    contents = []
    for turn in history:
        role = turn.get("role")
        text = turn.get("text")
        if role and text:
            contents.append({
                "role": "user" if role == "user" else "model",
                "parts": [{"text": text}]
            })
            
    # Append the new user message
    contents.append({
        "role": "user",
        "parts": [{"text": message}]
    })
    
    # Configure the model call
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[check_order_status, file_support_ticket, search_policies_and_faqs],
        temperature=0.0, # low temp for deterministic tool calling
    )
    
    logs = []
    max_turns = 5
    
    for turn_idx in range(max_turns):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=contents,
                config=config
            )
        except Exception as e:
            logs.append({"error": f"Gemini API call failed: {str(e)}"})
            return {
                "response": "Sorry, I encountered an error communicating with the AI service. Please try again.",
                "logs": logs
            }
            
        # Check if the model requested function calls
        if response.function_calls:
            # We record model's function calls in conversation history
            model_parts = []
            for call in response.function_calls:
                model_parts.append({
                    "function_call": {
                        "name": call.name,
                        "args": call.args
                    }
                })
            contents.append({"role": "model", "parts": model_parts})
            
            # Now run the functions and compile the responses
            tool_parts = []
            for call in response.function_calls:
                tool_name = call.name
                tool_args = call.args
                
                logs.append({
                    "step": len(logs) + 1,
                    "action": f"Calling tool `{tool_name}`",
                    "arguments": tool_args
                })
                
                # Execute the tool
                func = TOOL_MAP.get(tool_name)
                if func:
                    try:
                        result = func(**tool_args)
                    except Exception as ex:
                        result = f"Error during execution: {str(ex)}"
                else:
                    result = f"Error: Tool '{tool_name}' not found."
                    
                logs.append({
                    "step": len(logs),
                    "action": f"Tool `{tool_name}` result",
                    "result": result
                })
                
                # Format response back to Gemini
                tool_parts.append({
                    "function_response": {
                        "name": tool_name,
                        "response": {"output": result}
                    }
                })
            contents.append({"role": "tool", "parts": tool_parts})
            
        else:
            # No function call, this is the final text response
            final_text = response.text
            contents.append({
                "role": "model",
                "parts": [{"text": final_text}]
            })
            return {
                "response": final_text,
                "logs": logs
            }
            
    # If we exceeded max turns without a final text response
    return {
        "response": "I'm still investigating this issue. Let me file a ticket for you to make sure we resolve it.",
        "logs": logs + [{"action": "Loop terminated", "result": "Exceeded maximum agent execution turns."}]
    }
