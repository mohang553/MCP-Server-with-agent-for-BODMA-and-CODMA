#!/usr/bin/env python3
"""
Calculator Agent API - EXACT ORIGINAL LOGIC
HTTP Transport Version
Handles BODMA, CODMA, and complex multi-step queries
"""

import asyncio
import json
import os
import uvicorn
import sys
import re
from typing import Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

import google.generativeai as genai

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not found")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://calculator-mcp-74e1.onrender.com")
print(f"📡 MCP Server URL: {MCP_SERVER_URL}")


# ============================================================================
# CALCULATOR AGENT - ORIGINAL LOGIC PRESERVED
# ============================================================================

class CalculatorAgent:

    def __init__(self, model_name="gemini-2.5-flash"):
        self.model_name = model_name
        try:
            self.model = genai.GenerativeModel(model_name)
            print(f"✅ Using model: {model_name}")
        except Exception as e:
            print(f"⚠️  Model {model_name} failed: {e}")
            print(f"🔄 Trying fallback model: gemini-pro")
            self.model = genai.GenerativeModel("gemini-pro")
            self.model_name = "gemini-pro"

    def _extract_numeric_answer(self, response_text: str, decimal_places: int = 2):
        """Extract numeric answer - ORIGINAL LOGIC"""
        import re
        numbers = re.findall(r'-?\d+\.?\d*', response_text)
        if numbers:
            try:
                return str(round(float(numbers[0]), decimal_places))
            except:
                pass
        return response_text

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call tool via HTTP - ORIGINAL FLOW"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{MCP_SERVER_URL}/tools/{tool_name}",
                    params=arguments,
                    timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
                return str(data.get("result", "No result"))
        except Exception as e:
            print(f"❌ Tool call error: {e}")
            raise

    async def _route_with_gemini(self, user_message: str):
        """Route with Gemini - EXACT ORIGINAL LOGIC"""

        system_prompt = f"""
You are a math agent.

Available tools:
- bodma_calculate
- codma_calculate

If user asks for:
- Only BODMA → use bodma_calculate
- Only CODMA → use codma_calculate
- Both → use "both"

Return JSON:
{{
 "action": "use_tool",
 "tool_name": "...",
 "arguments": {{"a": number, "b": number}}
}}
"""

        prompt = system_prompt + f"\nUser: {user_message}\nJSON:"

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()

            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()

            return json.loads(text)
        except Exception as e:
            print(f"❌ Routing error: {e}")
            return {"action": "error", "message": str(e)}

    async def run(self, user_message: str):
        """Main run - EXACT ORIGINAL FLOW"""

        try:
            decision = await self._route_with_gemini(user_message)

            if decision.get("action") == "error":
                return {
                    "type": "direct_response",
                    "response": decision.get("message", "Error processing request")
                }

            if decision.get("action") == "use_tool":

                tool_name = decision["tool_name"]
                arguments = decision["arguments"]

                print(f"🔧 Decision: tool_name={tool_name}, args={arguments}")

                if tool_name == "both":
                    # EXACT ORIGINAL LOGIC: Call both tools
                    print(f"📊 Calling BOTH tools with a={arguments['a']}, b={arguments['b']}")
                    
                    bodma_result = await self._call_tool("bodma_calculate", arguments)
                    print(f"  ✅ BODMA result: {bodma_result}")
                    
                    codma_result = await self._call_tool("codma_calculate", arguments)
                    print(f"  ✅ CODMA result: {codma_result}")

                    # EXACT ORIGINAL: Send both results to Gemini
                    follow_up_prompt = f"""
User asked: {user_message}

BODMA result: {bodma_result}
CODMA result: {codma_result}

Return only final numeric answer.
"""

                else:
                    # EXACT ORIGINAL: Single tool call
                    print(f"📊 Calling {tool_name} with a={arguments['a']}, b={arguments['b']}")
                    
                    tool_result = await self._call_tool(tool_name, arguments)
                    print(f"  ✅ Result: {tool_result}")

                    follow_up_prompt = f"""
User asked: {user_message}

Tool result:
{tool_result}

Return only final numeric answer.
"""

                # EXACT ORIGINAL: Get final response from Gemini
                final_response = self.model.generate_content(follow_up_prompt)
                numeric = self._extract_numeric_answer(final_response.text)

                print(f"📝 Final numeric answer: {numeric}")

                return {
                    "type": "direct_response",
                    "response": numeric
                }

            return {
                "type": "direct_response",
                "response": "Could not process request"
            }

        except Exception as e:
            print(f"❌ Agent error: {e}")
            return {
                "type": "direct_response",
                "response": f"Error: {str(e)}"
            }


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Calculator Agent API (HTTP Transport)",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = CalculatorAgent()


# ============================================================================
# REQUEST MODELS
# ============================================================================

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "status": "running",
        "transport": "HTTP",
        "mcp_server": MCP_SERVER_URL,
        "model": agent.model_name
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat-agent", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """Process user message - EXACT ORIGINAL LOGIC"""

    result = await agent.run(request.message)

    return ChatResponse(response=result["response"])


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8005))

    print("=" * 70)
    print("🤖 CALCULATOR AGENT API - HTTP TRANSPORT")
    print("=" * 70)
    print(f"📡 Agent running on port: {port}")
    print(f"🔗 Connected to MCP: {MCP_SERVER_URL}")
    print("✨ Features: BODMA, CODMA, Complex queries")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=port)
