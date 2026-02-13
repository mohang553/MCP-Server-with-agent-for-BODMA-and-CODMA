#!/usr/bin/env python3
"""
Calculator Agent API - PLAIN HTTP TRANSPORT
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
# CALCULATOR AGENT
# ============================================================================

class CalculatorAgent:

    def __init__(self, model_name="gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model_name)
        self.tools = []
        self.initialized = False

    async def initialize(self):
        """Initialize agent by fetching available tools"""
        if self.initialized:
            return
        
        try:
            print("🔧 Initializing agent...")
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{MCP_SERVER_URL}/tools", timeout=10.0)
                response.raise_for_status()
                data = response.json()
                self.tools = data.get("tools", [])
                self.initialized = True
                print(f"✅ Agent initialized with {len(self.tools)} tools")
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            raise

    def _extract_numeric_answer(self, text: str) -> str:
        """Extract numeric answer from text"""
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return str(round(float(numbers[-1]), 2))
            except ValueError:
                pass
        return text

    async def _route(self, user_message: str) -> Dict[str, Any]:
        """Route user message to appropriate tool"""
        system_prompt = """You are a math router that decides which calculation tool to use.

Available tools:
- bodma: Calculates (a^b) / (a * b)
- codma: Calculates (a * b) + (a / b)

Respond ONLY with valid JSON:
{
  "action": "use_tool",
  "tool_name": "bodma or codma",
  "arguments": {"a": number, "b": number}
}"""

        prompt = system_prompt + f"\n\nUser request: {user_message}\n\nJSON Response:"

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()

            # Remove markdown code blocks
            if text.startswith("```"):
                text = re.sub(r'```json\n?|\n?```', '', text).strip()

            decision = json.loads(text)
            return decision

        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            return {"action": "error", "message": f"Invalid JSON response"}
        except Exception as e:
            print(f"❌ Routing error: {e}")
            return {"action": "error", "message": str(e)}

    async def run(self, user_message: str) -> Dict[str, Any]:
        """Main agent loop"""
        try:
            if not self.initialized:
                await self.initialize()

            decision = await self._route(user_message)

            if decision.get("action") == "error":
                return {
                    "response": f"Error: {decision.get('message', 'Unknown error')}",
                    "success": False
                }

            if decision.get("action") == "use_tool":
                tool_name = decision.get("tool_name")
                arguments = decision.get("arguments", {})

                # Validate arguments
                if not isinstance(arguments, dict) or "a" not in arguments or "b" not in arguments:
                    return {
                        "response": "Error: Invalid arguments. Need 'a' and 'b'",
                        "success": False
                    }

                try:
                    a = arguments["a"]
                    b = arguments["b"]
                    
                    print(f"🔧 Calling {tool_name}({a}, {b})")
                    
                    # Call the HTTP endpoint directly
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            f"{MCP_SERVER_URL}/tools/{tool_name}",
                            params={"a": a, "b": b},
                            timeout=10.0
                        )
                        response.raise_for_status()
                        data = response.json()
                        tool_result = str(data.get("result", "No result"))

                    print(f"✅ Tool result: {tool_result}")
                    
                    # Extract the numeric result
                    numeric = self._extract_numeric_answer(tool_result)
                    
                    return {
                        "response": numeric,
                        "success": True,
                        "tool": tool_name,
                        "args": arguments
                    }

                except Exception as e:
                    print(f"❌ Tool execution error: {e}")
                    return {
                        "response": f"Error executing tool: {str(e)}",
                        "success": False
                    }

            return {
                "response": "Error: Unknown action",
                "success": False
            }

        except Exception as e:
            print(f"❌ Agent error: {e}")
            return {
                "response": f"Error: {str(e)}",
                "success": False
            }


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Calculator Agent API",
    description="Routes math requests to calculation tools via HTTP"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = CalculatorAgent()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.get("/")
def root():
    return {
        "status": "running",
        "mcp_server": MCP_SERVER_URL,
        "initialized": agent.initialized
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/chat-agent", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process user message and return calculation result"""
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    result = await agent.run(request.message)
    return ChatResponse(response=result["response"])


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8005))

    print("=" * 60)
    print("🤖 CALCULATOR AGENT API (HTTP Transport)")
    print("=" * 60)
    print(f"📡 Running on port: {port}")
    print(f"🔗 MCP Server: {MCP_SERVER_URL}")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )
