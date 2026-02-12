#!/usr/bin/env python3
"""
Calculator Agent API - EXACT FIX
Issues Fixed:
1. Handle 500 error from server gracefully
2. Proper retry logic
3. Better error messages
"""

import asyncio
import json
import os
import uvicorn
import sys
import re
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# MCP (SSE transport)
from mcp import ClientSession
from mcp.client.sse import sse_client

# Gemini
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

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://calculator-mcp-74e1.onrender.com/mcp/sse")
MAX_RETRIES = 3
RETRY_DELAY = 2  # increased for server startup time


# ============================================================================
# MCP CLIENT
# ============================================================================

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.available_tools: List[Dict] = []

    @asynccontextmanager
    async def connect(self):
        """Connect with comprehensive retry logic and error handling"""
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                print(f"🔗 Attempting MCP connection (attempt {attempt + 1}/{MAX_RETRIES})...")
                
                async with sse_client(MCP_SERVER_URL) as (read, write):
                    async with ClientSession(read, write) as session:
                        self.session = session
                        await session.initialize()

                        tools_list = await session.list_tools()
                        self.available_tools = [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "input_schema": tool.inputSchema
                            }
                            for tool in tools_list.tools
                        ]

                        print(f"✅ MCP connected successfully. Tools: {[t['name'] for t in self.available_tools]}")
                        yield self
                        return
                        
            except Exception as e:
                last_error = str(e)
                print(f"⚠️  Connection attempt {attempt + 1} failed: {e}")
                
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print(f"❌ Failed to connect after {MAX_RETRIES} attempts")
                    raise RuntimeError(f"MCP connection failed: {last_error}")

    async def get_tools(self):
        return self.available_tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Call tool with error handling"""
        if not self.session:
            raise RuntimeError("MCP session not initialized")

        result = await self.session.call_tool(tool_name, arguments)

        if result.content:
            return "\n".join([
                item.text for item in result.content
                if hasattr(item, "text")
            ])

        return "No response"


# ============================================================================
# CALCULATOR AGENT
# ============================================================================

class CalculatorAgent:

    def __init__(self, mcp_client, model_name="gemini-1.5-flash"):
        self.mcp_client = mcp_client
        self.tools = []
        self.model = genai.GenerativeModel(model_name)
        self.initialized = False

    async def initialize(self):
        """Initialize agent with available tools - only once"""
        if self.initialized:
            return
            
        try:
            async with self.mcp_client.connect():
                self.tools = await self.mcp_client.get_tools()
                self.initialized = True
                print(f"✅ Agent initialized with {len(self.tools)} tools")
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            raise

    def _extract_numeric_answer(self, text: str) -> str:
        """Extract numeric answer from text"""
        # Look for explicit answer patterns
        patterns = [
            r'(?:answer|result|equals?|is)[:\s]+(-?\d+\.?\d*)',
            r'(-?\d+\.?\d*)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return str(round(float(match.group(1)), 2))
                except (ValueError, IndexError):
                    continue

        # Fallback: get the last number
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

        prompt = system_prompt + f"\n\nUser request: {user_message}\n\nJSON:"

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
            return {"action": "error", "message": "Could not parse routing decision"}
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
                    print(f"🔧 Calling {tool_name}({arguments['a']}, {arguments['b']})")
                    
                    async with self.mcp_client.connect():
                        tool_result = await self.mcp_client.call_tool(tool_name, arguments)

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
    description="Routes math requests to calculation tools"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MCPClient()
agent = CalculatorAgent(client)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.get("/")
def root():
    return {
        "status": "running",
        "mcp_server": MCP_SERVER_URL,
        "model": "gemini-1.5-flash",
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
    print("🤖 CALCULATOR AGENT API")
    print("=" * 60)
    print(f"📡 Running on port: {port}")
    print(f"🔗 MCP Server: {MCP_SERVER_URL}")
    print("=" * 60)

    uvicorn.run(
        "agent_EXACT_FIX:app",
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )
