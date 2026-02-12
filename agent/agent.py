#!/usr/bin/env python3
"""
MCP Agent FastAPI Server - Calculator Edition with Gemini AI
HTTP Transport Version (SSE) - No STDIO
"""

import asyncio
import json
import sys
import os
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# MCP imports (HTTP transport)
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
    print("❌ ERROR: GEMINI_API_KEY not found")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

MCP_SERVER_URL = "http://localhost:8001/mcp/sse"

mcp_manager = None
agents = {}

# ============================================================================
# MCP CLIENT (HTTP VERSION)
# ============================================================================

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.available_tools: List[Dict] = []

    @asynccontextmanager
    async def connect(self):
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

                yield self

    async def get_tools(self):
        return self.available_tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        result = await self.session.call_tool(tool_name, arguments)

        if result.content:
            return "\n".join([
                item.text for item in result.content
                if hasattr(item, "text")
            ])
        return "No response"


# ============================================================================
# GEMINI CALCULATOR AGENT (UNCHANGED LOGIC)
# ============================================================================

class CalculatorAgent:

    def __init__(self, mcp_client, model_name="gemini-2.5-flash"):
        self.mcp_client = mcp_client
        self.tools = []
        self.model = genai.GenerativeModel(model_name)

    async def initialize(self):
        async with self.mcp_client.connect():
            self.tools = await self.mcp_client.get_tools()

    def _extract_numeric_answer(self, response_text: str, decimal_places: int = 2):
        import re
        numbers = re.findall(r'-?\d+\.?\d*', response_text)
        if numbers:
            try:
                return str(round(float(numbers[0]), decimal_places))
            except:
                pass
        return response_text

    async def _route_with_gemini(self, user_message: str):

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

        response = self.model.generate_content(prompt)
        text = response.text.strip()

        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        return json.loads(text)

    async def run(self, user_message: str):

        if not self.tools:
            await self.initialize()

        decision = await self._route_with_gemini(user_message)

        if decision.get("action") == "use_tool":

            tool_name = decision["tool_name"]
            arguments = decision["arguments"]

            async with self.mcp_client.connect():

                if tool_name == "both":
                    bodma = await self.mcp_client.call_tool("bodma_calculate", arguments)
                    codma = await self.mcp_client.call_tool("codma_calculate", arguments)

                    follow_up_prompt = f"""
User asked: {user_message}

BODMA result: {bodma}
CODMA result: {codma}

Return only final numeric answer.
"""

                else:
                    tool_result = await self.mcp_client.call_tool(tool_name, arguments)

                    follow_up_prompt = f"""
User asked: {user_message}

Tool result:
{tool_result}

Return only final numeric answer.
"""

            final_response = self.model.generate_content(follow_up_prompt)
            numeric = self._extract_numeric_answer(final_response.text)

            return {
                "type": "direct_response",
                "response": numeric
            }

        return {
            "type": "direct_response",
            "response": "Could not process request"
        }


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Calculator Agent API (HTTP MCP Version)",
    version="3.0.0"
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


# ============================================================================
# REQUEST MODELS
# ============================================================================

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: float


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "status": "running",
        "transport": "HTTP (SSE)",
        "mcp_server": MCP_SERVER_URL
    }


@app.post("/chat-agent", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):

    result = await agent.run(request.message)

    try:
        return ChatResponse(response=float(result["response"]))
    except:
        raise HTTPException(status_code=500, detail="Invalid numeric response")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("🤖 CALCULATOR AGENT API - HTTP MCP VERSION")
    print("=" * 70)
    print("📡 Agent running at: http://localhost:8005")
    print("🔌 Connected to MCP: http://localhost:8001/mcp")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=8005)
