#!/usr/bin/env python3
"""
Calculator Agent API
HTTP MCP Version (SSE)
Render Production Ready
"""

import asyncio
import json
import os
import uvicorn
import sys
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

MCP_SERVER_URL = "https://calculator-mcp-74e1.onrender.com/mcp"


# ============================================================================
# MCP CLIENT
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
# CALCULATOR AGENT
# ============================================================================

class CalculatorAgent:

    def __init__(self, mcp_client, model_name="gemini-2.5-flash"):
        self.mcp_client = mcp_client
        self.tools = []
        self.model = genai.GenerativeModel(model_name)

    async def initialize(self):
        async with self.mcp_client.connect():
            self.tools = await self.mcp_client.get_tools()

    def _extract_numeric_answer(self, text: str):
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return str(round(float(numbers[0]), 2))
        return text

    async def _route(self, user_message: str):

        system_prompt = """
You are a math router.

Tools:
- bodma_calculate
- codma_calculate

Return JSON:
{
 "action": "use_tool",
 "tool_name": "...",
 "arguments": {"a": number, "b": number}
}
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

        decision = await self._route(user_message)

        if decision.get("action") == "use_tool":

            async with self.mcp_client.connect():

                tool_result = await self.mcp_client.call_tool(
                    decision["tool_name"],
                    decision["arguments"]
                )

            final_prompt = f"""
User asked: {user_message}

Tool result:
{tool_result}

Return only final numeric answer.
"""

            final_response = self.model.generate_content(final_prompt)
            numeric = self._extract_numeric_answer(final_response.text)

            return {"response": numeric}

        return {"response": "Unable to process"}


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Calculator Agent API (HTTP MCP Version)")

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
    response: float


@app.get("/")
def root():
    return {
        "status": "running",
        "connected_to": MCP_SERVER_URL
    }


@app.post("/chat-agent", response_model=ChatResponse)
async def chat(request: ChatRequest):

    result = await agent.run(request.message)

    try:
        return ChatResponse(response=float(result["response"]))
    except:
        raise HTTPException(status_code=500, detail="Invalid numeric response")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8005))

    print("=" * 60)
    print("🤖 CALCULATOR AGENT API")
    print("=" * 60)
    print(f"📡 Running on port: {port}")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )
