#!/usr/bin/env python3
"""
MCP Agent FastAPI Server - Calculator Edition with Gemini AI
Intelligent tool routing for BODMA and CODMA calculations using Google Gemini
REQUIRES Google Gemini API key - No fallback to rule-based routing
"""
import asyncio
import json
import sys
import os
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Google Gemini imports (REQUIRED)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("❌ ERROR: google-generativeai package not installed!")
    print("   Install with: pip install google-generativeai")
    sys.exit(1)

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY not found in environment!")
    print("   Please set GEMINI_API_KEY in your .env file")
    print("   Example: GEMINI_API_KEY=your-api-key-here")
    sys.exit(1)

mcp_manager = None
agents = {}

print(f"✅ Gemini API Key configured")

# ============================================================================
# MCP CLIENT CODE
# ============================================================================

class MCPClient:
    """Client to interact with MCP servers"""
    
    def __init__(self, server_config: Dict[str, Any]):
        self.server_config = server_config
        self.session: Optional[ClientSession] = None
        self.available_tools: List[Dict] = []
        
    @asynccontextmanager
    async def connect(self):
        """Connect to the MCP server"""
        server_params = StdioServerParameters(
            command=self.server_config["command"],
            args=self.server_config.get("args", []),
            env=self.server_config.get("env")
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await session.initialize()
                
                # Fetch available tools
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
    
    async def get_tools(self) -> List[Dict]:
        """Get list of available tools from MCP server"""
        return self.available_tools
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool on the MCP server"""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
        
        result = await self.session.call_tool(tool_name, arguments)
        
        # Extract text content from response
        if result.content:
            return "\n".join([
                item.text for item in result.content 
                if hasattr(item, 'text')
            ])
        return "No response from tool"


class MCPClientManager:
    """Manages multiple MCP server connections"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.clients: Dict[str, MCPClient] = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load MCP server configuration from JSON file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                print(f"⚠️  Config file not found: {self.config_path}")
                print(f"   Looking in: {config_file.absolute()}")
                return {}
            
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get("mcpServers", {})
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return {}
    
    async def initialize_clients(self):
        """Initialize all MCP clients from config"""
        servers_config = self.load_config()
        
        if not servers_config:
            print("⚠️  No MCP servers found in config")
        
        for server_name, server_config in servers_config.items():
            self.clients[server_name] = MCPClient(server_config)
            print(f"✅ Initialized client for: {server_name}")
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool on a specific MCP server"""
        if server_name not in self.clients:
            raise ValueError(f"Server '{server_name}' not found")
        
        client = self.clients[server_name]
        async with client.connect():
            return await client.call_tool(tool_name, arguments)


# ============================================================================
# INTELLIGENT AGENT WITH GEMINI (NO FALLBACK)
# ============================================================================

import json
import google.generativeai as genai
from fastapi import HTTPException
from typing import Dict, Any


class CalculatorAgent:
    """
    Intelligent multi-step calculator agent using Gemini AI.
    Supports tool chaining and additional math reasoning.
    """

    def __init__(
        self,
        mcp_client,
        server_name: str,
        gemini_api_key: str,
        model_name: str = "gemini-2.5-flash"
    ):
        self.mcp_client = mcp_client
        self.server_name = server_name
        self.tools = []

        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"🤖 CalculatorAgent initialized with Gemini model: {model_name}")

    async def initialize(self):
        async with self.mcp_client.connect():
            self.tools = await self.mcp_client.get_tools()
            print(f"🔧 Loaded {len(self.tools)} tools from {self.server_name}")

    # -----------------------------
    # UTILITY: Extract Final Answer
    # -----------------------------
    def _extract_numeric_answer(self, response_text: str, decimal_places: int = 2) -> str:
        """Extract numeric answer - Gemini now returns only numbers"""
        import re
        
        text = response_text.strip()
        
        # Try to extract a number directly
        numbers = re.findall(r'-?\d+\.?\d*', text)
        
        if numbers:
            try:
                # Get the first/only number (since Gemini returns just the number now)
                value = float(numbers[0])
                return str(round(value, decimal_places))
            except:
                pass
        
        # If no number found, return as-is
        return text

    # -----------------------------
    # STEP 1: Intelligent Router
    # -----------------------------
    async def _route_with_gemini(self, user_message: str) -> Dict[str, Any]:

        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in self.tools
        ])

        system_prompt = f"""
You are an intelligent math agent that routes requests to calculators.

Available tools:
- bodma_calculate: Calculate BODMA(a,b) = (a^b) / (a*b)
- codma_calculate: Calculate CODMA(a,b) = (a*b) / (a^b)

CRITICAL RULES:
1. If user mentions "BODMA AND CODMA" or "BOTH" → use tool_name: "both"
2. If user asks for "PRODUCT of BODMA and CODMA" → use tool_name: "both" (you'll compute product)
3. If user asks for "SUM of BODMA and CODMA" → use tool_name: "both" (you'll compute sum)
4. If only BODMA mentioned → use "bodma_calculate"
5. If only CODMA mentioned → use "codma_calculate"

Examples:
- "BODMA of 2 and 3" → tool_name: "bodma_calculate"
- "CODMA of 2 and 3" → tool_name: "codma_calculate"
- "BODMA and CODMA of 2 and 3" → tool_name: "both"
- "Product of BODMA and CODMA of 2 and 3" → tool_name: "both"
- "Sum of BODMA and CODMA of 2 and 3" → tool_name: "both"

Respond ONLY in JSON format:
{{
    "action": "use_tool",
    "tool_name": "bodma_calculate" or "codma_calculate" or "both",
    "arguments": {{"a": number, "b": number}},
    "reasoning": "brief explanation"
}}
"""

        full_prompt = f"{system_prompt}\n\nUser: {user_message}\n\nJSON Response:"

        try:
            response = self.model.generate_content(full_prompt)
            response_text = response.text.strip()

            if response_text.startswith("```"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()

            return json.loads(response_text)

        except Exception as e:
            print(f"❌ Routing error: {e}")
            return {
                "action": "respond_directly",
                "direct_response": "I couldn't understand that request. Please try again.",
                "reasoning": "Routing failure"
            }

    # -----------------------------
    # STEP 2: Main Agent Logic
    # -----------------------------
    async def run(self, user_message: str) -> Dict[str, Any]:

        if not self.tools:
            await self.initialize()

        decision = await self._route_with_gemini(user_message)

        print(f"🧠 Decision: {decision.get('reasoning')}")

        # -----------------------------------
        # CASE 1: Tool Required
        # -----------------------------------
        if decision.get("action") == "use_tool":

            tool_name = decision.get("tool_name")
            arguments = decision.get("arguments")

            try:
                async with self.mcp_client.connect():
                    # Handle "both" case - execute both BODMA and CODMA
                    if tool_name == "both":
                        bodma_result = await self.mcp_client.call_tool("bodma_calculate", arguments)
                        codma_result = await self.mcp_client.call_tool("codma_calculate", arguments)
                        
                        try:
                            bodma_parsed = json.loads(bodma_result)
                        except:
                            bodma_parsed = {"result": bodma_result}
                        
                        try:
                            codma_parsed = json.loads(codma_result)
                        except:
                            codma_parsed = {"result": codma_result}
                        
                        # 🔥 SECOND GEMINI PASS for both results
                        follow_up_prompt = f"""
User originally asked:
{user_message}

You executed two tools with arguments {arguments}:
BODMA result: {bodma_parsed}
CODMA result: {codma_parsed}

Now determine what the user is asking for:
- If asking for "both", add them: BODMA + CODMA
- If asking for "product", multiply them: BODMA * CODMA
- If asking for "difference", subtract: BODMA - CODMA
- Otherwise return the appropriate value

IMPORTANT: Return ONLY the final numeric answer. Nothing else.
- Just the number, rounded to 2 decimal places
- No text, explanation, or words
"""
                    else:
                        # Single tool execution
                        tool_result = await self.mcp_client.call_tool(tool_name, arguments)

                        try:
                            parsed_result = json.loads(tool_result)
                        except:
                            parsed_result = {"result": tool_result}

                        # 🔥 SECOND GEMINI PASS
                        follow_up_prompt = f"""
User originally asked:
{user_message}

You executed tool:
{tool_name}
With arguments:
{arguments}

Tool result:
{parsed_result}

IMPORTANT: Return ONLY the final numeric answer. Nothing else.
- If calculation is complete, respond with just the number
- If additional math is needed, perform it and respond with just the result number
- Examples of correct responses: "1.33" or "0.75" or "2.08"
- Do NOT include any text, explanation, or words
- Just the number, rounded to 2 decimal places
"""

                final_response = self.model.generate_content(follow_up_prompt)
                response_text = final_response.text.strip()
                
                # Extract only the numeric answer (rounded to 2 decimal places)
                numeric_answer = self._extract_numeric_answer(response_text, decimal_places=2)

                return {
                    "type": "direct_response",
                    "response": numeric_answer,
                    "reasoning": decision.get("reasoning", "")
                }

            except Exception as e:
                return {
                    "type": "error",
                    "error": str(e),
                    "reasoning": decision.get("reasoning", "")
                }

        # -----------------------------------
        # CASE 2: Direct Response
        # -----------------------------------
        else:
            return {
                "type": "direct_response",
                "response": decision.get("direct_response"),
                "reasoning": decision.get("reasoning")
            }


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User message for the agent")
    server_name: str = Field(default="calculator-server", description="MCP server to use")


class SimplifiedChatResponse(BaseModel):
    """Simplified response model for chat endpoint - returns only the answer as float"""
    response: float = Field(..., description="Numeric answer from agent as float")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Agent's response")
    server_name: str = Field(..., description="Server that processed the request")
    decision_type: str = Field(..., description="Type of decision: tool_execution, direct_response, or error")
    reasoning: str = Field(default="", description="Agent's reasoning (from Gemini)")
    tool_name: Optional[str] = Field(default=None, description="Tool that was executed (if any)")
    result: Optional[Dict] = Field(default=None, description="Tool execution result (if any)")


class ToolInfo(BaseModel):
    """Information about a tool"""
    name: str
    description: str
    input_schema: Dict[str, Any]


class ServerToolsResponse(BaseModel):
    """Response with tools from a server"""
    server_name: str
    tools: List[ToolInfo]


class ExecuteToolRequest(BaseModel):
    """Request to execute a tool directly"""
    server_name: str = Field(default="calculator-server", description="MCP server name")
    tool_name: str = Field(..., description="Name of the tool to execute")
    a: float = Field(..., description="First parameter (a)")
    b: float = Field(..., description="Second parameter (b)")


# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    global mcp_manager
    print("\n🚀 Starting Calculator Agent API with Gemini AI...")
    mcp_manager = MCPClientManager(config_path="config.json")
    await mcp_manager.initialize_clients()
    print("✅ MCP Manager ready")
    print("✅ Gemini AI enabled\n")
    
    yield
    
    # Shutdown
    print("\n🛑 Shutting down...")


app = FastAPI(
    title="Calculator Agent API with Gemini",
    description="FastAPI interface for calculator MCP agent with BODMA and CODMA tools powered by Google Gemini AI. Test the agent with Swagger UI below.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Health check and API info endpoint"""
    return {
        "status": "running",
        "title": "Calculator Agent API with Gemini",
        "description": "MCP Agent for BODMA and CODMA calculations powered by Google Gemini AI",
        "version": "2.0.0",
        "ai_model": "gemini-2.5-flash",
        "gemini_enabled": True,
        "endpoints": {
            "Chat with Agent": "/chat-agent (POST) - AI-powered tool routing with Gemini",
            "Execute BODMA": "/tools/bodma (POST) - Execute BODMA calculation directly",
            "Execute CODMA": "/tools/codma (POST) - Execute CODMA calculation directly",
            "List Servers": "/servers (GET) - List configured MCP servers",
            "List Tools": "/tools (GET) - List all available tools",
            "API Docs": "/docs - Swagger UI for testing",
            "ReDoc": "/redoc - Alternative API documentation"
        }
    }


@app.get("/servers", tags=["Info"])
async def list_servers():
    """List all configured MCP servers"""
    if not mcp_manager:
        raise HTTPException(status_code=500, detail="MCP Manager not initialized")
    
    return {
        "servers": list(mcp_manager.clients.keys()),
        "count": len(mcp_manager.clients)
    }


@app.get("/tools", response_model=List[ServerToolsResponse], tags=["Info"])
async def list_all_tools():
    """Get all tools from all MCP servers"""
    if not mcp_manager:
        raise HTTPException(status_code=500, detail="MCP Manager not initialized")
    
    all_tools = []
    
    for server_name, client in mcp_manager.clients.items():
        try:
            async with client.connect():
                tools = await client.get_tools()
                all_tools.append({
                    "server_name": server_name,
                    "tools": tools
                })
        except Exception as e:
            print(f"❌ Error fetching tools from {server_name}: {str(e)}")
    
    return all_tools


@app.get("/tools/{server_name}", response_model=ServerToolsResponse, tags=["Info"])
async def list_server_tools(server_name: str):
    """Get tools from a specific MCP server"""
    if not mcp_manager:
        raise HTTPException(status_code=500, detail="MCP Manager not initialized")
    
    if server_name not in mcp_manager.clients:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")
    
    client = mcp_manager.clients[server_name]
    
    try:
        async with client.connect():
            tools = await client.get_tools()
            return {
                "server_name": server_name,
                "tools": tools
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching tools: {str(e)}")


@app.post("/chat-agent", response_model=SimplifiedChatResponse, tags=["Agent"])
async def chat_with_agent(request: ChatRequest):
    """
    Chat with the Gemini-powered calculator agent
    
    Returns ONLY the numeric answer (rounded to 2 decimal places)
    
    **Example requests:**
    - "Calculate BODMA for 2 and 3"
    - "What is CODMA of 5 and 2?"
    - "BODMA 10 and 5"
    
    **Example response:**
    ```json
    {
        "response": "1.33"
    }
    ```
    """
    if not mcp_manager:
        raise HTTPException(status_code=500, detail="MCP Manager not initialized")
    
    server_name = request.server_name
    
    if server_name not in mcp_manager.clients:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")
    
    # Get or create agent for this server (Gemini-powered)
    if server_name not in agents:
        client = mcp_manager.clients[server_name]
        agents[server_name] = CalculatorAgent(
            client, 
            server_name,
            gemini_api_key=GEMINI_API_KEY,
            model_name="gemini-2.5-flash"
        )
        await agents[server_name].initialize()
    
    agent = agents[server_name]
    
    try:
        # Get agent response (powered by Gemini)
        result = await agent.run(request.message)
        
        # Extract numeric response only
        if result["type"] == "direct_response":
            raw_response = result["response"]
            # Use the agent's extraction method to get ONLY the number
            numeric_answer = agent._extract_numeric_answer(raw_response, decimal_places=2)
            response_text = numeric_answer
        else:
            response_text = f"Error: {result.get('error', 'Unknown error')}"
        
        # Convert response to float
        try:
            response_float = float(response_text)
        except:
            response_float = 0.0
        
        return SimplifiedChatResponse(
            response=response_float
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.post("/tools/bodma", tags=["Tools"])
async def execute_bodma(request: ExecuteToolRequest):
    """
    Execute BODMA calculation directly
    
    **BODMA Formula:** (a^b) / (a*b)
    
    **Example:**
    - a=2, b=3 → (2^3) / (2*3) = 8/6 = 1.333...
    - a=5, b=2 → (5^2) / (5*2) = 25/10 = 2.5
    """
    if not mcp_manager:
        raise HTTPException(status_code=500, detail="MCP Manager not initialized")
    
    try:
        result = await mcp_manager.call_tool(
            request.server_name,
            "bodma_calculate",
            {"a": request.a, "b": request.b}
        )
        
        # Parse result
        try:
            parsed_result = json.loads(result)
        except:
            parsed_result = {"raw_result": result}
        
        return {
            "operation": "BODMA",
            "formula": f"({request.a}^{request.b}) / ({request.a}*{request.b})",
            "parameters": {"a": request.a, "b": request.b},
            "result": parsed_result
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")


@app.post("/tools/codma", tags=["Tools"])
async def execute_codma(request: ExecuteToolRequest):
    """
    Execute CODMA calculation directly
    
    **CODMA Formula:** (a*b) / (a^b)
    
    **Example:**
    - a=2, b=3 → (2*3) / (2^3) = 6/8 = 0.75
    - a=5, b=2 → (5*2) / (5^2) = 10/25 = 0.4
    """
    if not mcp_manager:
        raise HTTPException(status_code=500, detail="MCP Manager not initialized")
    
    try:
        result = await mcp_manager.call_tool(
            request.server_name,
            "codma_calculate",
            {"a": request.a, "b": request.b}
        )
        
        # Parse result
        try:
            parsed_result = json.loads(result)
        except:
            parsed_result = {"raw_result": result}
        
        return {
            "operation": "CODMA",
            "formula": f"({request.a}*{request.b}) / ({request.a}^{request.b})",
            "parameters": {"a": request.a, "b": request.b},
            "result": parsed_result
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("🤖 CALCULATOR AGENT API - Powered by Google Gemini AI")
    print("=" * 70)
    print("📡 Starting on: http://localhost:8000")
    print("📚 Swagger UI (Test API): http://localhost:8000/docs")
    print("📖 ReDoc (Alternative): http://localhost:8000/redoc")
    print("🤖 AI Model: gemini-2.5-flash")
    print("=" * 70)
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
        log_level="info"
    )