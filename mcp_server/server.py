#!/usr/bin/env python3
"""
Calculator MCP Server - PROPER HTTP TRANSPORT
Uses FastAPI with proper HTTP endpoints for MCP
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import json
import os
import uvicorn
from mcp.server.fastmcp import FastMCP
from contextlib import asynccontextmanager

# ============================================================================
# CREATE MCP SERVER
# ============================================================================

mcp = FastMCP(name="Calculator MCP")


# ============================================================================
# DEFINE CALCULATION TOOLS
# ============================================================================

@mcp.tool()
def bodma(a: float, b: float) -> float:
    """Calculates (a^b) / (a * b)"""
    try:
        denominator = a * b
        if denominator == 0:
            return 0.0
        result = (a ** b) / denominator
        return round(result, 2)
    except:
        return 0.0


@mcp.tool()
def codma(a: float, b: float) -> float:
    """Calculates (a * b) + (a / b)"""
    try:
        if b == 0:
            return 0.0
        result = (a * b) + (a / b)
        return round(result, 2)
    except:
        return 0.0


# ============================================================================
# CREATE FASTAPI APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("✅ Server starting...")
    yield
    # Shutdown
    print("✅ Server shutting down...")


app = FastAPI(
    title="Calculator MCP Server",
    description="MCP Server with HTTP transport",
    lifespan=lifespan
)


# ============================================================================
# HTTP ENDPOINTS (Not SSE, just plain HTTP)
# ============================================================================

@app.get("/")
async def root():
    return {
        "status": "MCP Server Running",
        "version": "1.0",
        "tools": ["bodma", "codma"]
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/tools/bodma")
async def call_bodma(a: float, b: float):
    """Direct HTTP endpoint for bodma"""
    try:
        result = bodma(a, b)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/tools/codma")
async def call_codma(a: float, b: float):
    """Direct HTTP endpoint for codma"""
    try:
        result = codma(a, b)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tools")
async def list_tools():
    """List available tools"""
    return {
        "tools": [
            {
                "name": "bodma",
                "description": "Calculates (a^b) / (a * b)",
            },
            {
                "name": "codma",
                "description": "Calculates (a * b) + (a / b)",
            }
        ]
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    
    print("=" * 60)
    print("🤖 CALCULATOR MCP SERVER (HTTP Transport)")
    print("=" * 60)
    print(f"📡 Running on port: {port}")
    print(f"🔗 Health: http://0.0.0.0:{port}/health")
    print(f"🔗 Tools: http://0.0.0.0:{port}/tools")
    print(f"🔗 Docs: http://0.0.0.0:{port}/docs")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
