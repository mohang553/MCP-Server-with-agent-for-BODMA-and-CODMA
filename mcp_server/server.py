#!/usr/bin/env python3
"""
Calculator MCP Server - EXACT FIX
The issue: FastMCP object is not callable
Solution: Use create_fastapi_app() instead of mounting directly
"""

from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
import os
import uvicorn

# ============================================================================
# CREATE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Calculator MCP Server",
    description="MCP Server with BODMA and CODMA calculation tools"
)


# ============================================================================
# HEALTH CHECK ENDPOINT
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


# ============================================================================
# CREATE MCP SERVER
# ============================================================================

mcp = FastMCP(name="Calculator MCP")


# ============================================================================
# DEFINE CALCULATION TOOLS
# ============================================================================

@mcp.tool()
def bodma(a: float, b: float) -> float:
    """
    BODMA Calculation: (a^b) / (a * b)
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Result of (a^b) / (a * b), or 0 if division by zero
    
    Example:
        bodma(2, 3) = (2^3) / (2*3) = 8/6 = 1.33
    """
    try:
        denominator = a * b
        if denominator == 0:
            return 0.0
        
        result = (a ** b) / denominator
        return round(result, 2)
    except Exception as e:
        return 0.0


@mcp.tool()
def codma(a: float, b: float) -> float:
    """
    CODMA Calculation: (a * b) + (a / b)
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Result of (a * b) + (a / b), or 0 if division by zero
    
    Example:
        codma(6, 2) = (6*2) + (6/2) = 12 + 3 = 15
    """
    try:
        if b == 0:
            return 0.0
        
        result = (a * b) + (a / b)
        return round(result, 2)
    except Exception as e:
        return 0.0


# ============================================================================
# MOUNT MCP TO FASTAPI - CORRECT WAY
# ============================================================================

# THIS IS THE FIX: Use create_fastapi_app() method, not direct mount
mcp_app = mcp.create_fastapi_app()
app.mount("/mcp", mcp_app)


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    
    print("=" * 60)
    print("🤖 CALCULATOR MCP SERVER")
    print("=" * 60)
    print(f"📡 Running on port: {port}")
    print(f"🔗 Root: http://0.0.0.0:{port}/")
    print(f"🔗 SSE Endpoint: http://0.0.0.0:{port}/mcp/sse")
    print(f"🔗 Health: http://0.0.0.0:{port}/health")
    print("=" * 60)
    
    uvicorn.run(
        "server_EXACT_FIX:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
