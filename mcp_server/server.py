#!/usr/bin/env python3
"""
Calculator MCP Server
"""

from mcp.server.fastmcp import FastMCP
import os


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
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    
    print("=" * 60)
    print("🤖 CALCULATOR MCP SERVER")
    print("=" * 60)
    print(f"📡 Running on port: {port}")
    print(f"🔗 SSE Endpoint: http://0.0.0.0:{port}/sse")
    print("=" * 60)
    
    # Use FastMCP as the ASGI app directly
    uvicorn.run(
        mcp,
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
