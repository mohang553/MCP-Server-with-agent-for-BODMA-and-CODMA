#!/usr/bin/env python3
"""
HTTP MCP Server for Calculator Tools (BODMA and CODMA)
Using FastMCP with SSE Transport
"""

import json
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
import uvicorn

# ============================================================================
# FASTAPI + MCP SETUP
# ============================================================================

app = FastAPI(
    title="Calculator MCP HTTP Server",
    description="HTTP-based MCP Server for BODMA and CODMA tools",
    version="2.0.0"
)

mcp = FastMCP("calculator-server")

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

@mcp.tool()
def bodma_calculate(a: float, b: float) -> dict:
    """
    BODMA Calculation: (a^b) / (a*b)
    """
    try:
        if a == 0 and b == 0:
            return {
                "status": "error",
                "message": "Cannot calculate 0^0 and division by 0",
                "result": None
            }

        numerator = a ** b
        denominator = a * b

        if denominator == 0:
            return {
                "status": "error",
                "message": "Cannot divide by 0 (a*b = 0)",
                "result": None
            }

        result = numerator / denominator

        return {
            "status": "success",
            "formula": f"({a}^{b}) / ({a}*{b})",
            "numerator": numerator,
            "denominator": denominator,
            "result": result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "result": None
        }


@mcp.tool()
def codma_calculate(a: float, b: float) -> dict:
    """
    CODMA Calculation: (a*b) / (a^b)
    """
    try:
        if a == 0 and b <= 0:
            return {
                "status": "error",
                "message": "Invalid 0^b for b <= 0",
                "result": None
            }

        numerator = a * b
        denominator = a ** b

        if denominator == 0:
            return {
                "status": "error",
                "message": "Cannot divide by 0 (a^b = 0)",
                "result": None
            }

        result = numerator / denominator

        return {
            "status": "success",
            "formula": f"({a}*{b}) / ({a}^{b})",
            "numerator": numerator,
            "denominator": denominator,
            "result": result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "result": None
        }


# ============================================================================
# MOUNT MCP TO FASTAPI
# ============================================================================

# This exposes MCP over HTTP using SSE transport
app.mount("/mcp/", mcp.sse_app())


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/")
def health():
    return {
        "status": "running",
        "message": "Calculator MCP HTTP Server is running",
        "mcp_endpoint": "http://localhost:8001/mcp"
    }



# ============================================================================
# REST ENDPOINT TO LIST TOOLS
# ============================================================================

@app.get("/tools")
def list_tools():
    """
    Expose available tools via normal REST endpoint
    """
    return {
        "server": "calculator-server",
        "tools": [
            {
                "name": "bodma_calculate",
                "description": "Calculate (a^b) / (a*b)",
                "input_schema": {
                    "a": "number",
                    "b": "number"
                }
            },
            {
                "name": "codma_calculate",
                "description": "Calculate (a*b) / (a^b)",
                "input_schema": {
                    "a": "number",
                    "b": "number"
                }
            }
        ]
    }



# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 CALCULATOR MCP HTTP SERVER")
    print("=" * 60)
    print("📡 Running on: http://localhost:8001")
    print("🔌 MCP Endpoint: http://localhost:8001/mcp")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
