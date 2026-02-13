#!/usr/bin/env python3
"""
Calculator MCP Server - CORRECT FORMULAS
BODMA: (a^b) / (a*b)
CODMA: (a*b) / (a^b)
"""

from fastapi import FastAPI, HTTPException
import os
import uvicorn

app = FastAPI(
    title="Calculator MCP Server",
    description="MCP Server with BODMA and CODMA calculation tools - CORRECT FORMULAS"
)

@app.get("/")
async def root():
    return {
        "status": "MCP Server Running",
        "version": "1.0",
        "tools": ["bodma_calculate", "codma_calculate"],
        "formulas": {
            "bodma_calculate": "(a^b) / (a*b)",
            "codma_calculate": "(a*b) / (a^b)"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/tools/bodma_calculate")
async def bodma_calculate(a: float, b: float):
    """
    BODMA: (a^b) / (a*b)
    
    Example: bodma(2, 3) = (2^3) / (2*3) = 8/6 = 1.33
    """
    try:
        denominator = a * b
        if denominator == 0:
            return {"result": 0.0}
        
        numerator = a ** b
        result = numerator / denominator
        return {"result": round(result, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/tools/codma_calculate")
async def codma_calculate(a: float, b: float):
    """
    CODMA: (a*b) / (a^b)
    
    Example: codma(2, 3) = (2*3) / (2^3) = 6/8 = 0.75
    """
    try:
        numerator = a * b
        denominator = a ** b
        
        if denominator == 0:
            return {"result": 0.0}
        
        result = numerator / denominator
        return {"result": round(result, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tools")
async def list_tools():
    """List available tools"""
    return {
        "tools": [
            {
                "name": "bodma_calculate",
                "description": "Calculates (a^b) / (a*b)",
            },
            {
                "name": "codma_calculate",
                "description": "Calculates (a*b) / (a^b)",
            }
        ]
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    
    print("=" * 70)
    print("🤖 CALCULATOR MCP SERVER (CORRECT FORMULAS)")
    print("=" * 70)
    print(f"📡 Running on port: {port}")
    print(f"")
    print(f"BODMA: (a^b) / (a*b)")
    print(f"CODMA: (a*b) / (a^b)")
    print(f"")
    print(f"🔗 Health: http://0.0.0.0:{port}/health")
    print(f"🔗 Tools: http://0.0.0.0:{port}/tools")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
