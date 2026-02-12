#!/usr/bin/env python3

import os
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from mcp.server.fastmcp import FastMCP
import uvicorn

# ============================================================================
# MCP
# ============================================================================

mcp = FastMCP("calculator-server")

@mcp.tool()
def bodma_calculate(a: float, b: float) -> float:
    return (a ** b) / (a * b)

@mcp.tool()
def codma_calculate(a: float, b: float) -> float:
    return (a * b) / (a ** b)


# ============================================================================
# FASTAPI
# ============================================================================

app = FastAPI(title="Calculator MCP HTTP Server")

# 🔥 VERY IMPORTANT FOR RENDER
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # allow Render proxy host
)

# Mount MCP
app.mount("/mcp", mcp.sse_app())


@app.get("/")
def root():
    return {"status": "running"}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )
