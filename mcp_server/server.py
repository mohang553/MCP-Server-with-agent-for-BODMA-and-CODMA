import os
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from mcp.server.fastmcp import FastMCP

# ============================================================
# Create FastAPI app
# ============================================================

app = FastAPI(title="Calculator MCP Server")

# ============================================================
# IMPORTANT: Allow Render host (Fixes 421 error)
# ============================================================

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # allow all hosts (Render safe)
)

# Optional but recommended for external calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Create MCP Server
# ============================================================

mcp = FastMCP(
    name="Calculator MCP",
    description="Provides BODMA and CODMA calculation tools",
)

# ============================================================
# Tool 1: BODMA
# Formula: (a^b) / (a * b)
# ============================================================

@mcp.tool()
def BODMA(a: float, b: float) -> float:
    if a * b == 0:
        raise ValueError("a * b cannot be zero")
    return round((a ** b) / (a * b), 2)

# ============================================================
# Tool 2: CODMA
# Formula: (a * b) / (a^b)
# ============================================================

@mcp.tool()
def CODMA(a: float, b: float) -> float:
    if a ** b == 0:
        raise ValueError("a^b cannot be zero")
    return round((a * b) / (a ** b), 2)

# ============================================================
# Mount MCP HTTP transport
# This exposes:
#   GET  /mcp/sse
#   POST /mcp/messages
# ============================================================

app.mount("/mcp", mcp.asgi())

# ============================================================
# Health Check Route
# ============================================================

@app.get("/")
def health():
    return {"status": "MCP Server Running"}

# ============================================================
# Run Server (Render compatible)
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
