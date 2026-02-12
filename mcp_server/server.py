from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
import os

# -----------------------------
# Create FastAPI App
# -----------------------------
app = FastAPI()

@app.get("/")
async def root():
    return {"status": "MCP Server Running"}

# -----------------------------
# Create MCP Server (NO description argument)
# -----------------------------
mcp = FastMCP(name="Calculator MCP")

# -----------------------------
# Define Tools
# -----------------------------

@mcp.tool()
def bodma(a: float, b: float) -> float:
    """
    Calculates (a^b) / (a * b)
    """
    if a * b == 0:
        return 0
    return round((a ** b) / (a * b), 2)


@mcp.tool()
def codma(a: float, b: float) -> float:
    """
    Calculates (a * b) + (a / b)
    """
    if b == 0:
        return 0
    return round((a * b) + (a / b), 2)

# -----------------------------
# Mount MCP to FastAPI
# -----------------------------
app.mount("/mcp", mcp)

# -----------------------------
# Run Server (Render compatible)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
