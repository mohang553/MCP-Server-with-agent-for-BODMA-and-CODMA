#!/usr/bin/env python3
"""
Test script to verify MCP server connection
Run this to debug the connection issue
"""
import asyncio
import json
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_connection():
    """Test if we can connect to the MCP server"""
    
    # Load config
    with open("config.json", 'r') as f:
        config = json.load(f)
    
    server_config = config["mcpServers"]["calculator-server"]
    
    print("=" * 70)
    print("🔍 Testing MCP Server Connection")
    print("=" * 70)
    print(f"Command: {server_config['command']}")
    print(f"Args: {server_config['args']}")
    print(f"Server script: {server_config['args'][0]}")
    print()
    
    # Check if server file exists
    server_path = Path(server_config['args'][0])
    if not server_path.exists():
        print(f"❌ ERROR: Server file not found at: {server_path}")
        print(f"   Absolute path: {server_path.absolute()}")
        return
    else:
        print(f"✅ Server file exists: {server_path}")
    
    print()
    print("Attempting to connect to MCP server...")
    print("-" * 70)
    
    try:
        server_params = StdioServerParameters(
            command=server_config["command"],
            args=server_config["args"],
            env=server_config.get("env")
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                print("✅ Successfully connected to MCP server!")
                print()
                
                # List tools
                tools_list = await session.list_tools()
                print(f"📋 Found {len(tools_list.tools)} tools:")
                for tool in tools_list.tools:
                    print(f"   - {tool.name}: {tool.description}")
                
                print()
                
                # Test BODMA calculation
                print("🧪 Testing BODMA calculation (2, 3)...")
                result = await session.call_tool("bodma_calculate", {"a": 2, "b": 3})
                
                if result.content:
                    response_text = "\n".join([
                        item.text for item in result.content 
                        if hasattr(item, 'text')
                    ])
                    print("✅ BODMA Result:")
                    print(response_text)
                
                print()
                print("=" * 70)
                print("✅ All tests passed! MCP server is working correctly.")
                print("=" * 70)
                
    except Exception as e:
        print(f"❌ Connection failed!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print()
        print("Troubleshooting:")
        print("1. Make sure server.py has no syntax errors")
        print("2. Try running server.py directly: python server.py")
        print("3. Check that all required packages are installed:")
        print("   pip install mcp fastapi uvicorn")

if __name__ == "__main__":
    asyncio.run(test_mcp_connection())