import asyncio
import os
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

async def main():
    url = os.getenv("MCP_SHEETS_URL", "http://localhost:8790/sse")

    async with sse_client(url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            print("initialize:", init)

            tools = await session.list_tools()
            print("tools/list:", tools)

            # Assumes GOOGLE_SHEETS_SPREADSHEET_ID is set in the server env
            rows = await session.call_tool("list_rows", {"limit": 5})
            print("list_rows:", rows)

            vals = await session.call_tool("fetch_values", {"range_a1": "A1:C5"})
            print("fetch_values:", vals)

if __name__ == "__main__":
    asyncio.run(main())
