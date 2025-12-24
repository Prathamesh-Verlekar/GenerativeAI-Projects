import asyncio
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

async def main():
    url = "http://localhost:8787/sse"

    async with sse_client(url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 1) initialize handshake
            init = await session.initialize()
            print("initialize:", init)

            # 2) list tools
            tools = await session.list_tools()
            print("tools/list:", tools)

            # 3) call get_quote
            quote = await session.call_tool("get_quote", {"symbol": "AAPL"})
            print("get_quote:", quote)

            # 4) call get_history
            hist = await session.call_tool("get_history", {"symbol": "AAPL", "days": 5})
            print("get_history:", hist)

            # 5) call search_ticker
            srch = await session.call_tool("search_ticker", {"query": "Tesla"})
            print("search_ticker:", srch)

if __name__ == "__main__":
    asyncio.run(main())
