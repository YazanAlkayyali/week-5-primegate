import asyncio
import sys

from mcp import Client, StdioServerParameters
from mcp.types import TextContent


server = StdioServerParameters(
    command=sys.executable,
    args=["mcp_server.py"],
)


async def main() -> None:
    async with Client(server) as client:
        tools = await client.list_tools()

        print("Tools exposed by MCP server:")

        for tool in tools.tools:
            print(f"- {tool.name}")
            print(f"  Description: {tool.description}")
            print(f"  Input schema: {tool.input_schema}")

        result = await client.call_tool(
            "lookup_leave_balance",
            {
                "employee_id": "E1002",
            },
        )

        print("\nTool result:")

        for block in result.content:
            if isinstance(block, TextContent):
                print(block.text)


if __name__ == "__main__":
    asyncio.run(main())