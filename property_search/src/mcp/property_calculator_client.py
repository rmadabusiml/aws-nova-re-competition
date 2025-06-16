import os
from mcp import StdioServerParameters

from InlineAgent.tools import MCPHttp
from InlineAgent.action_group import ActionGroup
from InlineAgent.agent import InlineAgent

async def main():
    property_calculator_mcp_client = await MCPHttp.create(url="http://localhost:8000/sse")
    print("Invoked Property Calculator MCP Tool here")

    try:
        property_calculator_action_group = ActionGroup(
            name="PropertyCalculatorGroup",
            mcp_clients=[property_calculator_mcp_client],
        )
        response = await InlineAgent(
            foundation_model=os.environ.get('property_calculator_mcp_tool_llm', 'amazon.nova-pro-v1:0'),
            instruction="""You are a friendly assistant that is responsible for resolving user queries related to Property calculator that is capable of performing different calcutions such as buyability, affordability, and mortgage calculator. Keep the response short and to the point within in 5 sentences long that is easy for any speech assistant to respond to the user""",
            agent_name="property_calculator_agent",
            action_groups=[
                property_calculator_action_group,
            ],
        ).invoke(
            input_text="Estimate how much house I can buy with my annual income of $255K with monthly debts of $500 and will do $75K as down payment with 30 years loan term"
        )
    finally:
        await property_calculator_mcp_client.cleanup()

    print("Property Calculator Tool Response: ", response)
    

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
