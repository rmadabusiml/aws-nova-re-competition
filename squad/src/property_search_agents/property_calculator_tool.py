from agent_squad.utils import AgentTools, AgentTool
from dotenv import load_dotenv
import os

from InlineAgent.tools.mcp import MCPHttp, MCPStdio
from mcp import StdioServerParameters

from InlineAgent.action_group import ActionGroup
from InlineAgent.agent import InlineAgent
from InlineAgent import AgentAppConfig
from agent_squad.types import ConversationMessage, ParticipantRole
from typing import List, Dict, Any
# from langfuse_callbacks import ToolsCallbacks

config = AgentAppConfig()

async def get_property_calculator(query:str):
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
            input_text=query
        )
    finally:
        await property_calculator_mcp_client.cleanup()

    print("Property Calculator Tool Response: ", response)
    return response

property_calculator_tools:AgentTools = AgentTools(tools=[AgentTool(name="PropertyCalculator_Tool",
        description="Provides different property calculator tools such as buyability, affordability, and mortgage calculator.",
        func=get_property_calculator
    )],
    # callbacks=ToolsCallbacks()
    )
