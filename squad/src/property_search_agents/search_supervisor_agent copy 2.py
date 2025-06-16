from typing import Any
import sys, asyncio, uuid
import os
from datetime import datetime, timezone
from agent_squad.utils import Logger
from agent_squad.agents import SupervisorAgent, SupervisorAgentOptions, AgentResponse, AgentStreamResponse
from agent_squad.agents import LambdaAgent, LambdaAgentOptions
from agent_squad.agents import AmazonBedrockAgent, AmazonBedrockAgentOptions
from agent_squad.agents import BedrockLLMAgent, BedrockLLMAgentOptions
from agent_squad.classifiers import ClassifierResult
from agent_squad.types import ConversationMessage
from agent_squad.storage import InMemoryChatStorage
from agent_squad.orchestrator import AgentSquad, AgentSquadConfig
import json
from typing import List, Optional, Dict
import logging

from property_search_agents.property_calculator_tool import property_calculator_tools
from langfuse.decorators import observe, langfuse_context
from langfuse import Langfuse
# from langfuse_callbacks import LLMAgentCallbacks, BedrockClassifierCallbacks

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

memory_storage = InMemoryChatStorage()
langfuse = Langfuse()

from langfuse.decorators import observe, langfuse_context
from langfuse import Langfuse
from typing import Optional, Any
import os
from dotenv import load_dotenv
from uuid import UUID
from datetime import datetime, timezone
from agent_squad.utils import AgentTools, AgentToolCallbacks, AgentTool
from agent_squad.classifiers import BedrockClassifier, BedrockClassifierOptions, ClassifierCallbacks, ClassifierResult
from agent_squad.agents import AgentCallbacks
import logging

# langfuse = Langfuse()

class BedrockClassifierCallbacks(ClassifierCallbacks):

    async def on_classifier_start(
        self,
        name,
        input: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            inputs = []
            inputs.append({'role':'system', 'content':kwargs.get('system')})
            inputs.extend([{'role':'user', 'content':input}])
            langfuse_context.update_current_observation(
                name=name,
                start_time=datetime.now(timezone.utc),
                input=inputs,
                model=kwargs.get('modelId'),
                model_parameters=kwargs.get('inferenceConfig'),
                tags=tags,
                metadata=metadata
            )
        except Exception as e:
            logging.error(e)
            pass

    async def on_classifier_stop(
        self,
        name,
        output: ClassifierResult,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            langfuse_context.update_current_observation(
                output={'role':'assistant', 'content':{
                            'selected_agent' : output.selected_agent.name if output.selected_agent is not None else 'No agent selected',
                            'confidence' : output.confidence,
                        }
                },
                end_time=datetime.now(timezone.utc),
                name=name,
                tags=tags,
                metadata=metadata,
                usage={
                    'input':kwargs.get('usage',{}).get('inputTokens'),
                    "output": kwargs.get('usage', {}).get('outputTokens'),
                    "total": kwargs.get('usage', {}).get('totalTokens')
                },
            )
        except Exception as e:
            logging.error(e)
            pass


class LLMAgentCallbacks(AgentCallbacks):

    async def on_agent_start(
        self,
        agent_name,
        payload_input: Any,
        messages: list[Any],
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            print("Inside on_agent_start")
            print(payload_input)
            langfuse_context.update_current_observation(
                input=payload_input,
                start_time=datetime.now(timezone.utc),
                name=agent_name,
                tags=tags,
                metadata=metadata
            )
        except Exception as e:
            logging.error(e)
            pass

    async def on_agent_end(
        self,
        agent_name,
        response: Any,
        messages:list[Any],
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            langfuse_context.update_current_observation(
                end_time=datetime.now(timezone.utc),
                name=agent_name,
                user_id=kwargs.get('user_id'),
                session_id=kwargs.get('session_id'),
                output=response,
                tags=tags,
                metadata=metadata
            )
        except Exception as e:
            logging.error(e)
            pass

    async def on_llm_start(
        self,
        name:str,
        payload_input: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        logging.debug('on_llm_start')


    @observe(as_type='generation', capture_input=False)
    async def on_llm_end(
        self,
        name:str,
        output: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            msgs = []
            msgs.append({'role':'system', 'content': kwargs.get('input').get('system')})
            msgs.extend(kwargs.get('input').get('messages'))
            langfuse_context.update_current_observation(
                name=name,
                input=msgs,
                output=output,
                model=kwargs.get('input').get('modelId'),
                model_parameters=kwargs.get('inferenceConfig'),
                usage={
                    'input':kwargs.get('usage',{}).get('inputTokens'),
                    "output": kwargs.get('usage', {}).get('outputTokens'),
                    "total": kwargs.get('usage', {}).get('totalTokens')
                },
                tags=tags,
                metadata=metadata
            )
        except Exception as e:
            logging.error(e)
            pass


class ToolsCallbacks(AgentToolCallbacks):

    @observe(as_type='span', name='on_tool_start', capture_input=False)
    async  def on_tool_start(
        self,
        tool_name,
        payload_input: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        langfuse_context.update_current_observation(
            name=tool_name,
            input=input
        )

    @observe(as_type='span', name='on_tool_end', capture_input=False)
    async def on_tool_end(
        self,
        tool_name,
        payload_input: Any,
        output: dict,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        langfuse_context.update_current_observation(
            input=payload_input,
            name=tool_name,
            output=output
        )

# property_search_agent = AmazonBedrockAgent(AmazonBedrockAgentOptions(
#     name='Property Search Agent',
#     description='You are a Property Assistant that helps users find and analyze real estate properties. Your capabilities include retrieving property ID for a given physical address, and retrieving core details (price, size, bedrooms/bathrooms) of a property, neighborhood, climate risk factors, interior, exterior, sales history, tax records, etc.',
#     agent_id='7BFZUJUBHI',
#     agent_alias_id='JSLSXE2LKO',
#     region='us-east-1',
#     callbacks=LLMAgentCallbacks()
# ))

# property_kb_retriever_agent = LambdaAgent(LambdaAgentOptions(
#     name='Property KB Retriever Agent',
#     description='You are a Property KB Retriever Agent that helps users to provide property address and id based on their natural language property search characteristics. This agent will do semantic search on property descriptions stored in the knowledge base matching user query. ',
#     function_name='property_kb_retriever',
#     function_region='us-east-1',
#     callbacks=LLMAgentCallbacks()
# ))

# property_calculator_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
#     name='Property Calculator Agent',
#     description='Agent specialized in providing property calculators such as buyability, affordability, and mortgage calculator based on user provided inputs such as annual income, down payment, monthly debt payments, credit score, location, loan term, interest rate, etc. Not all calculators require all inputs. The respective tools use default values for missing inputs.',
#     model_id = os.environ.get('property_calculator_agent_llm', 'amazon.nova-pro-v1:0'),
#     tool_config={
#         'tool': property_calculator_tools,
#         'toolMaxRecursions': 5,
#     },
#     guardrail_config={
#         'guardrailIdentifier': '3pt5fpo4foxt',
#         'guardrailVersion': '1'
#     },
#     callbacks=LLMAgentCallbacks()
# ))

# property_appraiser_agent = LambdaAgent(LambdaAgentOptions(
#     name='Property Appraiser Agent',
#     description='You are a Property Appraiser Agent that helps users to analyze a home inspection report to determine the Fannie Mae property Condition (C1–C5) and Quality (Q1–Q5) ratings with explanations. It also provides a detailed repair cost breakdown for major issues and the total estimated repair cost. Keep the response short and to the point within in 6 to 8 sentences long that is easy for any speech assistant to respond to the user',
#     function_name='fn_property_appraiser',
#     function_region='us-east-1',
#     callbacks=LLMAgentCallbacks()
# ))

# property_mortgage_contract_assessor_agent = LambdaAgent(LambdaAgentOptions(
#     name='Property Mortgage Contract Assessor Agent',
#     description='You are a Property Mortgage Contract Assessor Agent that helps users to compare two mortgage contracts and provide the following: 1. Which contract is more borrower-friendly and why. 2. Highlight key differences in the following clauses: - Prepayment penalties - Escrow management - Default handling and grace periods - Dispute resolution process - Recourse and personal liability terms',
#     function_name='fn_property_mortgage_contract_assessor',
#     function_region='us-east-1',
#     callbacks=LLMAgentCallbacks()
# ))

# property_image_search_agent = LambdaAgent(LambdaAgentOptions(
#     name='Property Image Assistant Agent',
#     description='You are a property image assistant. When someone shares an image of a property, you find other images that look similar from a large collection. The user simply needs to provide the path or location of their image, and you’ll return matching images with easy-to-use links so they can quickly explore and compare visually alike properties. The output should only contain comma separated list of relative URLs of the similar images. DO NOT INCLUDE ANY ADDITIONAL INFORMATION OR EXPLANATION.',
#     function_name='fn_property_image_search',
#     function_region='us-east-1',
#     callbacks=LLMAgentCallbacks()
# ))

# property_image_canvas_agent = LambdaAgent(LambdaAgentOptions(
#     name='Property Image Canvas Agent',
#     description='You are a property image canvas agent. When someone shares an image of a property and asks to modify the image for a different look or replace some objects in the image with another object or modernize the look of different section of a property then use this agent. The user simply needs to provide the path or location of their image, and you’ll return modified image with easy-to-use links so they can quickly explore. The output should only contain modified images. DO NOT INCLUDE ANY ADDITIONAL INFORMATION OR EXPLANATION.',
#     function_name='fn_property_image_canvas',
#     function_region='us-east-1',
#     callbacks=LLMAgentCallbacks()
# ))

# supervisor = SupervisorAgent(SupervisorAgentOptions(
#     name="Property Search Supervisor Agent",
#     description=(
#         "You are a team supervisor managing a Property Search Agent, a Property KB Retriever Agent, a Property Calculator Agent, a Property Appraiser Agent, and a Property Mortgage Contract Assessor Agent. "
#         "For a specific property information related queries, use Property Search Agent. "
#         "For property search based on user property natural language description related queries, use Property KB Retriever Agent."
#         "For property calculator related queries such as buyability, affordability, and mortgage calculator, use Property Calculator Agent."
#         "For property appraiser related queries such as home inspection report analysis, use Property Appraiser Agent."
#         "For property mortgage contract related queries such as comparing two mortgage contracts, use Property Mortgage Contract Assessor Agent."
#         "When a user asks to provide similar property images by giving a sample image path, use Property Image Assistant Agent. This agent knows how to search for similar images and provides easy-to-use links. The output should only contain comma separated list of relative URLs of the similar images. DO NOT INCLUDE ANY ADDITIONAL INFORMATION OR EXPLANATION."
#         "When a user asks to modify a property image for a different look or replace some objects in the image with another object or modernize the look of different section of a property then use Property Image Canvas Agent. The output should only contain modified images. DO NOT INCLUDE ANY ADDITIONAL INFORMATION OR EXPLANATION."
#         "Keep the response short and to the point within in 6 to 8 sentences long that is easy for any speech assistant to respond to the user"
#     ),
#     lead_agent=BedrockLLMAgent(BedrockLLMAgentOptions(
#         name="LeadPropertySearchSupervisorAgent",
#         description="You are a supervisor agent that has team of agents capable of answering text and image based questions. You are responsible for managing the flow of the conversation. When a user asks to provide similar property images by giving a sample image path, use Property Image Assistant Agent but DO NOT INCLUDE ANY ADDITIONAL INFORMATION OR EXPLANATION. You are only allowed to manage the flow of the conversation. You are not allowed to answer questions about anything else. DO NOT suggest any follow up questions. Keep the response short, concise and within 5 sentences long",
#         model_id=os.environ.get('property_supervisor_lead_agent_llm', 'anthropic.claude-3-5-sonnet-20240620-v1:0'),
#         # model_id=os.environ.get('property_supervisor_lead_agent_llm', 'amazon.nova-pro-v1:0'),
#         custom_system_prompt={
#             'template': 'Keep the response short and to the point within in 5 to 8 sentences long that is easy for any speech assistant to respond to the user'
#         },
#         # callbacks=LLMAgentCallbacks()
#     )),
#     team=[property_search_agent, property_kb_retriever_agent, property_calculator_agent, property_appraiser_agent, property_mortgage_contract_assessor_agent, property_image_search_agent, property_image_canvas_agent],
#     trace=True,
#     storage=memory_storage
# ))

# @observe(as_type='generation', name='classify_request')
async def classify_request(_orchestrator: AgentSquad, _user_input:str, _user_id:str, _session_id:str) -> ClassifierResult:
    result:ClassifierResult = await _orchestrator.classify_request(_user_input, _user_id, _session_id)
    return result

# @observe(as_type='generation', name='agent_process_request')
async def agent_process_request(_orchestrator: AgentSquad, user_input: str,
                               user_id: str,
                               session_id: str,
                               classifier_result: ClassifierResult,
                               additional_params: dict[str, str],
                               stream_response):
    # response = await _orchestrator.agent_process_request(user_input, user_id, session_id, classifier_result, additional_params, stream_response)
    # response:AgentResponse = await _orchestrator.agent_process_request(user_input, user_id, session_id, classifier_result, additional_params, stream_response)
    # logger.info(f"response: {response}")

    # # Print metadata
    # print("\nMetadata:")
    # print(f"Selected Agent: {response.metadata.agent_name}")
    # if isinstance(response, AgentResponse) and response.streaming is False:
    #     # Handle regular response
    #     if isinstance(response.output, str):
    #         print(f"\033[34m{response.output}\033[0m")
    #     elif isinstance(response.output, ConversationMessage):
    #             print(f"\033[34m{response.output.content[0].get('text')}\033[0m")

    print(user_input)
    response = await _orchestrator.agent_process_request(user_input, user_id, session_id, classifier_result, additional_params, stream_response)
    
    # Print metadata
    print("\nMetadata:")
    print(f"Selected Agent: {response.metadata.agent_name}")
    final_response = ''

    if isinstance(response.output, ConversationMessage):
        print(response.output.content[0]['text'])
        final_response = response.output.content[0]['text']
    elif isinstance(response.output, str):
        print(response.output)
        final_response = response.output
    else:
        print(response.output)
        final_response = response.output

    return final_response


# @observe(as_type='generation', name='handle_request')
async def handle_request(_orchestrator: AgentSquad, _user_input:str, _user_id:str, _session_id:str) -> str:

    stream_response = False
    classification_result:ClassifierResult = await classify_request(_orchestrator, _user_input, _user_id, _session_id)
    print(f"classification_result: {classification_result.selected_agent.name}")

    if classification_result.selected_agent is None:
        return "No agent selected. Please try again."

    return await agent_process_request(_orchestrator, _user_input, _user_id, _session_id, classification_result,{}, stream_response)

# async def handle_request(_orchestrator: AgentSquad, _user_input:str, _user_id:str, _session_id:str):
#     classifier_result=ClassifierResult(selected_agent=supervisor, confidence=1.0)

#     response:AgentResponse = await _orchestrator.agent_process_request(_user_input, _user_id, _session_id, classifier_result, {}, True)
#     logger.info(f"response: {response}")

#     # Print metadata
#     print("\nMetadata:")
#     print(f"Selected Agent: {response.metadata.agent_name}")
#     if isinstance(response, AgentResponse) and response.streaming is False:
#         # Handle regular response
#         if isinstance(response.output, str):
#             print(f"\033[34m{response.output}\033[0m")
#         elif isinstance(response.output, ConversationMessage):
#                 print(f"\033[34m{response.output.content[0].get('text')}\033[0m")

# @observe(as_type="generation", name="nova-re-competition-001")
def run_main():
    # Initialize orchestrator with configuration options
    # orchestrator = AgentSquad(options=AgentSquadConfig(
    #     LOG_AGENT_CHAT=True,
    #     LOG_CLASSIFIER_CHAT=True,
    #     LOG_EXECUTION_TIMES=True,
    #     MAX_RETRIES=3,
    #     USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
    #     MAX_MESSAGE_PAIRS_PER_AGENT=10,
    # ))

    # classifier = BedrockClassifier(BedrockClassifierOptions(
    #     # model_id="anthropic.claude-3-haiku-20240307-v1:0",
    #     model_id="amazon.nova-pro-v1:0",
    #     callbacks=BedrockClassifierCallbacks()
    # ))
    # Initialize the orchestrator with some options
    orchestrator = AgentSquad(storage=memory_storage,options=AgentSquadConfig(
        LOG_AGENT_CHAT=True,
        LOG_CLASSIFIER_CHAT=True,
        LOG_CLASSIFIER_RAW_OUTPUT=True,
        LOG_CLASSIFIER_OUTPUT=True,
        LOG_EXECUTION_TIMES=True,
        MAX_RETRIES=3,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
        MAX_MESSAGE_PAIRS_PER_AGENT=10,
    ),
    # classifier=classifier
    )

    property_search_agent = AmazonBedrockAgent(AmazonBedrockAgentOptions(
        name='Property Search Agent',
        description='You are a Property Assistant that helps users find and analyze real estate properties. Your capabilities include retrieving property ID for a given physical address, and retrieving core details (price, size, bedrooms/bathrooms) of a property, neighborhood, climate risk factors, interior, exterior, sales history, tax records, etc.',
        agent_id='7BFZUJUBHI',
        agent_alias_id='JSLSXE2LKO',
        region='us-east-1',
        # callbacks=LLMAgentCallbacks()
    ))

    property_kb_retriever_agent = LambdaAgent(LambdaAgentOptions(
        name='Property KB Retriever Agent',
        description='You are a Property KB Retriever Agent that helps users to provide property address and id based on their natural language property search characteristics. This agent will do semantic search on property descriptions stored in the knowledge base matching user query. ',
        function_name='property_kb_retriever',
        function_region='us-east-1',
        # callbacks=LLMAgentCallbacks()
    ))

    property_calculator_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name='Property Calculator Agent',
        description='Agent specialized in providing property calculators such as buyability, affordability, and mortgage calculator based on user provided inputs such as annual income, down payment, monthly debt payments, credit score, location, loan term, interest rate, etc. Not all calculators require all inputs. The respective tools use default values for missing inputs.',
        model_id = os.environ.get('property_calculator_agent_llm', 'amazon.nova-pro-v1:0'),
        tool_config={
            'tool': property_calculator_tools,
            'toolMaxRecursions': 5,
        },
        guardrail_config={
            'guardrailIdentifier': '3pt5fpo4foxt',
            'guardrailVersion': '1'
        },
        # callbacks=LLMAgentCallbacks()
    ))

    property_appraiser_agent = LambdaAgent(LambdaAgentOptions(
        name='Property Appraiser Agent',
        description='You are a Property Appraiser Agent that helps users to analyze a home inspection report to determine the Fannie Mae property Condition (C1–C5) and Quality (Q1–Q5) ratings with explanations. It also provides a detailed repair cost breakdown for major issues and the total estimated repair cost. Keep the response short and to the point within in 6 to 8 sentences long that is easy for any speech assistant to respond to the user',
        function_name='fn_property_appraiser',
        function_region='us-east-1',
        # callbacks=LLMAgentCallbacks()
    ))

    property_mortgage_contract_assessor_agent = LambdaAgent(LambdaAgentOptions(
        name='Property Mortgage Contract Assessor Agent',
        description='You are a Property Mortgage Contract Assessor Agent that helps users to compare two mortgage contracts and provide the following: 1. Which contract is more borrower-friendly and why. 2. Highlight key differences in the following clauses: - Prepayment penalties - Escrow management - Default handling and grace periods - Dispute resolution process - Recourse and personal liability terms',
        function_name='fn_property_mortgage_contract_assessor',
        function_region='us-east-1',
        # callbacks=LLMAgentCallbacks()   
    ))

    property_image_search_agent = LambdaAgent(LambdaAgentOptions(
        name='Property Image Assistant Agent',
        description='You are a property image assistant. When someone shares an image of a property, you find other images that look similar from a large collection. The user simply needs to provide the path or location of their image, and you’ll return matching images with easy-to-use links so they can quickly explore and compare visually alike properties. The output should only contain comma separated list of relative URLs of the similar images. DO NOT INCLUDE ANY ADDITIONAL INFORMATION OR EXPLANATION.',
        function_name='fn_property_image_search',
        function_region='us-east-1',
        # callbacks=LLMAgentCallbacks()
    ))

    property_image_canvas_agent = LambdaAgent(LambdaAgentOptions(
        name='Property Image Canvas Agent',
        description='You are a property image canvas agent. When someone shares an image of a property and asks to modify the image for a different look or replace some objects in the image with another object or modernize the look of different section of a property then use this agent. The user simply needs to provide the path or location of their image, and you’ll return modified image with easy-to-use links so they can quickly explore. The output should only contain modified images. DO NOT INCLUDE ANY ADDITIONAL INFORMATION OR EXPLANATION.',
        function_name='fn_property_image_canvas',
        function_region='us-east-1',
        # callbacks=LLMAgentCallbacks()
    ))

    supervisor = SupervisorAgent(SupervisorAgentOptions(
        name="Property Search Supervisor Agent",
        description=(
            "You are a team supervisor managing a Property Search Agent, a Property KB Retriever Agent, a Property Calculator Agent, a Property Appraiser Agent, and a Property Mortgage Contract Assessor Agent. "
            "For a specific property information related queries, use Property Search Agent. "
            "For property search based on user property natural language description related queries, use Property KB Retriever Agent."
            "For property calculator related queries such as buyability, affordability, and mortgage calculator, use Property Calculator Agent."
            "For property appraiser related queries such as home inspection report analysis, use Property Appraiser Agent."
            "For property mortgage contract related queries such as comparing two mortgage contracts, use Property Mortgage Contract Assessor Agent."
            "When a user asks to provide similar property images by giving a sample image path, use Property Image Assistant Agent. This agent knows how to search for similar images and provides easy-to-use links. The output should only contain comma separated list of relative URLs of the similar images. DO NOT INCLUDE ANY ADDITIONAL INFORMATION OR EXPLANATION."
            "When a user asks to modify a property image for a different look or replace some objects in the image with another object or modernize the look of different section of a property then use Property Image Canvas Agent. The output should only contain modified images. DO NOT INCLUDE ANY ADDITIONAL INFORMATION OR EXPLANATION."
            "Keep the response short and to the point within in 6 to 8 sentences long that is easy for any speech assistant to respond to the user"
        ),
        lead_agent=BedrockLLMAgent(BedrockLLMAgentOptions(
            name="LeadPropertySearchSupervisorAgent",
            description="You are a supervisor agent that has team of agents capable of answering text and image based questions. You are responsible for managing the flow of the conversation. When a user asks to provide similar property images by giving a sample image path, use Property Image Assistant Agent but DO NOT INCLUDE ANY ADDITIONAL INFORMATION OR EXPLANATION. You are only allowed to manage the flow of the conversation. You are not allowed to answer questions about anything else. DO NOT suggest any follow up questions. Keep the response short, concise and within 5 sentences long",
            model_id=os.environ.get('property_supervisor_lead_agent_llm', 'anthropic.claude-3-5-sonnet-20240620-v1:0'),
            # model_id=os.environ.get('property_supervisor_lead_agent_llm', 'amazon.nova-lite-v1:0'),
            custom_system_prompt={
                'template': 'Keep the response short and to the point within in 5 to 8 sentences long that is easy for any speech assistant to respond to the user'
            },
            # callbacks=LLMAgentCallbacks()
        )),
        team=[property_search_agent, property_kb_retriever_agent, property_calculator_agent, property_appraiser_agent, property_mortgage_contract_assessor_agent, property_image_search_agent, property_image_canvas_agent],
        trace=True,
        storage=memory_storage
    ))

    orchestrator.add_agent(supervisor)
    # orchestrator.add_agent(property_search_supervisor_agent)
    # orchestrator.add_agent(property_search_agent)
    # orchestrator.add_agent(property_kb_retriever_agent)
    # orchestrator.add_agent(property_calculator_agent)
    # orchestrator.add_agent(property_appraiser_agent)
    # orchestrator.add_agent(property_mortgage_contract_assessor_agent)
    # orchestrator.add_agent(property_image_search_agent)
    # orchestrator.add_agent(property_image_canvas_agent)

    USER_ID = str(uuid.uuid4())
    SESSION_ID = str(uuid.uuid4())
    user_inputs = []
    final_responses = []

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            sys.exit()

        # Run async function to process user input
        if user_input:
            # user_inputs.append(user_input)

            # langfuse_context.update_current_trace(
            #     input=user_inputs,
            #     user_id=USER_ID,
            #     session_id=SESSION_ID
            # )

            response = asyncio.run(handle_request(orchestrator, user_input, USER_ID, SESSION_ID))
            # final_responses.append(response)
            # langfuse_context.update_current_trace(
            #     output=final_responses
            # )

            # langfuse.flush()

if __name__ == "__main__":
    run_main()
    