import os
import streamlit as st
import asyncio
import logging
import uuid
from PIL import Image
import boto3
import re

from agent_squad.agents import SupervisorAgent, SupervisorAgentOptions, AgentResponse, AgentStreamResponse
from agent_squad.utils import Logger
from agent_squad.agents import SupervisorAgent, SupervisorAgentOptions, AgentResponse, AgentStreamResponse
from agent_squad.classifiers import BedrockClassifier, BedrockClassifierOptions
from agent_squad.classifiers import ClassifierResult
from agent_squad.types import ConversationMessage
from agent_squad.storage import InMemoryChatStorage
from agent_squad.orchestrator import AgentSquad, AgentSquadConfig

from property_search_agents.search_supervisor_agent import supervisor as property_search_supervisor_agent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
BUCKET_NAME = "handsonllms-raghu"

status_placeholder = st.empty()
output_placeholder = st.empty()

memory_storage = InMemoryChatStorage()

# custom_bedrock_classifier = BedrockClassifier(BedrockClassifierOptions(
#     model_id=os.environ.get('property_search_chat_classifier_agent_llm', 'amazon.nova-pro-v1:0'),
#     region='us-east-1',
#     inference_config={
#         'maxTokens': 8192,
#         'temperature': 0.7,
#         'topP': 0.9
#     }
# ))

# orchestrator = AgentSquad(classifier=custom_bedrock_classifier, storage=memory_storage,
orchestrator = AgentSquad(storage=memory_storage,
    options=AgentSquadConfig(
        LOG_AGENT_CHAT=True,
        LOG_CLASSIFIER_CHAT=True,
        LOG_EXECUTION_TIMES=True,
        MAX_RETRIES=3,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
        MAX_MESSAGE_PAIRS_PER_AGENT=10,
    )
)

orchestrator.add_agent(property_search_supervisor_agent)

USER_ID = str(uuid.uuid4())
SESSION_ID = str(uuid.uuid4())

def display_agent():
    st.title("Property Search Agent - A Virtual assistant for Real Estate - Residential Property")
    similar_image_uploader = None
    modify_image_uploader = None

    with st.expander("🔍 Find similar or modify properties", expanded=True):
        similar_image_uploader = st.file_uploader("Choose a property image", 
                                       type=['png', 'jpg', 'jpeg'],
                                       key="similar_image_uploader")

        if similar_image_uploader is not None:
            st.session_state['uploaded_image_key'] = None
            modify_image_uploader = None
            image = Image.open(similar_image_uploader)
            st.image(image, caption='Uploaded Property Exterior', use_container_width=True)
            
            similar_image_uploader.seek(0)
            s3_key = f"real_estate/property_images/input/{similar_image_uploader.name}"
            try:
                s3.upload_fileobj(similar_image_uploader, BUCKET_NAME, s3_key)
                st.session_state['uploaded_image_key'] = s3_key
                st.success("Image uploaded successfully!")
            except Exception as e:
                st.error(f"Error uploading image: {str(e)}")

    property_question_expander = st.expander("Property Sample Questions")
     
    with property_question_expander:
        st.markdown("""
        ### Knowledge Base:
        - Looking for 3 to 5 bedroom property in Atlanta that is built in 2025
        - Looking for 4 bedroom property in Georgetown built in 2021
        - Provide neighborhood details of the first property
        - Compare interior features of first and second properties and recommend which one is better
        - Looking for 3 bedroom property in San Jose, California
        - Looking for 12000 sft commercial warehouse in Atlanta

        ### Property Calculators:
        - Calculate my house affordability with annual income is 255K, monthly debts of 500, 75K as down payment for 30 years
        - What is my monthly morgage payment for a home price 600K with 100K down payment and 5% interest rate for 30 years
        - Estimate how much house I can buy with my annual income of 255K with monthly debts of 500 and will do 75K as down payment for 30 years

        ### Property Appraiser:
        - Provide summary of inspection report for property id 174587298

        ### Property Mortgage Contract Assessor:
        - Compare the two mortgage contracts for property id 174587298

        ### Property Image Search:
        - Find similar properties as the uploaded image

        ### Property Image Canvas:
        - Replace left side grass area with a kids friendly mini golf course
        - Make kitchen cabinets olive green
        - Make the garage white and the house brown
        """)

    user_input = st.text_area("You:", key="input", placeholder="Ask me a question... ")
    if 'uploaded_image_key' in st.session_state:
        user_input += f" [IMAGE_REF:{st.session_state['uploaded_image_key']}]"
    
    col1, col2 = st.columns(2)
    
    status_container = st.container()
    output_container = st.container()

    if 'processing' not in st.session_state:
        st.session_state.processing = False

    if col1.button("Submit", disabled=st.session_state.processing) or (user_input and user_input[-1] == "\n"):
        st.session_state.processing = True
        
        status_container.empty()
        output_container.empty()
        
        with status_container:
            status_placeholder = st.empty()
            
        with output_container:
            output_placeholder = st.empty()

        print(f"User ID: {USER_ID}")
        print(f"Session ID: {SESSION_ID}")

        asyncio.run(run_query(orchestrator, user_input.strip(), USER_ID, SESSION_ID, status_placeholder, output_placeholder))
        st.session_state.processing = False

        if 'conversations' in st.session_state:
            for question, final_response in reversed(st.session_state["conversations"]):
                st.info(f"\nQuestion:\n + {question}")
                st.success(f"\nFinal Response:\n +  {final_response}")

async def run_query(_orchestrator: AgentSquad, _user_input:str, _user_id:str, _session_id:str, status_placeholder, output_placeholder):
    try:
        response:AgentResponse = await _orchestrator.route_request(
            _user_input,
            _user_id,
            _session_id
        )

        logger.info(f"response: {response}")
        final_response = None

        print("\nMetadata:")
        print(f"Selected Agent: {response.metadata.agent_name}")

        if isinstance(response, AgentResponse):
            if isinstance(response.output, str):
                print(f"\033[34m{response.output}\033[0m")
                final_response = response.output
            elif isinstance(response.output, ConversationMessage):
                    print(f"\033[34m{response.output.content[0].get('text')}\033[0m")
                    final_response = response.output.content[0].get('text')

        print(f"Response in UI: {final_response}")

        assistant_message = st.chat_message("assistant")

        if 'uploaded_image_key' in st.session_state:
            await display_similar_properties(final_response)
        else:
            cleaned_output = re.sub(r'\$(\d)', r'$ \1', final_response)

            if cleaned_output != final_response:
                assistant_message.text(cleaned_output)
            else:
                assistant_message.markdown(cleaned_output)

        status_placeholder.markdown("")
        output_placeholder.markdown("")

        st.session_state.setdefault("conversations", []).append((_user_input, final_response))

    except Exception as e:
        st.error(f"Error occurred: {e}")

async def display_similar_properties(response_text):
    """Parse and display similar property images"""
    pattern = r'(real_estate/property_images/exterior_\d+\.(?:png|jpg|jpeg)|real_estate/property_images/output/[\w-]+\.(?:png|jpg|jpeg))'

    image_keys = re.findall(pattern, response_text)
    
    if image_keys:
        st.subheader("Similar Properties")
        cols = st.columns(2)
        
        for idx, key in enumerate(image_keys):
            try:
                url = s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': BUCKET_NAME, 'Key': key},
                    ExpiresIn=3600
                )
                
                with cols[idx % 2]:
                    st.image(url, use_container_width=True)
                    st.caption(os.path.basename(key))
                    
            except Exception as e:
                st.error(f"Error loading image {key}: {str(e)}")
