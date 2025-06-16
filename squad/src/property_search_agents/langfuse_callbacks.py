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

load_dotenv()  # take environment variables

langfuse = Langfuse()

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