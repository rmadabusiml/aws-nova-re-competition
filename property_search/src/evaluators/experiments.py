from datetime import datetime
import time
import re
from langchain_core.prompts import PromptTemplate
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from langchain_community.chat_models import ChatSnowflakeCortex
from llm_eval import nova_micro_llm, nova_lite_llm, nova_pro_llm
from score_eval import score_bleu, score_rouge
import nltk

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
langfuse = Langfuse()
# nltk.download('punkt_tab') # this needs to be run first time to download tokenizers for scoring

@observe()
def extract_property_id(prompt_input, model):
  prompt_template = """{prompt_input}"""

  # print(prompt_input)

  PROMPT = PromptTemplate(template=prompt_template, input_variables=["prompt_input"])
  chain = PROMPT | model
  result = chain.invoke(input=prompt_input)
  # print(result)
  # print(result.content)
  original_question = result.content
  return original_question

@observe()
def extract_imagegen_task_type(prompt_input, model):
  prompt_template = """{prompt_input}"""
  # print(prompt_input)

  PROMPT = PromptTemplate(template=prompt_template, input_variables=["prompt_input"])
  chain = PROMPT | model
  result = chain.invoke(input=prompt_input)
  # print(result)
  # print(result.content)
  original_question = result.content
  return original_question

@observe()
def run_experiment(experiment_name=None, model_id=None, model=None, dataset=None):
  # print(dataset)
  dataset = langfuse.get_dataset(dataset)

  for item in dataset.items:
    print(item)
    generationStartTime = datetime.now()
    print(item.input)
    llm_input = item.input

    expected_output = item.expected_output
    print(expected_output)
    start_time = time.time()
    generationStartTime = datetime.fromtimestamp(start_time)
    llm_output = extract_property_id(llm_input, model)
    end_time = time.time()
    generationEndTime = datetime.fromtimestamp(end_time)
    print(llm_output)

    langfuse_generation = langfuse.generation(
      name=item.id,
      input=item.input,
      output=llm_output,
      model=model_id,
      start_time=generationStartTime,
      end_time=generationEndTime
    )
    
    langfuse_context.flush()
    time.sleep(5)

    rouge_score = score_rouge("rougeL", llm_output, expected_output)
    # bleu_score = score_bleu("bleu2", llm_output, expected_output)
    time.sleep(5)
    item.link(langfuse_generation, experiment_name)
    langfuse_context.flush()
    time.sleep(5)

    langfuse_generation.score(
      name="rouge-L",
      value=rouge_score
    )
    time.sleep(5)
    

eval_llms_dict = {
    "amazon.nova-micro-v1:0": nova_micro_llm,
    "amazon.nova-lite-v1:0": nova_lite_llm,
    "amazon.nova-pro-v1:0": nova_pro_llm,
}

dataset_name = "property-id-extraction-dataset"

# Loop through the dictionary and use model_id and model_name
for model_id, model in eval_llms_dict.items():
    experiment_name = f"{model_id}-{datetime.now()}"
    run_experiment(experiment_name=experiment_name, model_id=model_id, model=model, dataset=dataset_name)