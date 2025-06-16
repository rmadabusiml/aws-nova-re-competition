from langchain_community.chat_models import ChatSnowflakeCortex
from utils.logging_config import setup_logging
from dotenv import load_dotenv

logger = setup_logging()
load_dotenv()

L_MAX_TOKENS = 10240
XL_MAX_TOKENS = 32000
DEFAULT_MAX_TOKENS = 8192

claude_sonnet_llm = ChatSnowflakeCortex(model="claude-3-5-sonnet", name="claude-3-5-sonnet", cortex_function="complete", temperature=0, top_p=0.95, max_tokens=DEFAULT_MAX_TOKENS)
llama_31_8b_llm = ChatSnowflakeCortex(model="llama3.1-8b",  name="llama3.1-8b", cortex_function="complete", temperature=0, top_p=0.95, max_tokens=DEFAULT_MAX_TOKENS)
llama_31_405b_llm = ChatSnowflakeCortex(model="llama3.1-405b",  name="llama3.1-405b", cortex_function="complete", temperature=0, top_p=0.95, max_tokens=DEFAULT_MAX_TOKENS)
llama_33_70b_llm = ChatSnowflakeCortex(model="llama3.3-70b",  name="llama3.3-70b", cortex_function="complete", temperature=0, top_p=0.95, max_tokens=DEFAULT_MAX_TOKENS)
mistral_large_llm = ChatSnowflakeCortex(model="mistral-large",  name="mistral-large", cortex_function="complete", temperature=0, top_p=0.95, max_tokens=DEFAULT_MAX_TOKENS)
mistral_large2_llm = ChatSnowflakeCortex(model="mistral-large2",  name="mistral-large2", cortex_function="complete", temperature=0, top_p=0.95, max_tokens=DEFAULT_MAX_TOKENS)
