import json
import logging

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr, BaseModel, Field

from src.lib.paths import LOCAL_LLM_CONFIG_PATH

logger = logging.getLogger(__name__)

class LlmApi():

    def __init__(self, connection_profile: str):
        self.connection_profile = connection_profile

        with open(LOCAL_LLM_CONFIG_PATH, "r") as f:
            all_local_llm_configs = json.load(f)

        logger.info(f"Attempting LLM service connection via profile: {connection_profile}")

        self.connection_config = all_local_llm_configs[connection_profile]["connection_data"]
        self.config_profiles = all_local_llm_configs[connection_profile]["configuration_profiles"]
        
        self.clients = {} # Thinking and non-thinking clients will be stored here upon connection setup
        self.agents = {}  # LLM agents for different tasks will be stored here.

    def create_llm_clients(self):

        non_thinking_profile = self.config_profiles.get("non_thinking", None)
        if non_thinking_profile is None:
            logger.warning(f"'non_thinking' configuration profile not found within {self.connection_profile}. Continuing with default model config.")
            non_thinking_profile = {}
        self.clients["non_thinking"] = ChatOpenAI(
            base_url=self.connection_config["base_url"],
            api_key=SecretStr(self.connection_config["api_key"]),
            **non_thinking_profile
        )

        thinking_profile = self.config_profiles.get("thinking", None)
        if thinking_profile is None:
            logger.warning(f"'thinking' configuration profile not found within {self.connection_profile}. Continuing with default model config.")
            thinking_profile = {}
        self.clients["thinking"] = ChatOpenAI(
            base_url=self.connection_config["base_url"],
            api_key=SecretStr(self.connection_config["api_key"]),
            **thinking_profile
        )

    def invoke_client(self, client_name: str, messages, response_format = None):
        if client_name not in self.clients:
            raise ValueError(f"Client '{client_name}' not found. Available clients: {list(self.clients.keys())}")

        llm_client = self.clients[client_name]

        if response_format is not None:
            response = llm_client.invoke(messages, response_format=response_format)
            return response.additional_kwargs["parsed"]
        else:
            response = llm_client.invoke(messages)
            return response

