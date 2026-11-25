from typing import TypedDict

from src.article_processor import ArticleProcessor
from src.lib.llm_api import LlmApi

class GraphState(TypedDict):
    "Intermediate state carried through the pipeline"
    article_filename: str
    extract_filename: str
    ontology_filename: str
    llm_connection_profile: str
    knowledge_base: str
    last_node_status: dict

class GraphContext(TypedDict):
    "Contextual information/resources available to all nodes"
    article_processor: ArticleProcessor
    llm_api: LlmApi