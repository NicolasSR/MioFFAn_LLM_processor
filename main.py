

import logging
from pathlib import Path
import json
import argparse

from src.graph import GraphState, GraphContext, build_and_run_graph
from src.article_processor import ArticleProcessor

from src.lib.llm_api import LlmApi
from src.lib.paths import PROJECT_ROOT_DIR, DEFAULT_RUN_CONFIG_PATH, LANGGRAPH_CHECKPOINTS_DIR
from src.lib.cli_listing_utils import (
    resolve_article_filename,
    resolve_extract_filename,
    resolve_ontology_filename,
    resolve_llm_connection_profile
)
from src.lib.knowledge_base_api import (
    initialize_knowledge_base,
    add_article_to_kb,
    add_math_to_kb,
    add_chunks_to_kb,
    add_extract_to_kb,
    deduce_knowledge,
    serialize_kb
)
from src.article_processor import CHUNK_STRATEGIES


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def process_paper(article_path: Path,
                  extract_path: Path,
                  ontology_path: Path,
                  llm_connection_profile: str,
                  run_name: str,
                  chunk_strategy: dict,
                  restart_from: str | None):
    """Run the langgraph pipeline on an article (HTML file) and taking as reference a given extract (JSON file).
    The state knowledge base will follow the provided ontology (TTL file).
    The LLM connection profile is used to state which local llm provider to connect to.
    If *restart_from* is provided, the pipeline will restart from the specified checkpoint.
    """

    article_processor = ArticleProcessor(article_path)
    article_processor.chunk_article(**chunk_strategy)
    article_processor.write_chunks_to_file("chunks_test.txt")  # This is just for visualization and debugging

    article_processor.create_extract(extract_path)
    article_processor.write_extract_to_file("extract_test.txt")  # This is just for visualization and debugging

    knowledge_base = initialize_knowledge_base(ontology_path)
    article_id = article_path.stem
    add_article_to_kb(article_id, knowledge_base)
    display_math_ids = article_processor.get_display_math_ids()
    inline_math_ids = article_processor.get_inline_math_ids()
    add_math_to_kb(display_math_ids, inline_math_ids, knowledge_base)
    chunk_ids_and_content = article_processor.get_chunks_ids_and_content()
    add_chunks_to_kb(chunk_ids_and_content, article_id, knowledge_base)
    extract_info = article_processor.get_extract_id_chunks_and_content()
    add_extract_to_kb(extract_info, knowledge_base)
    serialized_kb = serialize_kb(knowledge_base)

    initial_state = GraphState(
        article_filename = article_path.stem,
        extract_filename = extract_path.stem,
        ontology_filename = ontology_path.stem,
        llm_connection_profile = llm_connection_profile,
        knowledge_base = serialized_kb,
        last_node_status = ""
    )

    llm_api = LlmApi(llm_connection_profile)
    llm_api.create_llm_clients()

    graph_context = GraphContext(
        article_processor = article_processor,
        llm_api=llm_api
    )

    graph_config = {
        "recursion_limit": 100,
        "configurable": {
            "thread_id": run_name,
        }
    }
    
    final_state = build_and_run_graph(initial_state, graph_config, graph_context, LANGGRAPH_CHECKPOINTS_DIR / "pipeline_checkpoints.db", restart_from)

    # Save the final knowledge base to a TTL file for inspection
    with open("out_knowledge_base.ttl", "w", encoding="utf-8") as f:
        f.write(final_state["knowledge_base"])

    return final_state

if __name__ == "__main__":

    with open(DEFAULT_RUN_CONFIG_PATH, "r") as f:
        default_run_config = json.load(f)

    parser = argparse.ArgumentParser(description="Chunk a scientific paper into equation-centered segments.")
    parser.add_argument("--article",default=default_run_config.get("default_article_filename"), help="Filename of the article HTML file to process (without .html extension).")
    parser.add_argument("--extract", default=default_run_config.get("default_extract_filename"), help="Filename of the extract JSON file to process (without .json extension).")
    parser.add_argument("--ontology", default=default_run_config.get("default_ontology_filename"), help="Filename of the ontology TTL file to use (without .ttl extension).")
    parser.add_argument("--llm_profile", default=default_run_config.get("default_llm_connection"), help="LLM connection profile name to use amongst those available in the configuration file. Default is the first one.")  # TODO: Manage selection through argparse's own choices field
    parser.add_argument("--run_name", default=default_run_config.get("default_run_name"), help="Name of the run. Default is 'default_run'. Will be used to create a langgraph checkpoints thread.")
    parser.add_argument("--restart_from", default=None, help="Specify the checkpoint ID to take as starting point. You can get the checkpoint ids by using the inspect_checkpoints python tool.")
    
    args = parser.parse_args()

    article_path = resolve_article_filename(args.article, parser)
    extract_path = resolve_extract_filename(args.extract, parser)
    ontology_path = resolve_ontology_filename(args.ontology, parser)
    llm_connection_profile = resolve_llm_connection_profile(args.llm_profile, parser)

    default_chunk_strategy = default_run_config["default_chunk_strategy"]
    if default_chunk_strategy["name"] not in CHUNK_STRATEGIES:
        raise ValueError(f"Default chunk strategy {default_chunk_strategy["name"]!r} is not valid. Expected one of: {', '.join(CHUNK_STRATEGIES)!r}.")


    final_state = process_paper(
        article_path,
        extract_path,
        ontology_path,
        llm_connection_profile,
        run_name=args.run_name,
        chunk_strategy=default_chunk_strategy,
        restart_from=args.restart_from
    )