from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.graph_state import GraphState, GraphContext

from src.nodes.entity_extraction_nodes import (
    ner_on_extract_node,
    deduplicate_entities_from_extract_node,
    get_missing_slts_node,
    ensure_findable_representations_on_extract_node,
    disambiguate_entities_against_expressions_from_extract_node
)
from src.nodes.conditional_edges import (
    should_rerun_deduplicate_entities_from_extract_node,
    should_rerun_ensure_findable_representations_on_extract_node
)

logger = logging.getLogger(__name__)

def _build_graph() -> StateGraph:
    """Build the LangGraph workflow."""

    graph = StateGraph(GraphState, context_schema=GraphContext)

    graph.add_node("ner_on_extract_node", ner_on_extract_node)
    graph.add_node("deduplicate_entities_from_extract_node", deduplicate_entities_from_extract_node)
    graph.add_node("get_missing_slts_node", get_missing_slts_node)
    graph.add_node("ensure_findable_representations_on_extract_node", ensure_findable_representations_on_extract_node)
    graph.add_node("disambiguate_entities_against_expressions_from_extract_node", disambiguate_entities_against_expressions_from_extract_node)
    # graph.add_node("paper_wide_exploration_of_entities", paper_wide_exploration_of_entities_node)
    
    graph.add_edge(START, "ner_on_extract_node")
    graph.add_edge("ner_on_extract_node", "deduplicate_entities_from_extract_node")
    graph.add_conditional_edges(
        "deduplicate_entities_from_extract_node",
        should_rerun_deduplicate_entities_from_extract_node,
        {
            "rerun": "deduplicate_entities_from_extract_node",
            "continue": "get_missing_slts_node"
        }
    )
    graph.add_edge("get_missing_slts_node", "ensure_findable_representations_on_extract_node")
    graph.add_conditional_edges(
            "ensure_findable_representations_on_extract_node",
            should_rerun_ensure_findable_representations_on_extract_node,
            {
                "rerun": "get_missing_slts_node",
                "continue": "disambiguate_entities_against_expressions_from_extract_node"
            }
        )
    graph.add_edge("disambiguate_entities_against_expressions_from_extract_node", END)
    # graph.add_edge("parse_html", "chunk_by_section")
    # graph.add_edge("chunk_by_section", "symbol_extraction")
    # graph.add_edge("symbol_extraction", "entity_symbol_corrector")
    # graph.add_edge("entity_symbol_corrector", "paper_wide_exploration_of_entities")
    # graph.add_edge("paper_wide_exploration_of_entities", END)

    return graph


def build_and_run_graph(initial_state, config, context, checkpoint_path, restart_from):
    graph = _build_graph()
    with SqliteSaver.from_conn_string(checkpoint_path) as memory:
        """Compile and return the runnable graph."""
        compiled_graph = graph.compile(checkpointer=memory)

        if restart_from is not None:
            checkpoints = list(compiled_graph.get_state_history(config))
            if restart_from == "print_results":
                last_checkpoint = checkpoints[0]
                return last_checkpoint.values
            restart_checkpoint_id = restart_from
            # matched_ids = [snapshot.config["configurable"]["checkpoint_id"] for snapshot in checkpoints if restart_from in snapshot.next]
            # if not matched_ids:
            #     logger.error(f"No checkpoint with starting node '{restart_from}' is available. Aborting.")
            #     raise
            # restart_checkpoint_id = matched_ids[0]
            config["configurable"]["checkpoint_id"] = restart_checkpoint_id
            initial_state = None

        return compiled_graph.invoke(initial_state,
            config=config,
            context=context,
            durability="sync")