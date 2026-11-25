import logging

from src.graph_state import GraphState

logger = logging.getLogger(__name__)

def should_rerun_deduplicate_entities_from_extract_node(state: GraphState) -> str:
    last_node_status = state["last_node_status"]
    print(f"Last node status: {last_node_status}")
    if "last_node" in last_node_status and last_node_status["last_node"] == "deduplicate_entities_from_extract_node":
        if last_node_status["should_rerun"]:
            return "rerun"

    else:
        logger.warning("State field 'last_node_status' was neither empty nor containing the right information. Continuing to next node.")

    return "continue"

def should_rerun_ensure_findable_representations_on_extract_node(state: GraphState) -> str:
    last_node_status = state["last_node_status"]
    print(f"Last node status: {last_node_status}")
    if "last_node" in last_node_status and last_node_status["last_node"] == "ensure_findable_representations_on_extract_node":
        if last_node_status["should_rerun"]:
            return "rerun"

    else:
        logger.warning("State field 'last_node_status' was neither empty nor containing the right information. Continuing to next node.")

    return "continue"