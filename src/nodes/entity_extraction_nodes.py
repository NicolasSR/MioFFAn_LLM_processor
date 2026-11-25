import logging
import re

from langgraph.runtime import Runtime
# from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

from src.graph_state import GraphState, GraphContext
import src.lib.knowledge_base_api as kb_api
from src.lib.generic_utils import get_hasse_diagram_for_lists_content
from src.lib.pydantic_models import (
    Mathematical_entites_list,
    Written_representation
)
from src.lib.equation_utils import get_slt_from_latex, get_slt_from_mathml

logger = logging.getLogger(__name__)

def ner_on_extract_node(state: GraphState, runtime: Runtime[GraphContext]) -> GraphState:
    """Perform Named Entity Recognition on the given extract."""

    logger.info("ENTERED 'NER on extract' node")

    llm_api = runtime.context["llm_api"]
    article_processor = runtime.context["article_processor"]

    extract = article_processor.get_extract_text()

    system_query=f"""
You are a research assistant providing assistance to understanding scientific papers. Specifically, understanding mathematical formulations within them.
You are given some paragraphs from a scientific paper, with some mathematical formulation.
Your task is to perform named entity recognition on the different individual mathematical entities present within the given text.
For each entity, collect its written representations (whether in plain text or LaTeX of both) and its properties, specifying if the property is explicitly stated within the text.
Be very careful about marking properties as confirmed. Only do it if the information is explicitly stated or can be directly inferred from explicitly stated facts.
If the entity is a common operator using standard notation, mark it as a standard operator.
"""
    
    user_query=f"""
Equation and surrounding text: {extract}
"""

    messages = [
        ("system", system_query),
        ("user", user_query)
    ]

    response = llm_api.invoke_client("non_thinking", messages, response_format=Mathematical_entites_list)

    kb = kb_api.parse_kb_from_string(state["knowledge_base"])

    extract_id = article_processor.get_extract_id()
    extract_uri = kb_api.get_object_uri_by_index_id(kb, extract_id, "Extract")

    kb_api.add_entities_from_mathematical_entities_list(kb, response.mathematical_entities_list, extract_uri)

    serialized_kb = kb.serialize(format="ttl")
    state["knowledge_base"] = serialized_kb
    state["last_node_status"] = {}
    
    return state

def deduplicate_entities_from_extract_node(state: GraphState, runtime: Runtime[GraphContext]) -> GraphState:
    """Remove duplicates from the entities extracted from the extract."""
    
    logger.info("ENTERED 'deduplicate_entities_from_extract_node' node")

    last_node_status = state["last_node_status"]
    if "last_node" in last_node_status and last_node_status["last_node"] == "deduplicate_entities_from_extract_node":
        rerun_count = last_node_status.get("rerun_count", 0)
    else:
        rerun_count = 0

    llm_api = runtime.context["llm_api"]
    article_processor = runtime.context["article_processor"]

    kb = kb_api.parse_kb_from_string(state["knowledge_base"])

    repeated_entities = kb_api.get_entities_with_same_representation(kb)

    extract = article_processor.get_extract_text()

    rerun_count += 1
    state["last_node_status"] = {
        "last_node": "deduplicate_entities_from_extract_node",
        "rerun_count": rerun_count,
        "should_rerun": False
    }

    for repr_uri, entities_uri_list in repeated_entities.items():

        logger.info(f"Comparing entities with common representation:  {kb_api.get_representation_str_and_type(kb, repr_uri)}.")
        
        extra_reprs_dict = {}
        for entity_uri in entities_uri_list:
            extra_reprs = kb_api.get_representations_of_object(kb, entity_uri)
            extra_reprs.remove(repr_uri)
            extra_reprs_dict[entity_uri] = extra_reprs
            
        zero_prop_entities = []
        nonzero_prop_entities = []
        entity_property_enunciates_dict = {}
        for entity_uri in entities_uri_list:
            properties_dict = kb_api.get_properties_of_object(kb, entity_uri)
            if properties_dict:
                nonzero_prop_entities.append(entity_uri)
                entity_property_enunciates_dict[entity_uri] = [prop_enunciate for prop_enunciate in properties_dict.values()]
                logger.info(f" - {', '.join(extra_reprs_dict[entity_uri])}. | Has properties")
            else:
                zero_prop_entities.append(entity_uri)
                logger.info(f" - {', '.join(extra_reprs_dict[entity_uri])}.")

        if not zero_prop_entities:
            # We have no way of deterministically comparing properties. So if all conflicting entities have properties assigned, we will assume that they are independent entities.
            logger.info("All conflicting entities have properties, therefore they are maintained.")
            continue

        conflicting_entities_dict = {}
        for entity_uri in zero_prop_entities:
            conflicting_entities_dict[entity_uri] = extra_reprs_dict[entity_uri]
        for i, entity_uri in enumerate(nonzero_prop_entities):
            # As we cannot deterministically compare whether properties are equal, we will mark every entity with properties as relevant by adding a unique item in its dict list.
            conflicting_entities_dict[entity_uri] = extra_reprs_dict[entity_uri] + [i]

        # We use a hesse diagram with the contents within conflicting_entities_dict to get the elements that are relevant to add in our prompt
        extra_reprs_hasse_diagram = get_hasse_diagram_for_lists_content(conflicting_entities_dict)
        leaf_entities = [node for node in conflicting_entities_dict.keys() if not extra_reprs_hasse_diagram["children_dict"].get(node)]

        # If there is only one leaf, it means that all proposed entities have the same representations (or subsets of other ones') and at most one of them has properties.
        # So we just conserve the leaf (the one with most representations and possibly properties) and remove the rest.
        if len(leaf_entities) == 1:
            logger.info("Keeping only the most informative entity.")
            for old_entity_uri in conflicting_entities_dict.keys():
                if old_entity_uri != leaf_entities[0]:
                    kb_api.remove_object_from_kb(old_entity_uri, kb)

        # Check whether any leaf entity has zero properties. If so, resolve conflict. Otherwise, just remove all the entities without properties.
        elif any([leaf in zero_prop_entities for leaf in leaf_entities]):
            logger.info("Solving conflict via LLM.")
            common_repr_str, common_repr_type = kb_api.get_representation_str_and_type(kb, repr_uri)
            common_repr_str = f"{common_repr_str} ({common_repr_type})"
            reprs_and_props_lists_for_query = ""
            for leaf_entity_uri in leaf_entities:
                reprs_and_props_lists_for_query += f"\n    - Representations: {common_repr_str}"
                for repr_uri in extra_reprs_dict[leaf_entity_uri]:
                    repr_str, repr_type = kb_api.get_representation_str_and_type(kb, repr_uri)
                    reprs_and_props_lists_for_query += f", {repr_str} ({repr_type})"
                reprs_and_props_lists_for_query += f"\n      Properties:"
                if leaf_entity_uri in entity_property_enunciates_dict:
                    for prop_enunciate in entity_property_enunciates_dict[leaf_entity_uri]:
                        reprs_and_props_lists_for_query += f"\n        + {prop_enunciate}"
                else:
                    reprs_and_props_lists_for_query += " None"

            messages = [
                ("system", system_query),
                ("user", user_query)
            ]

            system_query=f"""
            You are a research assistant providing assistance to understanding scientific papers. Specifically, understanding mathematical formulations within them.
            You are given some paragraphs from a scientific paper, with some mathematical formulation.
            Some mathematical entities have been extracted from the given text. Multiple entities have been proposed with or without properties and with at least one common written representation.
            Your task is to study this list of conflicting entities and create a new list that eliminates duplicates.
            If multiple entities with a representation in common should really exist, differentiate them via their properties.
            """
                                
            user_query=f"""
            Equation and surrounding text: {extract}
            
            Conflicting entities:{reprs_and_props_lists_for_query}
            
            Resolved entity or entities:
            """

            print(user_query)

            response = llm_api.invoke_client("non_thinking", messages, response_format=Mathematical_entites_list)

            print(response)
            
            for old_entity_uri in conflicting_entities_dict.keys():
                kb_api.remove_object_from_kb(old_entity_uri, kb)

            logger.info("Removed all old entities.")

            extract_id = article_processor.get_extract_id()
            extract_uri = kb_api.get_object_uri_by_index_id(kb, extract_id, "Extract")

            kb_api.add_entities_from_mathematical_entities_list(kb, response.mathematical_entities_list, extract_uri)

        else:
            logger.info("Removed entities without properties.")
            for zero_prop_entity_uri in zero_prop_entities:
                kb_api.remove_object_from_kb(zero_prop_entity_uri, kb)

        should_rerun = rerun_count <= 10*len(repeated_entities.keys())
        state["last_node_status"]["should_rerun"] = should_rerun

        if should_rerun:
            logger.info(f"Node needs re-running.")
        break

    kb_api.remove_orphan_representations_and_properties(kb)

    serialized_kb = kb.serialize(format="ttl")
    state["knowledge_base"] = serialized_kb
    
    return state

def get_missing_slts_node(state: GraphState, runtime: Runtime[GraphContext]) -> GraphState:
    """Assigns SLTs to Mathematical_representation instances missing this field."""
    
    logger.info("ENTERED 'get_missing_slts' node")

    kb = kb_api.parse_kb_from_string(state["knowledge_base"])
    
    repr_dict = kb_api.get_mathematical_representations_without_slt(kb)

    for repr_uri, repr_info in repr_dict.items():
        if repr_info["type"] == "latex":
            slt_str = get_slt_from_latex(repr_info["str"])
        elif repr_info["type"] == "mathml":
            slt_str = get_slt_from_mathml(repr_info["str"])
        else:
            logger.warning(f"Type unknown for representation {repr_info}. Could not create SLT")
            continue
        kb_api.add_data_property_to_kb(repr_uri, "representation_slt", slt_str, kb)
        logger.info(f"Created SLT '{slt_str}' for representation '{repr_info['str']}'")

    serialized_kb = kb.serialize(format="ttl")
    state["knowledge_base"] = serialized_kb
    
    return state

def ensure_findable_representations_on_extract_node(state: GraphState, runtime: Runtime[GraphContext]) -> GraphState:
    """Ensures that proposed written representations are findable within the extract."""

    last_node_status = state["last_node_status"]
    if "last_node" in last_node_status and last_node_status["last_node"] == "ensure_findable_representations_on_extract_node":
        rerun_count = last_node_status.get("rerun_count", 0)
    else:
        rerun_count = 0

    logger.info("ENTERED 'ensure findable representations on extract' node")

    llm_api = runtime.context["llm_api"]
    article_processor = runtime.context["article_processor"]

    kb = kb_api.parse_kb_from_string(state["knowledge_base"])

    extract_id = article_processor.get_extract_id()
    extract_uri = kb_api.get_object_uri_by_index_id(kb, extract_id, "Extract")
    total_unfound_reprs_dict = kb_api.get_unfound_entity_representations(kb, subdocument_part_uri=extract_uri)

    rerun_count += 1
    state["last_node_status"] = {
        "last_node": "ensure_findable_representations_on_extract_node",
        "rerun_count": rerun_count,
        "should_rerun": False
    }

    if not total_unfound_reprs_dict:
        # All symbols are already findable in the extract, 
        return state

    
    repr_suggestions_dict = {}
    for entity_uri, unfound_repr_uris_for_entity_dict in total_unfound_reprs_dict.items():
        entity_repr_matches_list = []

        mathematical_reprs_list = unfound_repr_uris_for_entity_dict["Mathematical_representation"]
        repr_slts_list = [str(kb_api.get_data_property_of_object(kb, repr_uri, "representation_slt")[0]) for repr_uri in mathematical_reprs_list]
        entity_math_repr_matches_list, entity_repr_suggestions = article_processor.find_slts_in_extract(repr_slts_list)
        entity_repr_matches_list.extend(entity_math_repr_matches_list)
        repr_suggestions_dict = {**repr_suggestions_dict, **entity_repr_suggestions}

        text_reprs_list = unfound_repr_uris_for_entity_dict["Text_representation"]
        repr_strs_list = [kb_api.get_data_property_of_object(kb, repr_uri, "representation_str")[0] for repr_uri in text_reprs_list]
        entity_text_repr_matches_list, entity_repr_suggestions = article_processor.find_plain_text_in_extract(repr_strs_list)
        entity_repr_matches_list.extend(entity_text_repr_matches_list)
        repr_suggestions_dict = {**repr_suggestions_dict, **entity_repr_suggestions}

        written_reprs_list = unfound_repr_uris_for_entity_dict["Mathematical_representation"]
        for written_repr_uri in written_reprs_list:
            written_repr_str = kb_api.get_data_property_of_object(kb, written_repr_uri, "representation_str")[0]
            written_repr_type = kb_api.get_data_property_of_object(kb, written_repr_uri, "representation_type")[0]
            entity_repr_suggestions = {written_repr_uri: ["Invalid representation type. Try 'text', 'latex' or 'mathml'."]}
            repr_suggestions_dict = {**repr_suggestions_dict, **entity_repr_suggestions}
            logger.warining(f"Pure Written_representation found: '{written_repr_str}' ({written_repr_type}). Should be categorized in a representation subclass.")
        
        for repr_matches_dict in entity_repr_matches_list:
            repr_uri = repr_matches_dict["uri"]
            doc_part_uris = []
            for id in repr_matches_dict["extract"]:
                doc_part_uris.append(kb_api.get_object_uri_by_index_id(kb, id, "Extract"))
            for id in repr_matches_dict["chunk"]:
                doc_part_uris.append(kb_api.get_object_uri_by_index_id(kb, id, "Chunk"))
            for id in repr_matches_dict["display_math"]:
                doc_part_uris.append(kb_api.get_object_uri_by_index_id(kb, id, "Display_math"))
            for id in repr_matches_dict["inline_math"]:
                doc_part_uris.append(kb_api.get_object_uri_by_index_id(kb, id, "Inline_math"))
            for doc_part_uri in doc_part_uris:
                kb_api.add_object_property_to_kb(repr_uri, "found_in", doc_part_uri, kb)

    # Check again for unfound representations. We will try to replace them for representations matching the text
    total_unfound_reprs_dict = kb_api.get_unfound_entity_representations(kb, subdocument_part_uri=extract_uri)

    if total_unfound_reprs_dict:

        state["last_node_status"]["should_rerun"] = rerun_count <= 10

        total_found_reprs_dict = kb_api.get_found_entity_representations(kb, subdocument_part_uri=extract_uri)

        system_query=f"""
    You are a research assistant providing assistance to understanding scientific papers. Specifically, understanding mathematical formulations within them.
    You are given some paragraphs from a scientific paper, with some mathematical formulation.
    You are also given information about a mathematical entity extracted from this text.
    The information includes written representations (and whether they are 'text', 'latex' or 'mathml') that represent it and were already confirmed, and also includes different mathematical properties related to it.
    Finally, you are given, one by one, other representations that were proposed but not found. They may come with some suggestion on how to change them.
    For each one determine if the representation is relevant and, if so, propose a fixed string for it that is found in the text.
    """
        base_user_query=f"""
    Paragraphs: {article_processor.get_extract_text()}
    """

        for entity_uri, unfound_repr_uris_for_entity_dict in total_unfound_reprs_dict.items():
            unfound_repr_uris_for_entity_list = [item for sublist in unfound_repr_uris_for_entity_dict.values() for item in sublist]
            unfound_reprs_str = ""
            for i, repr_uri in enumerate(unfound_repr_uris_for_entity_list):
                repr_str = kb_api.get_data_property_of_object(kb, repr_uri, "representation_str")[0]
                repr_type = kb_api.get_data_property_of_object(kb, repr_uri, "representation_type")[0]
                unfound_reprs_str += f"        {i}. {repr_str} ({repr_type})."
                if repr_uri in repr_suggestions_dict :
                    suggestions = repr_suggestions_dict[repr_uri]
                    if suggestions:
                        unfound_reprs_str += " Suggestions: " + ". ".join(suggestions)
                        unfound_reprs_str += "\n"

            found_repr_uris_for_entity_dict = total_found_reprs_dict.get(entity_uri,[])
            found_repr_uris_for_entity_list = [item for sublist in found_repr_uris_for_entity_dict.values() for item in sublist]
            found_reprs_str = ', '.join([kb_api.get_data_property_of_object(kb, repr_uri, "representation_str")[0]+' ('+kb_api.get_data_property_of_object(kb, repr_uri, "representation_type")[0]+')' for repr_uri in found_repr_uris_for_entity_list])

            entity_property_enunciates = kb_api.get_property_enunciates_of_math_entity(kb, entity_uri)
            properties_str = "\n            + ".join(entity_property_enunciates)

            user_query = base_user_query + f"""
    Mathematical entity information:
        - Found representations: {found_reprs_str}
        - Properties:
            + {properties_str}
            
    Unfound representations:
{unfound_reprs_str}

    Resolution of each representation:
    """
            messages = [
                ("system", system_query),
                ("user", user_query)
            ]

            class Representation_resolution(BaseModel):
                """Resolution of the written representation study, either proposing a replacement for the representation or marking it as irrelevant."""
                id: int = Field(description="Integer ID for the representation being resolved (as defined in the prompt).")
                written_representation: Written_representation | None = Field("If the representation is relevant, correction for the originally proposed one. Otherwise, None.")

            class Representation_resolutions_list(BaseModel):
                """List of resolutions of the written representations study, either proposing replacement for the representations or marking them as irrelevant."""
                representation_resolutions_list: list[Representation_resolution] = Field(description="List of resolutions of the written representations study, either proposing replacement for the representations or marking them as irrelevant.")

            response = llm_api.invoke_client("non_thinking", messages, response_format=Representation_resolutions_list)

            for repr_resolution in response.representation_resolutions_list:
                orig_repr = unfound_repr_uris_for_entity_list[repr_resolution.id]
                kb_api.remove_object_from_kb(orig_repr["uri"], kb)
                if repr_resolution.written_representation is not None:
                    new_repr = repr_resolution.written_representation
                    kb_api.add_representation_of_mathematical_entity(kb, new_repr, entity_uri)

    
    serialized_kb = kb.serialize(format="ttl")
    state["knowledge_base"] = serialized_kb
    return state


def disambiguate_entities_against_expressions_from_extract_node(state: GraphState, runtime: Runtime[GraphContext]) -> GraphState:
    """Determine whether representations containing other representations should represent expressions or entities, or the entity should be removed"""

    logger.info("ENTERED 'disambiguate_entities_against_expressions_from_extract_node' node")

    llm_api = runtime.context["llm_api"]
    article_processor = runtime.context["article_processor"]

    kb = kb_api.parse_kb_from_string(state["knowledge_base"])





    state["last_node_status"] = {}

    return state

