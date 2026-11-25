from pathlib import Path
import uuid
import logging
import hashlib

from pydantic import BaseModel
import rdflib
from owlrl import DeductiveClosure, OWLRL_Semantics

from src.lib.pydantic_models import (
    Mathematical_entites_list,
    Written_representation,
    Text_representation,
    Mathematical_representation
)

logger = logging.getLogger(__name__)

BASE = rdflib.Namespace("http://www.semanticweb.org/nsibuet/ontologies/2026/9/mioffan_llm_processor#")
OWL_CLASS = rdflib.Namespace(BASE["OWLClass_"])
OWL_OBJECT_PROP = rdflib.Namespace(BASE["OWLObjectProperty_"])
OWL_DATA_PROP = rdflib.Namespace(BASE["OWLDataProperty_"])
OBJECT = rdflib.Namespace(BASE["Object_"])
RDF_TYPE = rdflib.RDF.type

def initialize_knowledge_base(ontology_path: Path) -> rdflib.Graph:
    """Initialize the knowledge base using the provided ontology (TTL file)."""
    kb = rdflib.Graph()
    kb.parse(ontology_path, format="ttl")
    return kb

def parse_kb_from_string(kb_string: str) -> rdflib.Graph:
    """Parse the knowledge base from a string representation (TTL format)."""
    kb = rdflib.Graph()
    kb.parse(data=kb_string, format="ttl")
    return kb

def serialize_kb(kb: rdflib.Graph) -> str:
    return kb.serialize(format="ttl")

def deduce_knowledge(kb: rdflib.Graph):
    """Apply OWL RL reasoning to the knowledge base."""
    DeductiveClosure(OWLRL_Semantics).expand(kb)

def get_deduced_knowledge_copy(kb: rdflib.Graph):
    kb_ttl = serialize_kb(kb)
    kb_deduced = parse_kb_from_string(kb_ttl)
    deduce_knowledge(kb_deduced)
    return kb_deduced

def add_article_to_kb(article_id: str, kb: rdflib.Graph):
    """Adds an article (via the provided article_id) to the provided knwoledge base"""
    object_uuid = str(uuid.uuid4()) # Creates unique identifier following the uuid standard
    kb.add((OBJECT[object_uuid], RDF_TYPE, OWL_CLASS["Article"]))
    kb.add((OBJECT[object_uuid], OWL_DATA_PROP["index_id"], rdflib.Literal(article_id)))

def add_math_to_kb(display_math_ids: list, inline_math_ids: list, kb: rdflib.Graph):
    """Adds math expressions (via the provided math_ids) to the provided knowledge base"""
    for math_id in display_math_ids:
        _add_math_to_kb_single(math_id, "Display_math", kb)
    for math_id in inline_math_ids:
        _add_math_to_kb_single(math_id, "Inline_math", kb)

def _add_math_to_kb_single(math_id: str, owl_class_name: str, kb: rdflib.Graph):
    """Adds a single math expression (via the provided math_id) to the provided knowledge base"""
    object_uuid = str(uuid.uuid4())
    kb.add((OBJECT[object_uuid], RDF_TYPE, OWL_CLASS[owl_class_name]))
    kb.add((OBJECT[object_uuid], OWL_DATA_PROP["index_id"], rdflib.Literal(math_id)))

def _add_math_lists_from_dict_to_uri(info_dict: dict, object_uri: rdflib.URIRef, kb: rdflib.Graph):
    """Provided a dictionary with entries 'display_math_ids' and/or 'inline_math_ids',
    add those math ids as subdocument parts of the object with the given uri """
    for display_math_id in info_dict["display_math_ids"]:
        display_math_uri = get_object_uri_by_index_id(kb, display_math_id, "Display_math")
        kb.add((display_math_uri, OWL_OBJECT_PROP["sub_document_part_of"], OBJECT[object_uri]))
    for inline_math_id in info_dict["inline_math_ids"]:
        inline_math_uri = get_object_uri_by_index_id(kb, inline_math_id, "Inline_math")
        kb.add((inline_math_uri, OWL_OBJECT_PROP["sub_document_part_of"], OBJECT[object_uri]))

def add_chunks_to_kb(chunks: list[tuple[str, dict]], article_id: str, kb: rdflib.Graph):
    """Adds a list of chunks (via the provided chunk_id for each one) to the provided knowledge base
    Then, assigns it as subpart of the article and assigns the corresponding math expressions to each one."""
    for chunk_id, chunk_content in chunks:
        object_uuid = str(uuid.uuid4())
        kb.add((OBJECT[object_uuid], RDF_TYPE, OWL_CLASS["Chunk"]))
        kb.add((OBJECT[object_uuid], OWL_DATA_PROP["index_id"], rdflib.Literal(chunk_id)))
        article_uri = get_object_uri_by_index_id(kb, article_id, "Article")
        kb.add((OBJECT[object_uuid], OWL_OBJECT_PROP["sub_document_part_of"], article_uri))
        _add_math_lists_from_dict_to_uri(chunk_content, object_uuid, kb)

def add_extract_to_kb(extract: tuple[str, list[str], dict], kb: rdflib.Graph):
    """Adds an extract (via the provided extract_id for each one) to the provided knowledge base
    Then, assigns it as subpart of the stated chunks and assigns the corresponding math expressions to each one."""
    extract_id, extract_chunk_ids, extract_content = extract
    object_uuid = str(uuid.uuid4())
    kb.add((OBJECT[object_uuid], RDF_TYPE, OWL_CLASS["Extract"]))
    kb.add((OBJECT[object_uuid], OWL_DATA_PROP["index_id"], rdflib.Literal(extract_id)))
    for chunk_id in extract_chunk_ids:
        chunk_uri = get_object_uri_by_index_id(kb, chunk_id, "Chunk")
        kb.add((OBJECT[object_uuid], OWL_OBJECT_PROP["sub_document_part_of"], chunk_uri))
    _add_math_lists_from_dict_to_uri(extract_content, object_uuid, kb)

def add_new_pydantic_object_to_kb(pydantic_object: BaseModel, kb: rdflib.Graph, data_properties: list[str] = [], uri_hash_key = None) -> rdflib.URIRef:
    """Adds a new object to the knowledge base. The object is given as a Pydantic Model the class
    of which should correspond to the object's class name within the ontology.
    It is given a unique uid and assigned its OWL class.
    Optionally, a list of data properties may be specified to be added within the knowledge base.
    If a uri_hash_key is specified, the URI will be determined by a hash on that string, instead of a generic uuid.
    """
    if uri_hash_key is None:
        object_uri_end = str(uuid.uuid4())
    else:
        object_uri_end = hashlib.sha256(uri_hash_key.encode("utf-8")).hexdigest()[:20]
    object_uri = OBJECT[object_uri_end]
    object_class = str(type(pydantic_object).__name__)
    kb.add((object_uri, RDF_TYPE, OWL_CLASS[object_class]))
    for data_prop in data_properties:
        prop_val = getattr(pydantic_object, data_prop, None)  # Ensure the property exists in the object
        if prop_val is None:
            logger.warning(f"Data property {data_prop} is not included within given {object_class} object. Skipping the property.")
            continue
        kb.add((object_uri, OWL_DATA_PROP[data_prop], rdflib.Literal(prop_val)))
    return object_uri

def add_object_property_to_kb(subject_uri: str | rdflib.URIRef, prop_name: str, object_uri: rdflib.URIRef, kb: rdflib.Graph):
    """Add a given object property relation triplet based on two URIs (subject and object) and the name of the property"""
    subject_uri = rdflib.URIRef(subject_uri)
    kb.add((subject_uri, OWL_OBJECT_PROP[prop_name], object_uri))

def remove_object_from_kb(uri: str | rdflib.URIRef, kb: rdflib.Graph):
    """Removes an object determined by its URIRef by removing all triplets where it is mentioned."""
    uri = rdflib.URIRef(uri)
    kb.remove((uri, None, None))
    kb.remove((None, None, uri))

def add_data_property_to_kb(subject_uri: str | rdflib.URIRef, data_prop: str, prop_val, kb: rdflib.Graph, remove_old = True):
    subject_uri = rdflib.URIRef(subject_uri)
    if remove_old:
        kb.remove((subject_uri, OWL_DATA_PROP[data_prop], None))
    kb.add((subject_uri, OWL_DATA_PROP[data_prop], rdflib.Literal(prop_val)))

def get_data_property_of_object(kb: rdflib.Graph, subject_uri: rdflib.URIRef, data_property:str):
    query = f"""
    SELECT ?requested_data
    WHERE {{
        <{subject_uri}> <{OWL_DATA_PROP[data_property]}> ?requested_data
    }}
    """
    results = kb.query(query)
    return [row.requested_data for row in results]
    
def get_object_uri_by_index_id(kb: rdflib.Graph, index_id: str, class_name: str | None = None) -> rdflib.URIRef | None:
    """Retrieve the object URI in the knowledge base corresponding to the given index_id.
    If class_name is provided, it will filter the results to only include objects of that class.
    Returns the URI of the object if found, otherwise returns None.
    """
    
    query = f"""
    SELECT ?object
    WHERE {{
        ?object <{OWL_DATA_PROP['index_id']}> "{index_id}" ."""
    if class_name:
        query+=f"""
        ?object a <{OWL_CLASS[class_name]}> ."""
    query+=f"""
    }}
    """
    results = kb.query(query)
    for row in results:
        return row.object
    return None

def _inner_get_found_or_unfound_entity_representations(kb: rdflib.Graph, subdocument_part_uri: rdflib.URIRef | None, get_unfound: bool):
    kb_deduced = get_deduced_knowledge_copy(kb)

    with open("test_deduced_kb.ttl", "w") as f:
        f.write(serialize_kb(kb_deduced))

    if subdocument_part_uri is None:
        filter_str = "?doc_part"
    else:
        filter_str = "<"+str(subdocument_part_uri)+">"

    query = f"""
    SELECT ?entity ?repr ?repr_class
    WHERE {{
        ?repr a ?repr_class_uri .
        ?repr_class_uri rdfs:subClassOf* <{OWL_CLASS["Written_representation"]}> .
        ?entity <{OWL_OBJECT_PROP["written_as"]}> ?repr .
        
        FILTER {'NOT ' if get_unfound else ''}EXISTS {{
            ?repr <{OWL_OBJECT_PROP['found_in']}> {filter_str} .
        }}
        FILTER(STRSTARTS(STR(?repr_class_uri), "{str(OWL_CLASS)}"))
        FILTER NOT EXISTS {{
            ?repr a ?subClass .
            FILTER(isURI(?subClass))
            ?subClass rdfs:subClassOf+ ?repr_class_uri .
            FILTER(?subClass != ?repr_class_uri)
            FILTER NOT EXISTS {{
                ?repr_class_uri rdfs:subClassOf* ?subClass .
            }}
        }}
        BIND(STRAFTER(STR(?repr_class_uri), "{str(OWL_CLASS)}") AS ?repr_class)
    }}
    """

    results = kb_deduced.query(query)
    out = {}
    for row in results:
        if row.entity not in out:
            out[row.entity] = {
                "Written_representation": [],
                "Text_representation": [],
                "Mathematical_representation": []
            }
        out[row.entity][str(row.repr_class)].append(row.repr)
        
    return out

def get_unfound_entity_representations(kb: rdflib.Graph, subdocument_part_uri: rdflib.URIRef | None = None):
    """Gets all mathematical entities representations that have no outbound 'found_in' object property"""
    return _inner_get_found_or_unfound_entity_representations(kb, subdocument_part_uri, get_unfound=True)

def get_found_entity_representations(kb: rdflib.Graph, subdocument_part_uri: rdflib.URIRef | None = None):
    """Gets all mathematical entities representations that have outbound 'found_in' object property that is the given subdocument_part_uri (if provided)"""
    return _inner_get_found_or_unfound_entity_representations(kb, subdocument_part_uri, get_unfound=False)


def get_entities_with_same_representation(kb:  rdflib.Graph):

    query = f"""
    SELECT ?repr
            (GROUP_CONCAT(DISTINCT ?entity; separator=", ") AS ?entities_list)
            (COUNT(DISTINCT ?entity) AS ?count)
    WHERE {{
        ?entity <{OWL_OBJECT_PROP["written_as"]}> ?repr .
        ?entity a <{OWL_CLASS["Mathematical_entity"]}> .
        ?repr rdf:type/rdfs:subClassOf* <{OWL_CLASS["Written_representation"]}> .
    }}
    GROUP BY ?repr
    HAVING (COUNT(DISTINCT ?entity) > 1)
    """

    results = kb.query(query)
    out = {}
    for row in results:
        out[row.repr] = row.entities_list.split(", ")
    return out

def get_representations_of_object(kb: rdflib.Graph, object_uri: rdflib.URIRef):
    query = f"""
    SELECT ?repr
    WHERE {{
        <{object_uri}> <{OWL_OBJECT_PROP["written_as"]}> ?repr .
        ?repr rdf:type/rdfs:subClassOf* <{OWL_CLASS["Written_representation"]}> .
    }}
    """

    results = kb.query(query)
    return [row.repr for row in results]

def get_properties_of_object(kb: rdflib.Graph, object_uri: rdflib.URIRef):

    query = f"""
    SELECT ?property ?property_enunciate
    WHERE {{
        <{object_uri}> <{OWL_OBJECT_PROP["has_property"]}> ?property .
        ?property <{OWL_DATA_PROP['property_enunciate']}> ?property_enunciate .
    }}
    """

    results = kb.query(query)
    out = {}
    for row in results:
        out[row.property] = row.property_enunciate
    return out

def get_representation_str_and_type(kb: rdflib.Graph, repr_uri: rdflib.URIRef):
    query = f"""
    SELECT ?repr_str ?repr_type
    WHERE {{
        <{repr_uri}> rdf:type/rdfs:subClassOf* <{OWL_CLASS["Written_representation"]}> .
        <{repr_uri}> <{OWL_DATA_PROP['representation_str']}> ?repr_str .
        <{repr_uri}> <{OWL_DATA_PROP['representation_type']}> ?repr_type .
    }}
    """
    results = kb.query(query)
    for row in results:
        repr_str = str(row.repr_str)
        repr_type = str(row.repr_type)
    return repr_str, repr_type

def get_mathematical_representations_without_slt(kb: rdflib.Graph):
    query = f"""
    SELECT ?math_repr ?repr_str ?repr_type
    WHERE {{
        ?math_repr a <{OWL_CLASS["Mathematical_representation"]}> .
        ?math_repr <{OWL_DATA_PROP['representation_str']}> ?repr_str .
        ?math_repr <{OWL_DATA_PROP['representation_type']}> ?repr_type .
        FILTER NOT EXISTS {{
            ?math_repr <{OWL_DATA_PROP['representation_slt']}> ?repr_slt .
        }}
    }}
    """
    results = kb.query(query)
    out = {}
    for row in results:
        out[row.math_repr] = {
            "str": str(row.repr_str),
            "type": str(row.repr_type)
        }
    return out

def add_representation_of_mathematical_entity(kb: rdflib.Graph, representation: Written_representation, entity_uri: str | rdflib.URIRef):
    if representation.representation_type == "LaTeX":
        representation.representation_type = "latex"
    elif representation.representation_type in ["mathML", "MathML"]:
        representation.representation_type = "mathml"
    elif representation.representation_type in ["plain_text","plain text"]:
        representation.representation_type = "text"
    if representation.representation_type in ["latex" or "mathml"]:
        representation = Mathematical_representation(**representation.model_dump())
    elif representation.representation_type == "text":
        representation = Text_representation(**representation.model_dump())
    representation_uri = add_new_pydantic_object_to_kb(representation, kb, ["representation_str", "representation_type"], representation.representation_str+representation.representation_type)
    add_object_property_to_kb(entity_uri, "written_as", representation_uri, kb)
    
def add_entities_from_mathematical_entities_list(kb: rdflib.Graph, mathematical_entities_list: Mathematical_entites_list, doc_part_uri: str | rdflib.URIRef):
    doc_part_uri = rdflib.URIRef(doc_part_uri)
    for entity in mathematical_entities_list:
        logger.info(f"Added new entity. Representations and properties below.")
        entity_uri = add_new_pydantic_object_to_kb(entity, kb, data_properties=["is_standard_operator"])
        add_object_property_to_kb(entity_uri, "appears_in", doc_part_uri, kb)
        for representation in entity.representations_list:
            add_representation_of_mathematical_entity(kb, representation, entity_uri)
            logger.info(f"  Added representation '{representation.representation_str}' ({representation.representation_type})")
        for prop in entity.properties_list:
            prop_uri = add_new_pydantic_object_to_kb(prop, kb, ["property_enunciate"])
            add_object_property_to_kb(entity_uri, "has_property", prop_uri, kb)
            add_object_property_to_kb(prop_uri, "proposed_in", doc_part_uri, kb)
            if prop.is_confirmed:
                add_object_property_to_kb(prop_uri, "confirmed_in", doc_part_uri, kb)
            logger.info(f"  Added property: '{prop.property_enunciate}'. Confirmed: {prop.is_confirmed}")


def remove_orphan_representations_and_properties(kb: rdflib.Graph):

    query_repr = f"""
    SELECT ?repr
    WHERE {{
        ?repr rdf:type/rdfs:subClassOf* <{OWL_CLASS["Written_representation"]}> .
        FILTER NOT EXISTS {{
            ?object <{OWL_OBJECT_PROP['written_as']}> ?repr .
        }}
    }}
    """
    orphan_uris_list = [row.repr for row in kb.query(query_repr)]

    query_properties = f"""
    SELECT ?property
    WHERE {{
        ?property a <{OWL_CLASS["Mathematical_property"]}> .
        FILTER NOT EXISTS {{
            ?object <{OWL_OBJECT_PROP['has_property']}> ?property .
        }}
    }}
    """
    orphan_uris_list.extend([row.property for row in kb.query(query_properties)])

    for orphan_uri in orphan_uris_list:
        remove_object_from_kb(orphan_uri, kb)










def _____print_inferred_found_in(kb:  rdflib.Graph):
    query = f"""
    SELECT ?repr_str ?doc_part_id
    WHERE {{
        ?repr rdf:type/rdfs:subClassOf* <{OWL_CLASS["Written_representation"]}> .
        ?repr <{OWL_DATA_PROP['representation_str']}> ?repr_str .
        ?repr <{OWL_OBJECT_PROP['found_in']}> ?doc_part .
        ?doc_part <{OWL_DATA_PROP['index_id']}> ?doc_part_id .
    }}
    """
    results = kb.query(query)
    for row in results:
        print(row.repr_str, " | ", row.doc_part_id)

def get_property_enunciates_of_math_entity(kb: rdflib.Graph, entity_uri: rdflib.URIRef):
    
    enunciates_list = []

    query = f"""
    SELECT ?prop_enunciate
    WHERE {{
        <{entity_uri}> <{OWL_OBJECT_PROP['has_property']}> ?prop .
        ?prop <{OWL_DATA_PROP['property_enunciate']}> ?prop_enunciate.
    }}
    """

    results = kb.query(query)
    for row in results:
        enunciates_list.append(row.prop_enunciate)

    query = f"""
    SELECT ?is_oper
    WHERE {{
        <{entity_uri}> <{OWL_DATA_PROP['is_standard_operator']}> ?is_oper.
    }}
    """
        
    results = kb.query(query)
    if results:
        enunciates_list.append("Is standard operator")

    return enunciates_list


def print_math_entities(kb: rdflib.Graph):
    """Prints all the mathematical entities within the knowledge base,
    among with their representations, properties and associated information"""
    
    query = f"""
    SELECT ?entity
            (GROUP_CONCAT(DISTINCT ?repr_str; separator=", ") AS ?repr_str_list)
            (GROUP_CONCAT(DISTINCT COALESCE(?prop_enunciate, ""); separator=", ") AS ?prop_enunciate_list)
            ?is_oper
    WHERE {{
        ?entity a <{OWL_CLASS["Mathematical_entity"]}> .
        ?entity <{OWL_OBJECT_PROP['written_as']}> ?repr .
        ?repr rdf:type/rdfs:subClassOf* <{OWL_CLASS["Written_representation"]}> .
        ?repr <{OWL_DATA_PROP['representation_str']}> ?repr_str.
        OPTIONAL {{
            ?entity <{OWL_OBJECT_PROP['has_property']}> ?prop .
            ?prop <{OWL_DATA_PROP['property_enunciate']}> ?prop_enunciate.
        }}
        ?entity <{OWL_DATA_PROP['is_standard_operator']}> ?is_oper.
    }}
    GROUP BY ?entity
    """
    
    results = kb.query(query)
    table_str = ""
    for row in results:
        table_str+=f"{row.repr_str_list} | {row.prop_enunciate_list} | {row.is_oper} \n"

    print(table_str)


        