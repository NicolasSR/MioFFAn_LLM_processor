from types import SimpleNamespace

import rdflib

from src.lib.knowledge_base_api import (
    OBJECT,
    OWL_CLASS,
    OWL_DATA_PROP,
    OWL_OBJECT_PROP,
    RDF_TYPE,
    parse_kb_from_string,
)
from src.nodes.entity_extraction_nodes import ensure_findable_symbols_on_extract_node


class FakeArticleProcessor:
    def __init__(self, symbol_uri: rdflib.URIRef):
        self.symbol_uri = symbol_uri
        self.find_patterns_calls = 0

    def get_extract_text(self) -> str:
        return "The extract text mentions velocity and acceleration."

    def get_extract_id(self) -> str:
        return "eoi"

    def find_patterns_in_extract(self, symbols_list: list[dict]):
        self.find_patterns_calls += 1
        assert symbols_list == [
            {"uri": self.symbol_uri, "str": "velocity", "type": "text"}
        ]
        return ([{
            "uri": self.symbol_uri,
            "extract": ["eoi"],
            "chunk": [],
            "display_math": [],
            "inline_math": [],
        }], {})


class UnexpectedLlm:
    def invoke_client(self, *args, **kwargs):
        raise AssertionError("The LLM fallback should not be called in this test")


def make_state_and_symbol(*, already_found: bool = False):
    kb = rdflib.Graph()
    extract_uri = OBJECT["test-extract"]
    entity_uri = OBJECT["test-entity"]
    symbol_uri = OBJECT["test-symbol"]

    kb.add((extract_uri, RDF_TYPE, OWL_CLASS["Extract"]))
    kb.add((extract_uri, OWL_DATA_PROP["index_id"], rdflib.Literal("eoi")))
    kb.add((entity_uri, RDF_TYPE, OWL_CLASS["Mathematical_entity"]))
    kb.add((symbol_uri, RDF_TYPE, OWL_CLASS["Symbol"]))
    kb.add((symbol_uri, OWL_DATA_PROP["representation_str"], rdflib.Literal("x")))
    kb.add((symbol_uri, OWL_DATA_PROP["representation_type"], rdflib.Literal("text")))
    kb.add((entity_uri, OWL_OBJECT_PROP["written_as"], symbol_uri))
    if already_found:
        kb.add((symbol_uri, OWL_OBJECT_PROP["found_in"], extract_uri))

    state = {"knowledge_base": kb.serialize(format="ttl")}
    return state, symbol_uri, extract_uri


def make_runtime(article_processor: FakeArticleProcessor):
    return SimpleNamespace(
        context={
            "article_processor": article_processor,
            "llm_api": UnexpectedLlm(),
        }
    )


def test_node_marks_pattern_match_as_found_in_extract():
    state, symbol_uri, extract_uri = make_state_and_symbol()
    article_processor = FakeArticleProcessor(symbol_uri)

    result = ensure_findable_symbols_on_extract_node(
        state,
        make_runtime(article_processor),
    )

    result_kb = parse_kb_from_string(result["knowledge_base"])
    assert (symbol_uri, OWL_OBJECT_PROP["found_in"], extract_uri) in result_kb
    assert article_processor.find_patterns_calls == 1


def test_node_skips_pattern_search_when_symbol_is_already_found():
    state, _, _ = make_state_and_symbol(already_found=True)
    article_processor = FakeArticleProcessor(OBJECT["test-symbol"])

    result = ensure_findable_symbols_on_extract_node(
        state,
        make_runtime(article_processor),
    )

    assert result is state
    assert article_processor.find_patterns_calls == 0
