from pathlib import Path
import logging
from copy import deepcopy
import json
from typing import TypedDict
import re

from lxml import html

from src.lib.html_utils import parse_html, extract_and_substitute_mathml_elements, get_text_and_eq_per_dfs_elem
from src.lib.equation_utils import mathml_to_latex, get_slt_str_from_mathml_str
from src.lib.chunking_strategies import chunk_by_section, chunk_by_length, ChunkDict

logger = logging.getLogger(__name__)


SECTION_CHUNK_FLAG = "section"
LENGTH_CHUNK_FLAG = "length"
CHUNK_STRATEGIES = (SECTION_CHUNK_FLAG, LENGTH_CHUNK_FLAG)

class EquationDict(TypedDict):
    mathml_str: str
    latex_str: str
    slt_str: str

class ExtractDict(TypedDict):
    dfs_indices_list: list[int]         # list of DFS indices of the elements included in the extract
    inline_math_ids: list[str]          # list of ids for the inline math included in the chunks
    display_math_ids: list[str]         # list of ids for the display math included in the chunks
    text: str                           # plain text, stating formulas in latex

class ArticleProcessor():
    """Class that will hold the article html tree during runtime.
    It handles equation extraction, chunking, format conversion to
    LaTeX or plain text, among other utilities.
    """
    
    def __init__(self, article_path: Path):
        with open(article_path, encoding="utf-8") as f:
            html_str = f.read()
        
        complete_body_tree = parse_html(html_str)

        display_mathml_dict, inline_mathml_dict, self.mathless_html = extract_and_substitute_mathml_elements(complete_body_tree)
        logger.info("Processed article HTML to extract math expressions.")

        self.display_math_dict = { math_id: EquationDict(
            mathml_str = mathml_str,
            latex_str = mathml_to_latex(mathml_str),
            slt_str = get_slt_str_from_mathml_str(mathml_str))
            for math_id, mathml_str in display_mathml_dict.items()
        }
        self.inline_math_dict = { math_id: EquationDict(
            mathml_str = mathml_str,
            latex_str = mathml_to_latex(mathml_str),
            slt_str = get_slt_str_from_mathml_str(mathml_str))
            for math_id, mathml_str in inline_mathml_dict.items()
        }

        logger.info("Finished converting math expressions to LaTeX.")

        html_with_latex = self.get_html_with_latex()
        self.text_and_eq_per_dfs_elem = get_text_and_eq_per_dfs_elem(html_with_latex)

        self.chunks: list[ChunkDict] = []  # Will be populated by chunking methods
        self.eoi: ExtractDict | None = None  # Extract of interest. Will be populated by create_extract method
    
    def get_html_with_latex(self):
        tree = deepcopy(self.mathless_html)

        # Get element position for each equation id
        id_to_element_index: dict[str, int] = {}
        all_elements = list(tree.iter())
        for i, el in enumerate(all_elements):
            eq_id = el.get("id", "")
            if eq_id:
                id_to_element_index[eq_id] = i

        # Replace placeholder text with the latex string (preserve tail)
        total_math_dict = {**self.display_math_dict, **self.inline_math_dict}
        for eq_id, eq_data in total_math_dict.items():
            latex_str = eq_data["latex_str"]
            idx = id_to_element_index.get(eq_id)
            if idx is not None:
                el = all_elements[idx]
                if el.text:
                    el.text = latex_str
                if el.tail:
                    # Tail belongs to the next sibling — leave as-is
                    pass

        return tree

    def chunk_article(self, **kwargs):
        if self.chunks:
            logger.warning("Article has already been chunked. Overwriting existing chunks.")

        strategy = kwargs["name"]

        if strategy == SECTION_CHUNK_FLAG:
            # Get HTML with injected LaTeX
            latex_html_tree = self.get_html_with_latex()
            self.chunks = chunk_by_section(latex_html_tree, self.text_and_eq_per_dfs_elem)
        elif strategy == LENGTH_CHUNK_FLAG:
            self.chunks = chunk_by_length(self.text_and_eq_per_dfs_elem, kwargs["window"], kwargs["overlap"])
        else:
            raise ValueError(
                f"Unrecognized chunking strategy {strategy!r}. "
                f"Expected one of: {', '.join(CHUNK_STRATEGIES)!r}."
            )

        logger.info(f"Chunked article by strategy {strategy!r}. Total chunks created: {len(self.chunks)}.")

    def get_inline_math_ids(self):
        return self.inline_math_dict.keys()

    def get_display_math_ids(self):
        return self.display_math_dict.keys()

    def get_chunks_ids_and_content(self):
        out = []
        for i, chunk in enumerate(self.chunks):
            out.append((str(i),deepcopy(chunk)))
        return out

    def create_extract(self, extract_path: Path):
        """Create an ExtractDict object based on the JSON information within extract_path and the current article"""
        
        if self.eoi is not None:
            logger.warning("Extract of interest has already been created. Overwriting existing extract.")
        
        with open(extract_path, "r") as f:
            extract_json = json.load(f)
        ordered_parts = extract_json.get("ordered_parts", [])
        extract_dict: ExtractDict = {
            "dfs_indices_list": [],
            "inline_math_ids": [],
            "display_math_ids": [],
            "text": ""
        }

        # We get the DFS indices based on the self.mathless_html tree (which should give the same result as with the latex-injected tree (CHECK)))
        for part in ordered_parts:
            start_id = part["start"]["id"]
            start_tag = part["start"]["tag"]
            end_id = part["end"]["id"]
            end_tag = part["end"]["tag"]
            # Find the corresponding elements in the mathless_html tree
            start_el = self.mathless_html.xpath(f"//{start_tag}[@id='{start_id}']")
            end_el = self.mathless_html.xpath(f"//{end_tag}[@id='{end_id}']")
            if not start_el or not end_el:
                logger.warning(f"Could not find start or end element for part: {part}. Skipping.")
                continue
            start_el = start_el[0]
            end_el = end_el[0]
            # Get last children of end_el to find the correct end index
            while len(end_el) > 0:
                end_el = end_el[-1]
            # Get the DFS indices of these elements
            all_elements = list(self.mathless_html.iter())
            start_idx = all_elements.index(start_el)
            end_idx = all_elements.index(end_el)
            for idx in range(start_idx, end_idx + 1):
                extract_dict["dfs_indices_list"].append(idx)
                text_and_eq = self.text_and_eq_per_dfs_elem[idx]
                text = text_and_eq["text"]
                if text_and_eq["display_math_id"] is not None:
                    extract_dict["display_math_ids"].append(text_and_eq["display_math_id"])
                    # text = text.encode("unicode-escape").decode("utf-8")   # Not sure if I should provide it as unicode-escape
                elif text_and_eq["inline_math_id"] is not None:
                    extract_dict["inline_math_ids"].append(text_and_eq["inline_math_id"])
                    # text = text.encode("unicode-escape").decode("utf-8")
                extract_dict["text"] += text
            extract_dict["text"] += "\n"  # Add a newline after each part for readability

        self.eoi = extract_dict

    def get_extract_id_chunks_and_content(self):
        chunk_ids = self.get_chunk_idxs_for_extract()
        return ("eoi", chunk_ids, deepcopy(self.eoi))

    def get_chunk_idxs_for_extract(self, extract: ExtractDict | None = None):
        """Get the chunks that correspond to the given extract of interest."""
        if not self.chunks:
            logger.warning("Article has not been chunked yet. Returning empty list.")
            return []
        if extract is None:
            if self.eoi is None:
                logger.warning("No extract of interest has been created. Returning empty list.")
                return []
            extract = self.eoi

        extract_dfs_indices = set(extract["dfs_indices_list"])
        relevant_chunks = []
        for idx, chunk in enumerate(self.chunks):
            chunk_dfs_indices = set(range(chunk["init_html_dfs_idx"], chunk["end_html_dfs_idx"] + 1))
            if extract_dfs_indices.intersection(chunk_dfs_indices):
                relevant_chunks.append(str(idx))

        return relevant_chunks

    def get_extract_id(self, extract: ExtractDict | None = None):
        if extract is not None:
            logger.warining("No other extract than the eoi is implemented yet. Ignoring input and returning id of eoi.")
        
        return "eoi"

    def get_extract_text(self, extract: ExtractDict | None = None):
        if extract is None:
            if self.eoi is None:
                logger.warning("No extract of interest has been created. Returning empty string.")
                return ""
            extract = self.eoi

        return extract["text"]

    def get_extract_text_without_math(self, extract: ExtractDict | None = None):
        if extract is None:
            if self.eoi is None:
                logger.warning("No extract of interest has been created. Returning empty string.")
                return ""
            extract = self.eoi

        text = ""
        for idx in self.eoi["dfs_indices_list"]:
            text_and_eq = self.text_and_eq_per_dfs_elem[idx]
            display_math_ids = text_and_eq["display_math_id"]
            inline_math_ids = text_and_eq["inline_math_id"]
            if display_math_ids is None and inline_math_ids is None:
                text += text_and_eq["text"]
        return text

    def find_slts_in_extract(self, slts_list: list[str], extract: ExtractDict | None = None ):
        def _turn_style_diff_into_string(style_diff_list):
            return " & ".join([f"Change {diff['character']} from {'-'.join(diff['query_style'])} to {'-'.join(diff['doc_style'])}" for diff in style_diff_list])

        if extract is None:
            if self.eoi is None:
                logger.warning("No extract of interest has been created. Returning empty string.")
                return ""
            extract = self.eoi

        matched_targets_for_all_slts = []
        suggestions_all_reprs = {}
        for slt_str in slts_list:
            logger.info(f"Searching for SLT {slt_str} in extract.")
            matched_targets = {
                "display_math": [],
                "inline_math": []
            }
            suggestions = []
            for id in self.eoi["display_math_ids"]:
                print(search_slt_in_doc_slt(slt_str, self.display_math_dict[id]["slt_str"]))



            
        #             if latex_search_result["exact_style_match"]:
        #                 matched_targets["display_math"].append(id)
        #             else:
        #                 for style_diff in latex_search_result["style_differences"]:
        #                     suggestions[repr_data["uri"]].add(_turn_style_diff_into_string(style_diff))
        #         for id in self.eoi["inline_math_ids"]:
        #             mathml_str = self.inline_math_dict[id]["mathml_str"]
        #             latex_search_result = search_latex_in_mathml(pattern, mathml_str)
        #             if latex_search_result["exact_style_match"]:
        #                 matched_targets["inline_math"].append(id)
        #             else:
        #                 for style_diff in latex_search_result["style_differences"]:
        #                     suggestions[repr_data["uri"]].add(_turn_style_diff_into_string(style_diff))
        #         if not matched_targets["display_math"] and not matched_targets["inline_math"]:
        #             suggestions_all_reprs = {**suggestions_all_reprs, **suggestions}
        #     elif pattern_type == "mathml":
        #         logger.warning("MathML matching not implemented yet. Returning no match.")
        #     else:
        #         logger.warning(f"Unrecognized pattern type {pattern_type!r}. Returning no match.")
        #     matched_targets_all_reprs.append(matched_targets)
        # return matched_targets_all_reprs, suggestions_all_reprs
        


    def find_patterns_in_extract(self, reprs_list: list[dict], extract: ExtractDict | None = None ):

        def _turn_style_diff_into_string(style_diff_list):
            return " & ".join([f"Change {diff['character']} from {'-'.join(diff['query_style'])} to {'-'.join(diff['doc_style'])}" for diff in style_diff_list])

        if extract is None:
            if self.eoi is None:
                logger.warning("No extract of interest has been created. Returning empty string.")
                return ""
            extract = self.eoi

        text_only = self.get_extract_text_without_math()

        matched_targets_all_reprs = []
        suggestions_all_reprs = {}
        for repr_data in reprs_list:
            logger.info(f"Searching for pattern {repr_data['str']} of type {repr_data['type']} in extract.")
            pattern = repr_data["str"]
            pattern_type = repr_data["type"]
            matched_targets = {
                "uri": repr_data["uri"],
                "extract": [],
                "chunk": [],
                "display_math": [],
                "inline_math": []
            }
            suggestions = {
                repr_data["uri"]: set()
            }
            if pattern_type == "text":
                try:
                    compiled_pattern = re.compile(pattern)
                except re.error as e:
                    logger.warning(f"Invalid regex pattern {pattern!r}: {e}. Returning no match.")
                    matched_targets_all_reprs.append(matched_targets)
                    continue
                match = compiled_pattern.search(text_only)
                if match:
                    matched_targets["extract"].append(self.get_extract_id())
            elif pattern_type == "latex":
                for id in self.eoi["display_math_ids"]:
                    mathml_str = self.display_math_dict[id]["mathml_str"]
                    latex_search_result = search_latex_in_mathml(pattern, mathml_str)
                    if latex_search_result["exact_style_match"]:
                        matched_targets["display_math"].append(id)
                    else:
                        for style_diff in latex_search_result["style_differences"]:
                            suggestions[repr_data["uri"]].add(_turn_style_diff_into_string(style_diff))
                for id in self.eoi["inline_math_ids"]:
                    mathml_str = self.inline_math_dict[id]["mathml_str"]
                    latex_search_result = search_latex_in_mathml(pattern, mathml_str)
                    if latex_search_result["exact_style_match"]:
                        matched_targets["inline_math"].append(id)
                    else:
                        for style_diff in latex_search_result["style_differences"]:
                            suggestions[repr_data["uri"]].add(_turn_style_diff_into_string(style_diff))
                if not matched_targets["display_math"] and not matched_targets["inline_math"]:
                    suggestions_all_reprs = {**suggestions_all_reprs, **suggestions}
            elif pattern_type == "mathml":
                logger.warning("MathML matching not implemented yet. Returning no match.")
            else:
                logger.warning(f"Unrecognized pattern type {pattern_type!r}. Returning no match.")
            matched_targets_all_reprs.append(matched_targets)
        return matched_targets_all_reprs, suggestions_all_reprs

    def write_chunks_to_file(self, output_path: Path):
        """Write the chunks to a text file for inspection."""
        out = ""
        for chunk in self.chunks:
            out += chunk["text"]
            out += "\n"
            out += str(chunk["init_html_dfs_idx"])
            out += "\n"
            out += str(chunk["end_html_dfs_idx"])
            out += "\n"
            out += str(chunk["display_math_ids"])
            out += "\n"
            out += str(chunk["inline_math_ids"])
            out += "\n"
            out += str(chunk["misc"])
            out += "\n=====================================================================\n"
        with open(output_path, "w") as f:
            f.write(out)

    def write_extract_to_file(self, output_path: Path):
        """Write the extract of interest to a text file for inspection."""
        if self.eoi is None:
            logger.warning("No extract of interest has been created. Nothing to write.")
            return
        out = ""
        out += "DFS indices list:\n"
        out += str(self.eoi["dfs_indices_list"])
        out += "\n\nExtracted text:\n"
        out += self.eoi["text"]
        out += "\n\nDisplay math IDs:\n"
        out += str(self.eoi["display_math_ids"])
        out += "\n\nInline math IDs:\n"
        out += str(self.eoi["inline_math_ids"])
        with open(output_path, "w") as f:
            f.write(out)