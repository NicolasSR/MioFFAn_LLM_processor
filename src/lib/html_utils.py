import logging
from copy import deepcopy
import re

from lxml import html

logger = logging.getLogger(__name__)

DISPLAY_MATH_PLACEHOLDER_CLASS = "display_math_placeholder"
INLINE_MATH_PLACEHOLDER_CLASS = "inline_math_placeholder"

def parse_html(html_str: str) -> html.HtmlElement:
    """Parse an HTML string into an lxml.html.HtmlElement. keep only body section."""
    full_tree = html.document_fromstring(html_str.encode("utf-8"))
    body = full_tree.find(".//body")
    tree = deepcopy(body)
    return tree

def _serialize_mathml(mathml_element: html.HtmlElement) -> str:
    """Serialize a MathML element to a plain string (no surrounding tags)."""
    return html.tostring(mathml_element, encoding="unicode", method="xml")

def _process_inline_mathml(
    root: html.HtmlElement,
    formula_ids: set[str],
) -> tuple[dict[str, str]]:
    """Find all remaining <math> elements (not inside formula divs) and substitute them."""
    inline_dict: dict[str, str] = {}
    inline_counter = 0

    math_elements = root.xpath("//math")
    for math_elem in math_elements:
        # Skip if this math element is inside a formula div (already handled)
        parent = math_elem.getparent()
        if parent is not None and parent.get("class") == "formula":
            continue

        # Check if this math element already has an ID
        math_id = math_elem.get("id")
        if math_id and math_id in formula_ids:
            continue

        # Assign a new inline ID
        inline_counter += 1
        placeholder_id = f"inline_{inline_counter}"

        # Serialize the MathML
        mathml_str = _serialize_mathml(math_elem)
        inline_dict[placeholder_id] = mathml_str

        # Replace with placeholder span
        placeholder = html.Element(
            "span",
            {"class": INLINE_MATH_PLACEHOLDER_CLASS, "id": placeholder_id},
        )
        placeholder.text = f"[{placeholder_id}]"

        math_elem.addnext(placeholder)
        parent = math_elem.getparent()
        if parent is not None:
            parent.remove(math_elem)

    return inline_dict

def extract_and_substitute_mathml_elements(
    root: html.HtmlElement,
) -> tuple[dict[str, str], dict[str, str], html.HtmlElement]:
    """Extract and substitute all MathML elements from the tree with placeholders.

    Block formulas (inside ``<div class="formula">``) are processed first, then
    inline expressions. Inline elements get an ``inline_N`` id assigned based
    on document order.

    Returns a deep-copy of the original tree with all ``<math>`` replaced by
    ``<div class=DISPLAY_MATH_PLACEHOLDER_CLASS>`` or ``<span class=INLINE_MATH_PLACEHOLDER_CLASS>``.
    """
    root_copy = deepcopy(root)

    # ------------------------------------------------------------------ #
    # 1. Extract and substitute block formulas                           #
    # ------------------------------------------------------------------ #
    formula_divs = root_copy.xpath("//div[@class='formula']")
    formulas_dict: dict[str, str] = {}

    for formula_div in formula_divs:
        formula_id = formula_div.get("id", "")
        if not formula_id:
            continue

        math_elem = formula_div.find(".//math")
        if math_elem is None:
            continue

        mathml_str = _serialize_mathml(math_elem)
        formulas_dict[formula_id] = mathml_str

        # Replace formula div with placeholder
        placeholder = html.Element(
            "div",
            {"class": DISPLAY_MATH_PLACEHOLDER_CLASS, "id": formula_id},
        )
        placeholder.text = f"[{formula_id}]"

        formula_div.addnext(placeholder)
        parent = formula_div.getparent()
        if parent is not None:
            parent.remove(formula_div)

    # ------------------------------------------------------------------ #
    # 2. Extract and substitute remaining inline <math> elements         #
    # ------------------------------------------------------------------ #
    inline_dict = _process_inline_mathml(root_copy, set(formulas_dict.keys()))

    return formulas_dict, inline_dict, root_copy

def get_text_and_eq_per_dfs_elem(root: html.HtmlElement):
    """Get a list of dictionaries containing the text and inline/display math id for each element in DFS order."""
    
    def recursion_for_last_siblings(el, cumulative_text):
        # Get tails of all ancestors of el that are last sibling
        if el.getnext() is None:  # Meaning current element is the last sibling
            parent_el = el.getparent()
            if parent_el is not None:
                cumulative_text += parent_el.tail if parent_el.tail else ""
                return recursion_for_last_siblings(parent_el, cumulative_text)
            else:
                return cumulative_text
        else:
            return cumulative_text
        
    text_and_eq_list = []

    for el in list(root.iter()):
        own_text = el.text if el.text else ""
        extra_text = ""

        # Children and parent check
        # No children means that its own tail has to be added.
        # If so, then we also check for the ancestor's tails.
        children = list(el)
        if not children:
            extra_text += el.tail if el.tail else ""
            extra_text = recursion_for_last_siblings(el, extra_text)
        total_text = own_text+extra_text

        # Certain tags should not add a newline character to the text, as they are inline elements.
        if el.tag in ["span"]:
            total_text = total_text.removesuffix("\n      ")

        elem_dict = {
            "text": total_text,
            "inline_math_id": None,
            "display_math_id": None
        }

        # If element is an equation, get equation id
        if el.tag == "div" and el.attrib.get("class", "") == DISPLAY_MATH_PLACEHOLDER_CLASS:
            elem_dict["display_math_id"] = el.attrib["id"]
        elif el.tag == "span" and el.attrib.get("class", "") == INLINE_MATH_PLACEHOLDER_CLASS:
            elem_dict["inline_math_id"] = el.attrib["id"]

        text_and_eq_list.append(elem_dict)

    return text_and_eq_list


def divide_html_by_sections(html_root: html.HtmlElement):
    """Divide the HTML tree into sections based on <h1>, <h2>... tags."""

    # Find at which header level the paper title is located (h1, h2, ..., h6)
    for h_idx in range(1, 7):
        article_title = html_root.find(f".//h{h_idx}//span[@class='article-title']")
        if article_title is not None:
            logger.info(f"Found h-tag h{h_idx} being used for the article title")
            base_section_level = h_idx  # This level for the title, next level for sections
            break
    else:
        logger.warning("No article title found in the HTML. Setting h1 as the default section level.")
        base_section_level = 1

    section_indexing = [0]*6
    h_pattern = re.compile(r"h(\d)")
    sections = []
    end_idx = len(list(html_root.iter()))  # Default to end of document if no higher-level header is found
    for el_idx, el in enumerate(html_root.iter()):

        h_pattern_match = h_pattern.search(el.tag)

        if h_pattern_match is None:
            continue

        h_idx = int(h_pattern_match.group(1))
        section_level = h_idx - base_section_level

        if section_level < 0:
            # We assume that any level above the basic_section_level implies that the paper content is over.
            logger.warning("Section splitting stopped early as it found an h-tag priority higher than the article title.")
            end_idx = el_idx
            break

        # Get current section index
        section_indexing[section_level] += 1
        for i in range(section_level + 1, 6):
            section_indexing[i] = 0

        section_label = el.find(".//span[@class='section-label']")
        section_title = el.find(".//span[@class='section-title']")
        section_dict = {}
        section_dict["index"] = ".".join([str(i) for i in section_indexing])
        section_dict["label"] = section_label.text_content().strip() if section_label is not None else "Unlabeled"
        section_dict["title"] = section_title.text_content().strip() if section_title is not None else "Untitled"
        section_dict["init_html_dfs_idx"] = el_idx

        sections.append(section_dict)

    return sections, end_idx
