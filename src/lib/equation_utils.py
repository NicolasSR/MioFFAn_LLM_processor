import logging
import unicodedata

import pypandoc
from lxml import etree

from tangent_l.convert import get_slt_from_mathml

logger = logging.getLogger(__name__)


# Order matters: check multi-word variants before single-word variants
VARIANT_MAP = [
    ("SANS-SERIF BOLD ITALIC", "sans-serif-bold-italic"),
    ("SANS-SERIF BOLD", "bold-sans-serif"),
    ("SANS-SERIF ITALIC", "sans-serif-italic"),
    ("SANS-SERIF", "sans-serif"),
    ("BOLD ITALIC", "bold-italic"),
    ("BOLD SCRIPT", "bold-script"),
    ("BOLD FRAKTUR", "bold-fraktur"),
    ("DOUBLE-STRUCK", "double-struck"),
    ("SCRIPT", "script"),
    ("FRAKTUR", "fraktur"),
    ("MONOSPACE", "monospace"),
    ("BOLD", "bold"),
    ("ITALIC", "italic"),
]

# Legacy Letterlike Symbols (U+2100–U+214F) that lack "MATHEMATICAL" in their Unicode name
LEGACY_MATH_CHARS = {
    "\u2102": ("C", "double-struck"),   # ℂ
    "\u210A": ("g", "script"),          # ℊ
    "\u210B": ("H", "script"),          # ℋ
    "\u210C": ("H", "fraktur"),         # ℌ
    "\u210D": ("H", "double-struck"),   # ℍ
    "\u210E": ("h", "italic"),          # ℎ (Planck constant)
    "\u210F": ("\u0127", "italic"),     # ℏ (Planck constant over 2pi -> ħ)
    "\u2110": ("I", "script"),          # ℐ
    "\u2111": ("I", "fraktur"),         # ℑ
    "\u2112": ("L", "script"),          # ℒ
    "\u2113": ("l", "script"),          # ℓ
    "\u2115": ("N", "double-struck"),   # ℕ
    "\u2119": ("P", "double-struck"),   # ℙ
    "\u211A": ("Q", "double-struck"),   # ℚ
    "\u211B": ("R", "script"),          # ℛ
    "\u211C": ("R", "fraktur"),         # ℜ
    "\u211D": ("R", "double-struck"),   # ℝ
    "\u2124": ("Z", "double-struck"),   # ℤ
    "\u2128": ("Z", "fraktur"),         # ℨ
    "\u212C": ("B", "script"),          # ℬ
    "\u212D": ("C", "fraktur"),         # ℭ
    "\u212F": ("e", "script"),          # ℯ
    "\u2130": ("E", "script"),          # ℰ
    "\u2131": ("F", "script"),          # ℱ
    "\u2133": ("M", "script"),          # ℳ
    "\u2134": ("o", "script"),          # ℴ
}

TOKEN_TAGS_MATHML_TEXT = {"mi", "mn", "mo", "mtext", "ms"}

def mathml_to_latex(mathml_string: str) -> str:
    """Convert a MathML XML fragment to LaTeX.
    """
    try:
        latex = pypandoc.convert_text(mathml_string, 'latex', format='html')
        latex = latex.removeprefix("\\(").removesuffix("\\)\n")
        return latex
    except Exception as exc:
        logger.error("MathML to LaTeX conversion failed: %s", exc)
        return f"[LaTeX conversion error]{mathml_string}"


def latex_to_mathml(latex_string: str, as_string = True) -> str:
    """Convert a LaTeX math fragment to MathML.
    """
    try:
        latex_string = "\\("+latex_string+"\\)\n"
        mathml_str = pypandoc.convert_text(latex_string, 'html', format='latex', extra_args=['--mathml'])
        root = etree.fromstring(mathml_str)
        math_node = root[0] if root.tag == 'p' else root
        ns = '{http://www.w3.org/1998/Math/MathML}'
        etree.strip_elements(math_node, f'{ns}annotation')
        etree.strip_tags(math_node, f'{ns}semantics')
        if as_string:
            return etree.tostring(math_node, encoding='unicode')
        return etree
    except Exception as exc:
        logger.error("LaTeX to MathML conversion failed: %s", exc)
        if as_string:
            return f"[MathML conversion error]{latex_string}"
        else:
            return None

def parse_math_char(char: str) -> tuple[str, str | None]:
    """Returns (base_char, mathvariant) or (char, None) if unstyled."""
    if char in LEGACY_MATH_CHARS:
        return LEGACY_MATH_CHARS[char]

    name = unicodedata.name(char, "")
    if "MATHEMATICAL" in name:
        base_char = unicodedata.normalize("NFKD", char)
        for keyword, variant in VARIANT_MAP:
            if keyword in name:
                return base_char, variant
        return base_char, "normal"

    return char, None

def group_runs(text: str) -> list[tuple[str, str | None]]:
    """Groups consecutive characters sharing the same mathvariant."""
    runs: list[tuple[str, str | None]] = []
    for ch in text:
        base, variant = parse_math_char(ch)
        if runs and runs[-1][1] == variant:
            runs[-1] = (runs[-1][0] + base, variant)
        else:
            runs.append((base, variant))
    return runs

def local_name(tag: str) -> str:
    """Strips XML namespace prefix if present."""
    return etree.QName(tag).localname

def normalize_mathml_formatting(mathml_str: str) -> str:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(mathml_str.encode("utf-8"), parser)

    # Iterate over a static list because we may insert sibling elements
    for elem in list(root.iter()):
        tag = local_name(elem.tag)
        if tag not in TOKEN_TAGS_MATHML_TEXT or not elem.text:
            continue

        runs = group_runs(elem.text)
        existing_variant = elem.get("mathvariant")

        # Case 1: Entire node shares a single style (most common: <mi>𝐮</mi>, <mi>ℝ</mi>)
        if len(runs) == 1:
            text, variant = runs[0]
            elem.text = text
            if variant and not existing_variant:
                elem.set("mathvariant", variant)

        # Case 2: Mixed styles inside one node (e.g., <mi>𝐮x</mi> or <mtext>Vector 𝐮</mtext>)
        else:
            parent = elem.getparent()
            if parent is None:
                continue

            new_nodes = []
            # Update the original node with the first run
            first_text, first_variant = runs[0]
            elem.text = first_text
            if first_variant:
                elem.set("mathvariant", first_variant)
            elif existing_variant:
                elem.set("mathvariant", existing_variant)

            # Create sibling nodes for remaining runs
            for run_text, run_variant in runs:
                new_elem = etree.Element(elem.tag, attrib=dict(elem.attrib))
                new_elem.text = run_text
                if run_variant:
                    new_elem.set("mathvariant", run_variant)
                elif "mathvariant" in new_elem.attrib and not existing_variant:
                    del new_elem.attrib["mathvariant"]
                new_nodes.append(new_elem)

            qname = etree.QName(elem.tag)
            mrow_tag = f"{{{qname.namespace}}}mrow" if qname.namespace else "mrow"
            wrapper = etree.Element(mrow_tag)
            wrapper.extend(new_nodes)
            parent.replace(elem, wrapper)




            

    return etree.tostring(root, encoding="unicode")

def get_slt_str_from_mathml_str(mathml_string):
    mathml_string = normalize_mathml_formatting(mathml_string)
    return get_slt_from_mathml(mathml_string, ignore_formatting=False)

def get_slt_str_from_latex(latex_string):
    mathml_string = latex_to_mathml(latex_string)
    return get_slt_str_from_mathml_str(mathml_string)

# def slt_to_json_ready(tree):
#     """Recursively converts tuples and frozensets into JSON-safe lists."""
#     if isinstance(tree, list):
#         return [slt_to_json_ready(node) for node in tree]
    
#     tag, text, styles, children = tree
#     return [
#         tag, 
#         text, 
#         list(styles),  # frozenset -> list
#         [slt_to_json_ready(c) for c in children]  # tuple -> list
#     ]

# def slt_to_str(tree):
#     return json.dumps(slt_to_json_ready(tree))

# def json_ready_to_slt(data):
#     """Recursively reconstructs the exact tuple/frozenset tree from parsed JSON."""
#     if isinstance(data, list) and (not data or isinstance(data[0], list)):
#         return [json_ready_to_slt(node) for node in data]
    
#     tag, text, styles, children = data
#     return (
#         tag,
#         text,
#         frozenset(styles),  # list -> frozenset
#         tuple(json_ready_to_slt(c) for c in children)  # list -> tuple
#     )

# def str_to_slt(tree_str):
#     tree_json = json.loads(tree_str)
#     return json_ready_to_slt(tree_json)

# def get_slt_from_mathml(element, inherited_styles=None, as_json_str=True):
#     """
#     Parses MathML into a Symbol Layout Tree formed by (tag, text, styles, children) tuples.
#     Tags and texts represent structure; styles represent formatting.
#     """
#     if isinstance(element, str):
#         element = etree.fromstring(element)

#     if inherited_styles is None:
#         inherited_styles = frozenset()

#     # Track styles passing down through the tree
#     current_styles = set(inherited_styles)
#     variant = element.attrib.get('mathvariant', '').lower()
    
#     if 'bold' in variant: current_styles.add('bold')
#     if 'double-struck' in variant: current_styles.add('bb')
#     if 'fraktur' in variant: current_styles.add('frak')
#     if 'script' in variant: current_styles.add('cal')
    
#     tag = element.tag.split('}')[-1]
#     ignore_tags = {'math', 'mrow', 'mstyle', 'mpadded', 'mphantom', 'semantics'}
    
#     # Process children
#     children = []
#     for child in element:
#         child_tree = get_slt_from_mathml(child, current_styles, False)
#         # If a child was an ignored tag (like mrow), it returns a list of its children
#         if isinstance(child_tree, list):
#             children.extend(child_tree)
#         elif child_tree:
#             children.append(child_tree)
            
#     # Extract text and embedded Unicode styles
#     text = element.text.strip() if element.text and element.text.strip() else None
#     if text:
#         base_text = unicodedata.normalize('NFKC', text)
#         for char in text:
#             name = unicodedata.name(char, "")
#             if name.startswith("MATHEMATICAL"):
#                 if "BOLD" in name: current_styles.add('bold')
#                 if "DOUBLE-STRUCK" in name: current_styles.add('bb')
#                 if "FRAKTUR" in name: current_styles.add('frak')
#                 if "SCRIPT" in name: current_styles.add('cal')
#     else:
#         base_text = None
        
#     if tag in ignore_tags:
#         if as_json_str:
#             return slt_to_str(children)
#         return children # Pass children up through the ignored layer

#     # Strip semantic information from symbols
#     if tag in ["mi", "mo", "mn", "mtext"]:
#         tag = "miont"

#     slt_tuple = (tag, base_text, frozenset(current_styles), tuple(children))

#     if as_json_str:
#         return slt_to_str(slt_tuple)
#     return slt_tuple

# def get_slt_from_latex(latex_str, as_json_str = True):
#     mathml_tree_str = latex_to_mathml(latex_str, as_string=True)  # TODO: see why we get an error when passing mathml directly as tree instead of string.
#     return get_slt_from_mathml(mathml_tree_str, as_json_str=as_json_str)

# def compare_structure(query_node, doc_node):
#     """
#     Checks if two nodes match structurally. 
#     Returns `None` if they don't match.
#     Returns a list of style differences if they DO match structurally.
#     """
#     q_tag, q_text, q_styles, q_children = query_node
#     d_tag, d_text, d_styles, d_children = doc_node
    
#     # STRUCTURAL CHECK: Tags, base text, and number of children must be identical
#     if q_tag != d_tag or q_text != d_text or len(q_children) != len(d_children):
#         return None 
        
#     style_diffs = []
    
#     # STYLE CHECK: If structure matches, did the fonts differ?
#     if q_styles != d_styles and q_text:
#         style_diffs.append({
#             'character': q_text,
#             'query_style': list(set(q_styles)) or ['normal'],
#             'doc_style': list(set(d_styles)) or ['normal']
#         })
        
#     # Recursively check all children
#     for qc, dc in zip(q_children, d_children):
#         child_diffs = compare_structure(qc, dc)
#         if child_diffs is None:
#             return None # A child failed the structural match
#         style_diffs.extend(child_diffs)
        
#     return style_diffs

# def add_style_differences_if_new(style_differences_list, new_style_differences):
#     for new_style_diff in new_style_differences:
#         is_new = False
#         new_style_diff_str = json.dumps({i: diff for i, diff in enumerate(new_style_diff)}, sort_keys=True)
#         if not style_differences_list:
#             is_new = True
#         for style_diff in style_differences_list:
#             style_diff_str = json.dumps({i: diff for i, diff in enumerate(style_diff)}, sort_keys=True)
#             if style_diff_str != new_style_diff_str:
#                 is_new = True
#         if is_new:
#             style_differences_list.append(new_style_diff)
#     return style_differences_list

# def update_match_result(old_match_result, new_match_result):
#     is_exact = old_match_result["exact_style_match"] or new_match_result["exact_style_match"]
#     return {
#         'exact_style_match': is_exact,
#         'style_differences': add_style_differences_if_new(old_match_result["style_differences"],new_match_result["style_differences"])
#     }

# def find_stl_subtree_matches(query_tree, doc_tree, previous_result = None, break_if_exact = True):
#     """Recursively searches the document tree for structural matches."""
#     if previous_result is None:
#         current_result = {
#             'exact_style_match': False,
#             'style_differences': []
#         }
#     else:
#         current_result = previous_result

#     # Try matching the query against the current document node
#     diffs = compare_structure(query_tree, doc_tree)
#     if diffs is not None:
#         is_exact = len(diffs) == 0
#         new_result = {
#             'exact_style_match': is_exact,
#             'style_differences': [] if is_exact else [diffs]
#         }
#         current_result = update_match_result(current_result, new_result)
#         if break_if_exact and current_result['exact_style_match']:
#             return current_result
        
#     # Recurse into document children
#     _, _, _, d_children = doc_tree
#     for child in d_children:
#         current_result = find_informed_slt_matches(query_tree, child, previous_result=current_result)
#         if break_if_exact and current_result['exact_style_match']:
#             return current_result
        
#     return current_result

# def find_matches_among_slt_trees(query_tree, initial_results):




# def find_matches_of_slt_tree_against_compound_slts(query_tree, initial_results, only_check_first_sibling = False):
#     """Search a query SLT with unique root (tree) within a document SLT with multiple roots"""

#     final_results = []
#     for old_result in initial_results:
#         doc_slt = old_result["rest_of_siblings"]
#         if isinstance(doc_slt, tuple):
#             # STL for doc has only one root.
#             final_results.extend(find_matches_among_slt_trees(query_tree, old_result))
#         else:
#             for i in range(len(doc_slt)):
#                 final_results.extend(find_matches_among_slt_trees(query_tree, {"match_style_differences": old_result["match_style_differences"], "rest_of_siblings": doc_slt[1:]}))
                
#                 if only_check_first_sibling:
#                     break
#             for 
#     return final_results


    

# def find_matches_among_compound_slts(query_slt, initial_results, break_if_exact = True):
#     """Search a query SLT with multiple roots (symbols next to one another) within a document SLT with also multiple roots"""

#     if isinstance(query_slt, tuple):
#         # SLT for query has only one root.
#         final_results = find_matches_of_slt_tree_against_compound_slts(query_slt, initial_results)

#     else:
#         current_results = find_matches_of_slt_tree_against_compound_slts(query_slt[0], initial_results)
#         if len(query_slt>1):
#             final_results = find_matches_among_compound_slts(query_slt[1:], current_results)
#         else:
#             final_results = current_results

#     return final_results

#     for q_node in query_slt:
#         current_results  = find_matches_of_slt_tree_against_compound_slts(q_node, initial_results, break_if_exact = True)
        
#         for individual_result in current_results:

#             find_matches_of_slt_tree_against_compound_slts



#         for d_node in (doc_slt if isinstance(doc_slt, list) else [doc_slt]):
#             current_results = find_stl_subtree_matches(query_slt, d_node)
#             if final_results is None:
#                 final_results = current_results
#             else:
#                 final_results = update_match_result(final_results, current_results)

#             if break_if_exact and final_results["exact_style_match"]:
#                 return final_results
            

# def find_informed_matches_among_slt_strs(query_slt_str, doc_slt_str, break_if_exact = True):
#     query_slt = str_to_slt(query_slt_str)
#     doc_slt = str_to_slt(doc_slt_str)
#     initial_results = [{
#         "match_style_differences": [],
#         "rest_of_siblings": doc_slt
#     }]
#     return find_matches_among_compound_slts(query_slt, initial_results, break_if_exact)




# def search_latex_in_mathml(llm_latex, document_mathml, break_if_exact = True):
#     # Convert LLM latex to MathML
#     query_mathml = latex_to_mathml(llm_latex)
    
#     # Parse XML and extract trees
#     try:
#         q_nodes = get_slt_from_mathml(ET.fromstring(query_mathml))
#     except:
#         return {
#             'exact_style_match': False,
#             'style_differences': []
#         }
        
#     d_nodes = get_slt_from_mathml(ET.fromstring(document_mathml))
    
#     # (Since <math> is stripped, it returns a list. Take the first math element)
#     query_tree = q_nodes[0] if isinstance(q_nodes, list) else q_nodes
    
#     # Search all nodes in the document
#     final_results = None
#     for d_node in (d_nodes if isinstance(d_nodes, list) else [d_nodes]):
#         current_results = find_informed_slt_matches(query_tree, d_node)
#         if final_results is None:
#             final_results = current_results
#         else:
#             final_results = update_match_result(final_results, current_results)

#         if break_if_exact and final_results["exact_style_match"]:
#             return final_results

#     return final_results


# # def get_math_skeleton(element, inherited_styles=None, use_styling = True):
# #     """
# #     Strips structural noise but preserves core mathematical styles (bold, bb, cal, frak)
# #     by merging XML attributes and Unicode character names.
# #     """
# #     if inherited_styles is None or not use_styling:
# #         inherited_styles = set()
        
# #     # 1. Track styles passing down through the tree (copy the set to avoid mutating parents)
# #     current_styles = set(inherited_styles)
    
# #     if use_styling:
# #         variant = element.attrib.get('mathvariant', '').lower()
# #         if 'bold' in variant: current_styles.add('bold')
# #         if 'double-struck' in variant: current_styles.add('bb') # \mathbb
# #         if 'fraktur' in variant: current_styles.add('frak')     # \mathfrak
# #         if 'script' in variant: current_styles.add('cal')       # \mathcal
# #         # Note: 'italic' is intentionally ignored to allow \mathbf{u} to match bold-italic u
    
# #     tag = element.tag.split('}')[-1]
# #     ignore_tags = {'math', 'mrow', 'mstyle', 'mpadded', 'mphantom', 'semantics'}
    
# #     # 2. Process children
# #     children_skeletons = []
# #     for child in element:
# #         child_skel = get_math_skeleton(child, current_styles, use_styling)
# #         if isinstance(child_skel, list):
# #             children_skeletons.extend(child_skel)
# #         elif child_skel:
# #             children_skeletons.append(child_skel)
            
# #     if tag in ignore_tags:
# #         return children_skeletons

# #     # 3. Process text and extract embedded Unicode styles
# #     text = element.text.strip() if element.text and element.text.strip() else ""
# #     if text:
# #         # Get base ASCII character (e.g., 𝐮 -> u)
# #         base_text = unicodedata.normalize('NFKC', text)

# #         if not use_styling:
# #             node_content = base_text
# #         else:
# #             # Check if the Unicode character itself contains style information
# #             for char in text:
# #                 name = unicodedata.name(char, "")
# #                 if name.startswith("MATHEMATICAL"):
# #                     if "BOLD" in name: current_styles.add('bold')
# #                     if "DOUBLE-STRUCK" in name: current_styles.add('bb')
# #                     if "FRAKTUR" in name: current_styles.add('frak')
# #                     if "SCRIPT" in name: current_styles.add('cal')
        
# #             style_key = "-".join(sorted(current_styles))
            
# #             # Attach the style footprint if one exists (e.g., ('u', 'bold'))
# #             node_content = (base_text, style_key) if style_key else base_text
# #     else:
# #         node_content = None
        
# #     if not children_skeletons:
# #         return (tag, node_content)
    
# #     return (tag, tuple(children_skeletons))


# # def is_subtree(query_skel, doc_skel):
# #     """Checks if query_skel exists exactly anywhere within doc_skel."""
# #     if query_skel == doc_skel:
# #         return True
    
# #     # If the document node has children, check them recursively
# #     if isinstance(doc_skel, tuple) and len(doc_skel) > 1 and isinstance(doc_skel[1], tuple):
# #         for child in doc_skel[1:]:
# #             if is_subtree(query_skel, child):
# #                 return True
# #     return False

# # def search_latex_in_mathml(latex_query: str, document_mathml: str) -> str:
# #     # 1. Convert LLM LaTeX to MathML
# #     query_mathml = latex2mathml.converter.convert(latex_query)
    
# #     # 2. Parse XML
# #     query_root = ET.fromstring(query_mathml)
# #     doc_root = ET.fromstring(document_mathml)
    
# #     # 3. Extract pure mathematical skeletons (this strips out all font/style differences)
# #     # get_math_skeleton returns a list at the root level because <math> is an ignored tag
# #     query_skel = get_math_skeleton(query_root)[0] 
# #     doc_skels = get_math_skeleton(doc_root)

# #     # 4. Search

# #     match_type = "none"

# #     for doc_skel in doc_skels:
# #         if is_subtree(query_skel, doc_skel):
# #             match_type = "formatted"

# #     query_skel_unstyled = get_math_skeleton(query_root, use_styling = False)[0] 
# #     doc_skels_unstyled = get_math_skeleton(doc_root, use_styling = False)

# #     for doc_skel in doc_skels_unstyled:
# #         if is_subtree(query_skel_unstyled, doc_skel):
# #             if match_type == "formatted":
# #                 match_type == "both"
# #             else:
# #                 match_type == "unformatted"

# #     return match_type


    