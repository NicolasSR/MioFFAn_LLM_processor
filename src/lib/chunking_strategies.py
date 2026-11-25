
import logging
import math
import statistics
from typing import TypedDict, Any

from src.lib.html_utils import divide_html_by_sections


logger = logging.getLogger(__name__)


class ChunkDict(TypedDict):
    init_html_dfs_idx: int              # DFS position of first included element with respect to GraphState's root_with_placeholders
    end_html_dfs_idx: int               # DFS position of last included element with respect to GraphState's root_with_placeholders
    text: str                           # plain text, stating formulas in latex
    inline_math_ids: list[str]          # list of ids for the inline math included in the chunks
    display_math_ids: list[str]         # list of ids for the display math included in the chunks
    misc: Any                           # Strategy-dependant extra information

def chunk_by_section(latex_html_tree, text_and_eq_per_dfs_elem):
    """Split the paper into sections with latex injected for the equations.
    """

    chunks = []

    sections, end_idx = divide_html_by_sections(latex_html_tree)

    # Get text and equation ids per element in DFS order
    for i in range(len(sections)):
        init = sections[i]["init_html_dfs_idx"]
        if i == len(sections)-1:
            end = end_idx
        else:
            end = sections[i+1]["init_html_dfs_idx"]
        chunk_elements_data = text_and_eq_per_dfs_elem[init:end]

        chunk_text = ""
        display_math_ids = []
        inline_math_ids = []
        for el_data in chunk_elements_data:
            chunk_text += el_data["text"]
            if el_data["display_math_id"] is not None:
                display_math_ids.append(el_data["display_math_id"])
            elif el_data["inline_math_id"] is not None:
                inline_math_ids.append(el_data["inline_math_id"])

        chunks.append(ChunkDict(
            init_html_dfs_idx=init,
            end_html_dfs_idx=end-1,
            text=chunk_text,
            inline_math_ids=inline_math_ids,
            display_math_ids=display_math_ids,
            misc={
                "index": sections[i]["index"],
                "label": sections[i]["label"],
                "title": sections[i]["title"]
            }
        ))

    return chunks


def chunk_by_length(text_and_eq_per_dfs_elem, window, overlap):
    """Split the paper into chunks by pre-defined window length (of non-whitespace chars)

    Inject latex strings into the placeholder tree and walk
    it in DFS/document order with a sliding window.
    
    Each chunk stops before the window would be exceeded and starts early
    enough to honour the overlap.
    """

    chunks = []

    target_overlap = math.floor(overlap*window)

    # Get statistics of length per element and comparison to chunk window
    lens_list = [len(el["text"].replace(" ","")) for el in text_and_eq_per_dfs_elem]
    
    logger.info("HTML elements' text length statistics.\n"+\
        f"  Mean: {statistics.mean(lens_list):.2f}\n"+\
        f"  Median: {statistics.median(lens_list)}\n"+\
        f"  Max: {max(lens_list)}"
        )

    # Slide a window over text content lengths
    offset = 0
    end = offset
    n = len(text_and_eq_per_dfs_elem)
    chunk_idx = 0

    while end < n:
        # Forward scan to find chunk limits
        cum = 0
        end = offset
        next_start = None
        display_math_ids = []
        inline_math_ids = []

        current_text = text_and_eq_per_dfs_elem[end]["text"]
        current_text_len = len(current_text.replace(" ",""))

        while end < n:
            current_text = text_and_eq_per_dfs_elem[end]["text"]
            current_text_len = len(current_text.replace(" ",""))

            if cum + current_text_len > window:
                break
            
            # Add any found equation to the chunk's equation ids list
            display_math_id = text_and_eq_per_dfs_elem[end]["display_math_id"]
            inline_math_id = text_and_eq_per_dfs_elem[end]["inline_math_id"]
            if display_math_id is not None:
                display_math_ids.append(display_math_id)
            elif inline_math_id is not None:
                inline_math_ids.append(inline_math_id)

            if next_start is None and cum + current_text_len >= window-target_overlap:
                # Get starting point for next chunk based on overlap
                next_start = end+1
            # Advance counters for total length count and chunk end
            cum += current_text_len
            end += 1
            
        if end == offset:
            # Even a single element exceeds the window
            # Emit one-chunk anyway and no overlap for next chunk.
            # Add display/inline math id if it applies
            display_math_id = text_and_eq_per_dfs_elem[end]["display_math_id"]
            inline_math_id = text_and_eq_per_dfs_elem[end]["inline_math_id"]
            if display_math_id is not None:
                display_math_ids.append(display_math_id)
            elif inline_math_id is not None:
                inline_math_ids.append(inline_math_id)
            logger.warning(f"Chunk {chunk_idx} surpasses window ({window}). Chunk length: {current_text_len}, Associated eq: {eq_id}.")
            end += 1

        if next_start is None:
            next_start = end

        chunk_elements_data = text_and_eq_per_dfs_elem[offset:end]

        # Build text (concat of text content) ---
        chunk_text = "".join(el["text"] for el in chunk_elements_data)

        chunks.append(ChunkDict(
            init_html_dfs_idx=offset,
            end_html_dfs_idx=end-1,
            text=chunk_text,
            display_math_ids=display_math_ids,
            inline_math_ids=inline_math_ids,
            misc=None
        ))

        offset = next_start

    return chunks