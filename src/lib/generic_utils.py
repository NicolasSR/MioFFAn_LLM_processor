from collections import defaultdict

def get_hasse_diagram_for_lists_content(named_lists):
    # 1. Convert to sets and sort from smallest size to largest.
    # Sorting guarantees that any valid parent appears BEFORE its children,
    # and ties (identical sets) get chained cleanly in order.
    items = [(name, frozenset(lst)) for name, lst in named_lists.items()]
    items.sort(key=lambda x: len(x[1]))

    children = defaultdict(list)
    has_parent = set()

    for i, (child_name, child_set) in enumerate(items):
        # Look backwards through smaller (or equal) sets
        candidates = []
        for j in range(i - 1, -1, -1):
            cand_name, cand_set = items[j]
            if cand_set.issubset(child_set):
                candidates.append((cand_name, cand_set))

        # 2. Transitive Reduction: Keep only DIRECT parents.
        # Filter out any candidate that is already a subset of another candidate.
        direct_parents = []
        for c_name, c_set in candidates:
            is_ancestor = any(
                c_set.issubset(other_set) and c_name != other_name
                for other_name, other_set in candidates
            )
            if not is_ancestor:
                direct_parents.append(c_name)

        for p_name in direct_parents:
            children[p_name].append(child_name)
            has_parent.add(child_name)

    # Nodes with no parents are the roots of the tree/forest
    roots = [name for name, _ in items if name not in has_parent]

    return {
        "roots": roots,
        "children_dict": dict(children)
    }