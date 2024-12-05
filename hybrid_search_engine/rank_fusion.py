from collections import defaultdict

def reciprocal_rank_fusion(*list_of_list_ranks_system, k: int = 60):
    """
    Fuse rank from multiple IR systems using Reciprocal Rank Fusion.

    Args:
    * list_of_list_ranks_system: Ranked results from different IR system.
    K (int): A constant used in the RRF formula (default is 60).

    Returns:
    Tuple of list of sorted documents by score and sorted documents
    """
    # Dictionary to store RRF mapping
    rrf_map = defaultdict(float)

    # Calculate RRF score for each result in each list
    for rank_list in list_of_list_ranks_system:
        for rank, item in enumerate(rank_list, 1):
            rrf_map[item] += 1 / (rank + k)

    # Sort items based on their RRF scores in descending order
    sorted_items = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)

    # to dict
    sorted_items_dicts = []
    for item, score in sorted_items:
        sorted_items_dicts.append({"doc": item, "score": score})

    # Return tuple of list of sorted documents by score and sorted documents
    return sorted_items_dicts, [item for item, score in sorted_items]


if __name__ == "__main__":
    # Example ranked lists from different sources
    ir_system_a = ['Document1', 'Document3', 'Document5', 'Document7']
    ir_system_b = ['Document2', 'Document1', 'Document4']
    ir_system_c = ['Document5', 'Document3', 'Document2']

    # Combine the lists using RRF
    combined_list = reciprocal_rank_fusion(ir_system_a, ir_system_b, ir_system_c)
    print(combined_list)
