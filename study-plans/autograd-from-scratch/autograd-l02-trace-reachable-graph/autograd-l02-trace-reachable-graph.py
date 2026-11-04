def trace_reachable_graph(nodes: list, output_id: str) -> tuple:
    """
    Returns a tuple of reachable ID and parent-to-child edge lists.
    """
    graph = {node['id']:node['parents'] for node in nodes}
    pairs = []
    visited = []
    stack = []
    stack.append(output_id)
    while stack:
        node = stack.pop()
        if node not in visited:
           visited.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
               visited.append(neighbor)
               pairs.append([neighbor, node])
               stack.append(neighbor)

    ids = [node["id"] for node in nodes if node["id"] in visited]

    edges = []
    for node in nodes:
        if node["id"] in visited:
            for parent in node["parents"]:
                edges.append([parent, node["id"]])
    
    return (ids, edges)
