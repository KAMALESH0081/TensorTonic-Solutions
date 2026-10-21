def build_expression_graph(leaves: list, operations: list) -> tuple:
    """
    Returns a tuple of the node list and final node ID.
    """
    current_node = None
    leaf_map = {}
    for leaf in leaves:
        leaf_map[leaf['id']] = {'id':leaf['id'], 'data': float(leaf['data']), 'grad': 0.0, 'op':"", 'parents':[]}
        current_node = leaf['id']

    temp = {}
    for operation in operations:
        temp['id'] = operation['id']
        current_node = temp['id']
        if operation['op'] == '+':
            temp['data'] = float(leaf_map[operation['left']]['data'] + leaf_map[operation['right']]['data'])
        else:
            temp['data'] = float(leaf_map[operation['left']]['data'] * leaf_map[operation['right']]['data'])

        temp['grad'] = 0.0
        temp['op'] = operation['op']
        temp['parents'] = [operation['left'], operation['right']]
        leaf_map[temp['id']] = temp
        temp = {}
    graph = list(leaf_map.values())
    final_node = current_node
    return (graph, final_node)
        
        
