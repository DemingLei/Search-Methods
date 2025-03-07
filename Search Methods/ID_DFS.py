def iddfs_search(graph, start, goal, max_depth=50):
    #ID-DFS
    def dls(path, depth):
        node = path[-1]
        if node == goal:
            return path
        if depth == 0:
            return None
        for neighbor in graph[node]:
            if neighbor not in path:
                new_path = path + [neighbor]
                result = dls(new_path, depth - 1)
                if result is not None:
                    return result
        return None

    for depth in range(max_depth):
        result = dls([start], depth)
        if result is not None:
            return result, 0
    return None, 0


