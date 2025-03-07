def dfs_search(graph, start, goal):
    #DFS
    stack = [[start]]
    visited = set()
    max_frontier = 1
    while stack:
        if len(stack) > max_frontier:
            max_frontier = len(stack)
        path = stack.pop()
        node = path[-1]
        if node == goal:
            return path, max_frontier
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in path:
                    new_path = list(path)
                    new_path.append(neighbor)
                    stack.append(new_path)
    return None, max_frontier

