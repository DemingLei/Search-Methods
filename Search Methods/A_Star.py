from queue import PriorityQueue

def astar_search(graph, start, goal, coordinates):
    #AStar
    def heuristic(city, goal):
        (x1, y1) = coordinates[city]
        (x2, y2) = coordinates[goal]
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    def compute_distance(city1, city2):
        (x1, y1) = coordinates[city1]
        (x2, y2) = coordinates[city2]
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    pq = PriorityQueue()
    #Format of elements in priority queue: (f, g, path)
    pq.put((heuristic(start, goal), 0, [start]))
    max_frontier = 1
    visited = {}
    while not pq.empty():
        if pq.qsize() > max_frontier:
            max_frontier = pq.qsize()
        f, g, path = pq.get()
        node = path[-1]
        if node == goal:
            return path, max_frontier, g
        if node in visited and visited[node] <= g:
            continue
        visited[node] = g
        for neighbor in graph[node]:
            if neighbor not in path:
                cost = compute_distance(node, neighbor)
                new_g = g + cost
                new_path = path + [neighbor]
                pq.put((new_g + heuristic(neighbor, goal), new_g, new_path))
    return None, max_frontier, 0


