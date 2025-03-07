from queue import PriorityQueue
#Using Double-Ended Queues
def best_first_search(graph, start, goal, coordinates):
    #BestFirst
    def heuristic(city, goal):
        (x1, y1) = coordinates[city]
        (x2, y2) = coordinates[goal]
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    pq = PriorityQueue()
    pq.put((heuristic(start, goal), [start]))
    max_frontier = 1
    visited = set()
    while not pq.empty():
        if pq.qsize() > max_frontier:
            max_frontier = pq.qsize()
        _, path = pq.get()
        node = path[-1]
        if node == goal:
            return path, max_frontier
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in path:
                    new_path = path + [neighbor]
                    pq.put((heuristic(neighbor, goal), new_path))
    return None, max_frontier

