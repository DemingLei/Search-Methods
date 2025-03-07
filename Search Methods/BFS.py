from collections import deque
#Using Double-Ended Queues
def bfs_search(graph, start, goal):
    #Breadth First Search Algorithm
    queue = deque([[start]])
    visited = set()
    max_frontier = 1
    while queue:
        # Update the maximum value of the number of nodes in the queue
        if len(queue) > max_frontier:
            max_frontier = len(queue)
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path, max_frontier
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                #Avoiding circuits
                if neighbor not in path:
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
    return None, max_frontier


