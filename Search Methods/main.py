import time
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
np.alltrue = np.all


# Import each search algorithm
import BFS
import DFS
import ID_DFS
import best_first
import A_Star


def read_graph(adjacency_file):
    # Reading TXT Files to Construct Directionless Graphs
    graph = {}
    with open(adjacency_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            city1, city2 = parts[0], parts[1]
            if city1 not in graph:
                graph[city1] = []
            if city2 not in graph:
                graph[city2] = []
            # Bi-directionality
            if city2 not in graph[city1]:
                graph[city1].append(city2)
            if city1 not in graph[city2]:
                graph[city2].append(city1)
    return graph


def read_coordinates(coordinates_file):
    #Read coordinates.csv, return dictionary
    coords = {}
    df = pd.read_csv(coordinates_file)
    for index, row in df.iterrows():
        city = row['City']
        lat = float(row['Latitude'])
        lon = float(row['Longitude'])
        coords[city] = (lat, lon)
    return coords


def compute_total_distance(path, coordinates):
    #Calculate the cumulative Euclidean distance between neighboring cities in the pathway
    total = 0.0
    for i in range(len(path) - 1):
        c1, c2 = path[i], path[i + 1]
        (lat1, lon1) = coordinates[c1]
        (lat2, lon2) = coordinates[c2]
        dist = math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)
        total += dist
    return total


def plot_graph_full(graph, coordinates, path=None):
    #drawings. X: Longitude; Y: Latitude
    #Build matplot
    G = nx.Graph()
    for city, neighbors in graph.items():
        G.add_node(city)
        for nb in neighbors:
            G.add_edge(city, nb)

    # Generate node coordinates
    pos = {}
    for city in G.nodes():
        if city in coordinates:
            lat, lon = coordinates[city]
            pos[city] = (lon, lat)
        else:
            # If a city is missing coordinate information
            pos[city] = (0, 0)


    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1)
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color='blue')
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Highlight Path
    if path and len(path) > 1:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)


    # Setting up axis labels, grids
    ax = plt.gca()
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Graph of Cities and Edges")
    ax.grid(True)


    plt.show()


def main():
    # Reading plot and coordinate data
    graph = read_graph("Adjacencies.txt")
    coordinates = read_coordinates("coordinates.csv")

    # Show all cities
    print("Cities in the database:")
    for city in sorted(graph.keys()):
        print(city)

    # Enter the start and end points
    start = input("\nPlease select a starting city: ").strip()
    goal = input("Please select the end city: ").strip()
    if start not in graph or goal not in graph:
        print("The start or end city is not in the database, please check the input.")
        return

    # Selecting a search algorithm
    print("\nSelecting a search algorithm:")
    print("1. BFS")
    print("2. DFS")
    print("3. ID-DFS")
    print("4. Best-First")
    print("5. AStar")
    choice = input("Please enter your choice (1-5): ").strip()

    # Search Algorithm Mapping
    algorithms = {
        '1': ("BFS", BFS.bfs_search),
        '2': ("DFS", DFS.dfs_search),
        '3': ("IDDFS", ID_DFS.iddfs_search),
        '4': ("BestFirst", best_first.best_first_search),
        '5': ("AStar", A_Star.astar_search)
    }
    selected = algorithms.get(choice)
    if not selected:
        print("Ineligible Choice")
        return

    algo_name, algo_func = selected
    print(f"\n Using {algo_name} search the path form{start} to {goal}")

    # Timer
    t0 = time.perf_counter()

    path = None
    mem_usage = 0
    cost = 0.0

    if algo_name == "AStar":
        path, mem_usage, cost = algo_func(graph, start, goal, coordinates)
    elif algo_name == "BestFirst":
        path, mem_usage = algo_func(graph, start, goal, coordinates)
    else:
        path, mem_usage = algo_func(graph, start, goal)

    t1 = time.perf_counter()
    elapsed = t1 - t0

    if not path:
        print("No viable path has been identified")
        return

    # Calculate the total Euclidean distance
    if algo_name == "AStar":
        total_distance = cost
    else:
        total_distance = compute_total_distance(path, coordinates)

    print(f"\n{algo_name} Search results:")
    print(f"  Path: {path}")
    print(f"  Total path distance (Euclidean): {total_distance:.4f}")
    print(f"  Memory usage: {mem_usage}")
    print(f"  Spend time:{elapsed:.6f} seconds")


    plot_graph_full(graph, coordinates, path)

    #Run the rest of the algorithms for comparison
    all_results = []
    #Save the current algorithm results
    all_results.append({
        "Algorithm": algo_name,
        "Time": elapsed,
        "Distance": total_distance,
        "Memory": mem_usage,
        "Path": path
    })

    #Run the remaining algorithms in order
    for key, (name, func) in algorithms.items():
        if name == algo_name:
            continue
        print(f"\nRuning {name} Algorithm")
        st = time.perf_counter()
        if name == "AStar":
            p, m, c = func(graph, start, goal, coordinates)
            dist_ = c if p else None
        elif name == "BestFirst":
            p, m = func(graph, start, goal, coordinates)
            dist_ = compute_total_distance(p, coordinates) if p else None
        else:
            p, m = func(graph, start, goal)
            dist_ = compute_total_distance(p, coordinates) if p else None
        et = time.perf_counter()

        all_results.append({
            "Algorithm": name,
            "Time": et - st,
            "Distance": dist_,
            "Memory": m,
            "Path": p if p else [],
        })

    #Print Comparison Form
    try:
        df = pd.DataFrame(all_results)
        print("\n-----Algorithm Comparison Table-----")
        print(df.to_string())

    except ImportError:
        print("\n-----Algorithm Comparison Table-----")
        for r in all_results:
            print(df.to_string())



if __name__ == '__main__':
    main()
