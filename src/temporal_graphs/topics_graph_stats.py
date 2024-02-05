import argparse
import multiprocessing
import os

import numpy as np
import pandas as pd
from raphtory import Graph
from raphtory import algorithms as rp
from tqdm import tqdm


def generate_stats(window, filtered_graph):
    nodes = []
    edges = []
    mean_degrees = []
    degree_centrality = []
    top_node_names = []
    time = []

    for windowed_graph in filtered_graph.rolling(window=window):
        # Get the degree of each node
        # https://docs.raphtory.com/en/v0.7.0/reference/algorithms/metrics.html#raphtory.algorithms.average_degree
        mean_degrees.append(rp.average_degree(windowed_graph))

        # Get the degree centrality of each node
        # To know how important a node representing a topic is
        # https://docs.raphtory.com/en/v0.7.0/reference/algorithms/centrality.html#centrality
        result = rp.degree_centrality(windowed_graph)

        # Get the top three nodes with the highest degree centrality
        # https://docs.raphtory.com/en/v0.7.0/reference/algorithms/algorithmresult.html#raphtory.AlgorithmResult.top_k
        nodes_degree_centrality = result.top_k(5000)
        top_node_names.append(str([n[0].name for n in nodes_degree_centrality]))
        degree_centrality.append(str([n[1] for n in nodes_degree_centrality]))

        # Get the number of nodes and edges
        nodes.append(windowed_graph.count_nodes())
        edges.append(windowed_graph.count_edges())

        # Convert the unix timestamp to a datetime object
        # time_value = windowed_graph.earliest_time
        time.append(windowed_graph.start_date_time)

    # Convert the lists to numpy arrays for the plot
    time = np.array(time)
    numb_nodes = np.array(nodes)
    numb_edges = np.array(edges)
    avg_degrees = np.array(mean_degrees)
    top_node_names = np.array(top_node_names)
    degree_centrality = np.array(degree_centrality)

    # Flatten the arrays
    top_node_names = np.array(top_node_names).flatten()
    degree_centrality = np.array(degree_centrality).flatten()

    # Save the data to a csv file
    df = pd.DataFrame(
        {
            "time": time,
            "num_nodes": numb_nodes,
            "num_edges": numb_edges,
            "avg_degree": avg_degrees,
            "top_node_names": top_node_names,
            "degree_centrality": degree_centrality,
        }
    )

    # Add the window to the dataframe
    df["window"] = window

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temp_graph_dir", type=str, default="./../../data/temporal_graph/"
    )
    parser.add_argument("--temp_graph_name", type=str, default="topics_corpus_graph")
    parser.add_argument("--plot_dir", type=str, default="./../../reports/figures/")
    parser.add_argument("--temp_graph_path", type=str, default=None)

    args = parser.parse_args()

    if not args.temp_graph_path:
        # Load the temporal graph
        args.temp_graph_path = os.path.join(args.temp_graph_dir, args.temp_graph_name)

    print(f"Loading the temporal graph from {args.temp_graph_path}...")
    loaded_graph = Graph.load_from_file(args.temp_graph_path)

    # Filter the graph to an specific time range
    # Make the filtered_graph a global variable because it cannot be pickled by multiprocessing
    # global filtered_graph
    print("Filtering the temporal graph...")
    filtered_graph = loaded_graph.after("1990-01-01")
    # filter_graph = loaded_graph.window("1996-01-01", "2019-01-01")

    # window = 2000  # 2000 seconds
    windows = ["1 year", "5 years", "10 years"]

    dfs = []
    for window in tqdm(windows):
        print(f"Calculating the stats for the window: {window}")
        df = generate_stats(window, filtered_graph)
        if not df.empty:
            print("Appending the dataframe to the list...")
            dfs.append(df)

    # Concatenate the dataframes
    print("Concatenating the dataframes...")
    df = pd.concat(dfs)

    print("Saving the dataframe with the general statistics to a csv file...")
    df.to_csv(os.path.join(args.temp_graph_dir, "topic_graphs_stats.csv"), index=False)
