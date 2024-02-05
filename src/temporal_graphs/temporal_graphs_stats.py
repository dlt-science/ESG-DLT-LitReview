import argparse
import multiprocessing
import os

import numpy as np
import pandas as pd
from raphtory import Graph
from raphtory import algorithms as rp
from tqdm import tqdm


def generate_stats(window, filtered_graph):
    largest_connected_component = []
    num_components_more_than_one_node = []
    nodes = []
    edges = []
    mean_degrees = []
    clustering_coefficient = []
    time = []

    for windowed_graph in filtered_graph.rolling(window=window):
        if windowed_graph.count_nodes() == 0:
            print("The windowed graph has no nodes. Skipping...")
            continue

        result = rp.weakly_connected_components(windowed_graph)

        # Group the components together
        components = result.group_by()

        # Get the size of each component
        component_sizes = {key: len(value) for key, value in components.items()}

        # Calculate the number of connected components with more than 1 node
        num_components = sum(1 for size in component_sizes.values() if size > 1)
        num_components_more_than_one_node.append(num_components)

        # Get the key for the largest component
        largest_component = max(component_sizes, key=component_sizes.get)
        largest_connected_component.append(component_sizes[largest_component])

        # Get the degree of each node
        # https://docs.raphtory.com/en/v0.7.0/reference/algorithms/metrics.html#raphtory.algorithms.average_degree
        mean_degrees.append(rp.average_degree(windowed_graph))

        # Get the clustering coefficient of each node
        clustering_coefficient.append(rp.global_clustering_coefficient(windowed_graph))

        # Get the number of nodes and edges
        nodes.append(windowed_graph.count_nodes())
        edges.append(windowed_graph.count_edges())

        # Convert the unix timestamp to a datetime object
        # time_value = windowed_graph.earliest_time
        time.append(windowed_graph.start_date_time)

    # Convert the lists to numpy arrays for the plot
    time = np.array(time)
    largest_connected_component = np.array(largest_connected_component)
    num_components_more_than_one_node = np.array(num_components_more_than_one_node)
    numb_nodes = np.array(nodes)
    numb_edges = np.array(edges)
    avg_degrees = np.array(mean_degrees)
    clustering_coefficient = np.array(clustering_coefficient)

    # Save the data to a csv file
    df = pd.DataFrame(
        {
            "time": time,
            "largest_connected_component": largest_connected_component,
            "num_components_more_than_one_node": num_components_more_than_one_node,
            "num_nodes": numb_nodes,
            "num_edges": numb_edges,
            "avg_degree": avg_degrees,
            "clustering_coefficient": clustering_coefficient,
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
    parser.add_argument("--temp_graph_name", type=str, default="corpus_graph")
    parser.add_argument("--plot_dir", type=str, default="./../../reports/figures/")
    parser.add_argument("--temp_graph_path", type=str, default=None)

    args = parser.parse_args()

    if not args.temp_graph_path:
        # Load the temporal graph
        args.temp_graph_path = os.path.join(args.temp_graph_dir, args.temp_graph_name)

    args.temp_graph_name = os.path.basename(args.temp_graph_path)

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

    # # Calculate 2/3 of the available CPUs
    # num_processes = int(multiprocessing.cpu_count() * 2 / 3)

    # # Create a list of tuples
    # windows_filtered_graph = [(window, filtered_graph) for window in windows]
    #
    # # Use imap to apply generate_stats to each tuple
    # with multiprocessing.Pool(num_processes) as pool:
    #     dfs = list(tqdm(pool.imap(unpack_and_generate_stats, windows_filtered_graph), total=len(windows)))

    # with multiprocessing.Pool(num_processes) as pool:
    #     # dfs = pool.map(generate_stats, windows)
    #
    #
    #     dfs = list(tqdm(pool.imap(generate_stats, windows, filtered_graph), total=len(windows)))

    # # Use imap to apply unpack_and_generate_stats to each window
    # with multiprocessing.Pool(num_processes) as pool:
    #     dfs = list(tqdm(pool.imap(unpack_and_generate_stats, windows), total=len(windows)))

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
    df.to_csv(
        os.path.join(args.temp_graph_dir, f"{args.temp_graph_name}_stats.csv"),
        index=False,
    )
