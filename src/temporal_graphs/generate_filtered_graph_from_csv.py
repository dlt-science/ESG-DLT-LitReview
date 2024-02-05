import argparse
import os

import pandas as pd
from raphtory import Graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temporal_graph_dir", type=str, default="./../../data/temporal_graph/"
    )
    parser.add_argument("--csv_name", type=str, default="filtered_network.csv")
    parser.add_argument("--temporal_graph_name", type=str, default="filtered_graph")

    args = parser.parse_args()

    if not os.path.exists(args.temporal_graph_dir):
        os.makedirs(args.temporal_graph_dir)

    print("Loading the csv file...")
    filtered_network_path = os.path.join(args.temporal_graph_dir, args.csv_name)
    df = pd.read_csv(filtered_network_path)

    # Convert the time column to datetime
    df["source_year"] = pd.to_datetime(df["source_year"], utc=True).astype(
        "datetime64[ms, UTC]"
    )

    df["destination_year"] = pd.to_datetime(df["destination_year"], utc=True).astype(
        "datetime64[ms, UTC]"
    )

    # Generate the nodes dataframe
    nodes_df = df[["source_title", "source_year"]].copy()

    # Drop duplicates
    nodes_df.drop_duplicates(inplace=True)

    print("Creating the graph...")
    g = Graph()

    print("Adding nodes, edges and properties...")
    g = Graph.load_from_pandas(
        node_df=nodes_df,
        node_id="source_title",
        node_time="source_year",
        # node_const_props=["title"],
        edge_df=df,
        edge_src="source_title",
        edge_dst="destination_title",
        edge_time="source_year",
        # edge_const_props=['destination_year', 'source_level']
    )

    print(f"Generated graph: {g}")

    print("Saving the graph...")
    temp_graph_path = os.path.join(args.temporal_graph_dir, args.temporal_graph_name)
    g.save_to_file(temp_graph_path)
