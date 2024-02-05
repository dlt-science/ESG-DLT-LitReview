import argparse
import concurrent.futures
import glob
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
from raphtory import Graph
from tqdm import tqdm

# def add_nodes(g, nodes):
#     print("Adding nodes...")
#
#     for id, timestamp, properties in tqdm(nodes):
#         # Convert the year to datetime object
#         timestamp = datetime(int(timestamp), 1, 1)
#         g.add_node(timestamp=timestamp, id=id, properties=properties)
#
#     return g
#
#
# def add_edges(g, edges):
#     print("Adding edges...")
#
#     for edge in tqdm(edges):
#         for timestamp, src, dst in edge:
#             # Convert the year to datetime object
#             timestamp = datetime(int(timestamp), 1, 1)
#             g.add_edge(timestamp=timestamp, src=src, dst=dst)
#
#     return g


def ingest_graph(g, nodes_df, edges_df):
    print("Adding nodes...")
    g = Graph.load_from_pandas(
        node_df=nodes_df,
        node_id="paperId",
        node_time="timestamp",
        node_const_props=["title"],
        edge_df=edges_df,
        edge_src="source",
        edge_dst="destination",
        edge_time="timestamp",
    )
    return g


def add_properties(g, properties):
    print("Adding properties...")

    for timestamp, id, properties in tqdm(properties):
        timestamp = datetime.strptime(str(timestamp), "%Y")
        g.add_properties(timestamp=timestamp, id=id, properties=properties)

    return g


def add_node(g, node):
    timestamp, id, properties = node
    g.add_node(timestamp=timestamp, id=id, properties=properties)


def add_edge(g, edge):
    timestamp, src, dst = edge
    g.add_edge(timestamp=timestamp, src=src, dst=dst)


# def process_json(json_file):
#     with open(json_file, "r") as f:
#         data = json.load(f)
#
#     if "year" in data.keys() and data["year"]:
#         node = [data["paperId"], data["year"], {"title": data["title"]}]
#         edges = [
#             [r["year"], data["paperId"], r["paperId"]]
#             for r in data["references"]
#             if r["year"]
#         ]
#         # properties = {
#         #     "title": data["title"],
#         # }
#     else:
#         # node, edges = None, None, None
#         node, edges = None, None
#
#     # return node, edges, properties
#     return node, edges


def process_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    if "year" in data.keys():
        if data["year"]:
            node = pd.DataFrame(
                {
                    "paperId": [data["paperId"]],
                    "timestamp": [datetime(int(data["year"]), 1, 1)],
                    "title": [data["title"]],
                }
            )

            destination = [r["paperId"] for r in data["references"] if r["year"]]
            source = [data["paperId"]] * len(destination)
            timestamps = [
                datetime(int(r["year"]), 1, 1) for r in data["references"] if r["year"]
            ]

            edges = pd.DataFrame(
                {
                    "source": source,
                    "destination": destination,
                    "timestamp": timestamps,
                }
            )
        else:
            node, edges = None, None

    return node, edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsons_dir", type=str, default="./../../data/metadata/")
    # parser.add_argument("--text_path", type=str, description="Path to the text file to do inference on")
    parser.add_argument(
        "--temporal_graph_dir", type=str, default="./../../data/temporal_graph/"
    )
    parser.add_argument("--temporal_graph_name", type=str, default="corpus_graph")

    args = parser.parse_args()

    if not os.path.exists(args.temporal_graph_dir):
        os.makedirs(args.temporal_graph_dir)

    nodes_df_path = os.path.join(args.temporal_graph_dir, "nodes.csv")
    edges_df_path = os.path.join(args.temporal_graph_dir, "edges.csv")

    if not os.path.exists(nodes_df_path) and not os.path.exists(edges_df_path):
        print("Getting the json files...")
        json_files = glob.glob(
            os.path.join(args.jsons_dir, "**/*.json"), recursive=True
        )

        # Filter out jsons with errors
        json_files = [f for f in json_files if "error" not in f]

        num_cpus = os.cpu_count()

        # Use 2/3 of the available CPUs, rounded up
        num_workers = math.ceil(num_cpus * 2 / 3)

        print("Processing the json files...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
            results = list(
                tqdm(executor.map(process_json, json_files), total=len(json_files))
            )

        # nodes = [result[0] for result in results if result[0] is not None]
        # edges = [result[1] for result in results if result[1] is not None]

        nodes_df = pd.concat([result[0] for result in results if result[0] is not None])
        edges_df = pd.concat([result[1] for result in results if result[1] is not None])

        print(f"Saving the nodes to {args.temporal_graph_dir}...")
        nodes_df.to_csv(nodes_df_path, index=False)
        print(f"Saving the edges to {args.temporal_graph_dir}...")
        edges_df.to_csv(edges_df_path, index=False)

    else:
        print(f"Loading the nodes from {args.temporal_graph_dir}...")
        nodes_df = pd.read_csv(
            nodes_df_path, parse_dates=["timestamp"], dtype={"paperId": str}
        )
        print(f"Loading the edges from {args.temporal_graph_dir}...")
        edges_df = pd.read_csv(
            edges_df_path,
            parse_dates=["timestamp"],
            dtype={"source": str, "destination": str},
        )

        # Convert the timestamp column to datetime objects
        nodes_df["timestamp"] = pd.to_datetime(
            nodes_df["timestamp"], errors="coerce"
        ).dt.tz_localize("UTC")

        # Errors coerce to NaT because of old dates 1677-09-21 being the minimum date of pandas datetime to handle
        edges_df["timestamp"] = pd.to_datetime(
            edges_df["timestamp"], errors="coerce"
        ).dt.tz_localize("UTC")

        # Remove NaT values
        nodes_df = nodes_df.dropna(subset=["timestamp"])
        edges_df = edges_df.dropna(subset=["timestamp"])

        nodes_df["timestamp"] = pd.to_datetime(nodes_df["timestamp"]).astype(
            "datetime64[ms, UTC]"
        )
        edges_df["timestamp"] = pd.to_datetime(edges_df["timestamp"]).astype(
            "datetime64[ms, UTC]"
        )

        # Convert 'source' and 'destination' columns to string
        edges_df["source"] = edges_df["source"].astype(str)
        edges_df["destination"] = edges_df["destination"].astype(str)

    print("Creating the graph...")
    g = Graph()

    print("Adding nodes, edges and properties...")
    # g = add_nodes(g, nodes)
    # g = add_edges(g, edges)
    g = ingest_graph(g, nodes_df, edges_df)

    print(f"Generated graph: {g}")

    print("Saving the graph...")
    temp_graph_path = os.path.join(args.temporal_graph_dir, args.temporal_graph_name)
    g.save_to_file(temp_graph_path)
