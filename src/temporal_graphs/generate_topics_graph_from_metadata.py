import argparse
import concurrent.futures
import glob
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
from raphtory import Graph
from tqdm import tqdm


def ingest_graph(g, nodes_df, edges_df):
    print("Adding nodes...")
    g = Graph.load_from_pandas(
        node_df=nodes_df,
        node_id="topic",
        node_time="timestamp",
        # node_const_props=["title"],
        edge_df=edges_df,
        edge_src="source",
        edge_dst="destination",
        edge_time="timestamp",
        edge_layer="in_same_paper",
    )
    return g


def process_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    if "year" in data.keys():
        if data["year"]:
            if data["topics"]:
                edges_dfs = []
                nodes_dfs = []

                authorIds = [a["authorId"] for a in data["authors"]]
                authorNames = [a["name"] for a in data["authors"]]

                for source_topic in data["topics"]:
                    destination_topics = [
                        t["topic"] for t in data["topics"] if t != source_topic
                    ]

                    node = pd.DataFrame(
                        {
                            "topic": [source_topic["topic"]],
                            "timestamp": [datetime(int(data["year"]), 1, 1)],
                            "title": [data["title"]],
                            "author_ids": [authorIds],
                            "author_names": [authorNames],
                        }
                    )

                    # Send an empty dataframe if there are no destination topics
                    edges = pd.DataFrame()
                    if destination_topics:
                        edges = pd.DataFrame(
                            {
                                "timestamp": [datetime(int(data["year"]), 1, 1)]
                                * len(destination_topics),
                                "source": [source_topic["topic"]]
                                * len(destination_topics),
                                "destination": destination_topics,
                            }
                        )

                    edges_dfs.append(edges)
                    nodes_dfs.append(node)

                edges_df = pd.concat(edges_dfs)
                node_df = pd.concat(nodes_dfs)

                return node_df, edges_df

    return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsons_dir", type=str, default="./../../data/metadata/")
    # parser.add_argument("--text_path", type=str, description="Path to the text file to do inference on")
    parser.add_argument(
        "--temporal_graph_dir", type=str, default="./../../data/temporal_graph/"
    )
    parser.add_argument(
        "--temporal_graph_name", type=str, default="topics_corpus_graph"
    )

    args = parser.parse_args()

    print("Getting the json files...")
    json_files = glob.glob(os.path.join(args.jsons_dir, "**/*.json"), recursive=True)

    # Filter out jsons with errors
    json_files = [f for f in json_files if "error" not in f]

    num_cpus = os.cpu_count()

    # Use 2/3 of the available CPUs, rounded up
    num_workers = math.ceil(num_cpus * 2 / 3)

    if not os.path.exists(args.temporal_graph_dir):
        os.makedirs(args.temporal_graph_dir)

    nodes_df_path = os.path.join(args.temporal_graph_dir, "nodes_topics.csv")
    edges_df_path = os.path.join(args.temporal_graph_dir, "edges_topics.csv")

    if not os.path.exists(nodes_df_path) and not os.path.exists(edges_df_path):
        print("Processing the json files...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
            results = list(
                tqdm(executor.map(process_json, json_files), total=len(json_files))
            )

        nodes_df = pd.concat([result[0] for result in results if result[0] is not None])
        edges_df = pd.concat([result[1] for result in results if result[1] is not None])

        # Edges layer name
        edges_df["in_same_paper"] = "in_same_paper"

        # Remove NaT values
        nodes_df = nodes_df.dropna(subset=["timestamp"])
        edges_df = edges_df.dropna(subset=["timestamp"])

        nodes_df["timestamp"] = pd.to_datetime(nodes_df["timestamp"], utc=True).astype(
            "datetime64[ms, UTC]"
        )
        edges_df["timestamp"] = pd.to_datetime(edges_df["timestamp"], utc=True).astype(
            "datetime64[ms, UTC]"
        )

        print(f"Saving the nodes to {args.temporal_graph_dir}...")
        nodes_df.to_csv(nodes_df_path, index=False)
        print(f"Saving the edges to {args.temporal_graph_dir}...")
        edges_df.to_csv(edges_df_path, index=False)

    else:
        print(f"Loading the nodes from {args.temporal_graph_dir}...")
        nodes_df = pd.read_csv(
            nodes_df_path,
            parse_dates=["timestamp"],
            dtype={"topic": str, "title": str},
        )
        print(f"Loading the edges from {args.temporal_graph_dir}...")
        edges_df = pd.read_csv(
            edges_df_path,
            parse_dates=["timestamp"],
            dtype={"source": str, "destination": str},
        )

        # # Convert the timestamp column to datetime objects
        # nodes_df["timestamp"] = pd.to_datetime(
        #     nodes_df["timestamp"], errors="coerce"
        # ).dt.tz_localize("UTC")
        #
        # # Errors coerce to NaT because of old dates 1677-09-21 being the minimum date of pandas datetime to handle
        # edges_df["timestamp"] = pd.to_datetime(
        #     edges_df["timestamp"], errors="coerce"
        # ).dt.tz_localize("UTC")

        nodes_df["timestamp"] = pd.to_datetime(
            nodes_df["timestamp"], errors="coerce", utc=True
        ).astype("datetime64[ms, UTC]")
        edges_df["timestamp"] = pd.to_datetime(
            edges_df["timestamp"], errors="coerce", utc=True
        ).astype("datetime64[ms, UTC]")

        # Remove NaT values
        nodes_df = nodes_df.dropna(subset=["timestamp"])
        edges_df = edges_df.dropna(subset=["timestamp"])

        # Convert 'source' and 'destination' columns to string
        edges_df["source"] = edges_df["source"].astype(str)
        edges_df["destination"] = edges_df["destination"].astype(str)

        # Edges layer name
        edges_df["in_same_paper"] = "in_same_paper"

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
