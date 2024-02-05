import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from raphtory import Graph
from raphtory import algorithms as rp
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temp_graph_dir", type=str, default="./../../data/temporal_graph/"
    )
    parser.add_argument(
        "--temp_graph_stats_csv", type=str, default="temporal_graphs_stats.csv"
    )
    parser.add_argument("--plot_dir", type=str, default="./../../reports/figures/")
    parser.add_argument("--plot_name", type=str, default="connected_components.pdf")

    args = parser.parse_args()

    # Load the temporal graph
    temp_graph_path = os.path.join(args.temp_graph_dir, args.temp_graph_stats_csv)

    # Save the data to a csv file
    df = pd.read_csv(temp_graph_path)

    # Filter only for 1 year window
    df = df[df["window"] == "1 year"]

    # Keep only the values from 1990 onwards
    df["time"] = pd.to_datetime(df["time"])

    # Keep only the values for the year 1990 onwards (inclusive)
    df = df[df["time"] >= pd.to_datetime("1990-01-01")]

    # Keep all the values before 2020 (inclusive)
    df = df[df["time"] <= pd.to_datetime("2016-12-31")]

    # Sort by timestamp column
    df = df.sort_values(by="time")

    COLOR_LCC = "#69b3a2"
    COLOR_CC = "#3399e6"

    # Set the style of the plot
    sns.set_style("white")
    sns.set_context("notebook")
    # sns.set_context("talk")

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()  # Create a second y-axis that shares the same x-axis

    ax1.plot(
        df["time"],
        df["num_components_more_than_one_node"],
        color=COLOR_CC,
        label="Number of Connected Components with > 1 node",
    )
    ax2.plot(
        df["time"],
        df["largest_connected_component"],
        color=COLOR_LCC,
        linestyle="--",
        label="Size of the largest connected component (in thousands)",
    )

    # sns.lineplot(x=time, y=num_components_more_than_one_node, color=COLOR_CC, ax=ax1)
    # sns.lineplot(x=time, y=largest_connected_component,
    #              color=COLOR_LCC, linestyle='--', ax=ax2)

    ax1.set_xlabel("Year")
    # ax2.set_xlabel('Year')
    ax1.set_ylabel("Number of Connected Components \n> 1 node", color=COLOR_CC)
    # ax2.set_ylabel(
    #     "Size of the largest \nconnected component (in thousands)", color=COLOR_LCC
    # )

    ax2.set_ylabel("Size of the largest \nconnected component", color=COLOR_LCC)

    # Set y-axis to log scale
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    # Set the limits of the y-axes
    # ax1.set_ylim([0, max(df["largest_connected_component"])])
    # ax2.set_ylim([0, max(df["num_components_more_than_one_node"])])

    # Set the labels as bold
    # ax1.xaxis.label.set_weight('bold')
    # ax1.yaxis.label.set_weight('bold')
    # ax2.yaxis.label.set_weight('bold')

    # Remove the legend
    plt.legend([], [], frameon=False)

    # Adjust the size of the y-axis labels
    # ax1.tick_params(axis='y', labelsize='small')
    # ax2.tick_params(axis='y', labelsize='small')

    plt.tight_layout()
    plot_path = os.path.join(args.plot_dir, args.plot_name)
    plt.savefig(plot_path, dpi=300)
