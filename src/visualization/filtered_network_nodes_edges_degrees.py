import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temp_graph_dir", type=str, default="./../../data/temporal_graph/"
    )
    parser.add_argument(
        "--temp_graph_stats_csv", type=str, default="filtered_graph_stats.csv"
    )
    parser.add_argument("--plot_dir", type=str, default="./../../reports/figures/")

    args = parser.parse_args()

    # Load the temporal graph
    temp_graph_path = os.path.join(args.temp_graph_dir, args.temp_graph_stats_csv)

    # Save the data to a csv file
    df = pd.read_csv(temp_graph_path)

    # Keep only the values from 1990 onwards
    df["time"] = pd.to_datetime(df["time"])

    # # Keep only the values for the year 1990 onwards (inclusive)
    # df = df[df["time"] >= pd.to_datetime("1990-01-01")]
    # Keep all the values before 2020 (inclusive)
    df = df[df["time"] <= pd.to_datetime("2018-12-31")]

    # Sort by timestamp column
    df = df.sort_values(by="time")

    # Filter only for 1 year window
    df = df[df["window"] == "1 year"]

    # List of columns to reshape
    cols_to_reshape = ["num_nodes", "num_edges", "avg_degree"]

    # Reshape the dataframe
    df_reshaped = df.melt(
        id_vars=["time"],
        value_vars=cols_to_reshape,
        var_name="category",
        value_name="counts",
    )

    # Rename the values in the category column
    df_reshaped["category"] = (
        df_reshaped["category"]
        .replace("num_nodes", "Publications")
        .replace("num_edges", "Citations")
        .replace("avg_degree", "Average citations per publication")
    )

    plt.figure()

    # Set the style of the plot
    sns.set_style("white")
    sns.set_context("notebook")
    # sns.set_context("talk")

    sns.lineplot(
        x="time",
        y="counts",
        hue="category",
        data=df_reshaped,
        # palette="flare",
        # hue_norm=mpl.colors.LogNorm(),
        # Style with the hue variable is only for numeric data
        style="category",
        markers=False,
        dashes=False,
        linewidth=5,
        alpha=1,
        sort=True,
    )

    # Set y-axis to log scale
    plt.yscale("log")

    plt.ylabel("Counts", fontsize=18)
    plt.xlabel("Year", fontsize=18)

    sns.despine()

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(
        # bbox_to_anchor=(0.5, 1.15),
        # loc="upper center",
        loc="best",
        ncol=1,
        fontsize=16,
        frameon=False,
        framealpha=0.0,
    )

    plt.tight_layout()

    plot_path = os.path.join(args.plot_dir, "filtered_graph_publication_stats.pdf")

    plt.savefig(plot_path, dpi=300, bbox_inches="tight", transparent=True, format="pdf")

    plt.show()
