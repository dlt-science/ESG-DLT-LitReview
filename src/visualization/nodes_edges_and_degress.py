import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def generate_plot(
    df, y_col_name, y_label_name, plot_dir, x_col_name="time", hue_col_name="window"
):
    # Set the style of the plot
    sns.set_style("white")
    sns.set_context("notebook")

    plt.figure()

    sns.lineplot(
        x=x_col_name,
        y=y_col_name,
        hue=hue_col_name,
        data=df,
        # palette="flare",
        # hue_norm=mpl.colors.LogNorm(),
        # Style with the hue variable is only for numeric data
        style=hue_col_name,
        markers=False,
        dashes=False,
        linewidth=2.5,
        alpha=1,
        sort=True,
    )

    # Set y-axis to log scale
    plt.yscale("log")

    plt.ylabel(y_label_name, fontsize=14)
    plt.xlabel("Year", fontsize=14)

    sns.despine()

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(
        bbox_to_anchor=(0.5, 1.10),
        loc="upper center",
        ncol=3,
        fontsize=11,
        frameon=False,
    )

    plt.tight_layout()

    plot_path = os.path.join(plot_dir, f"{y_col_name.replace(' ', '')}.pdf")

    plt.savefig(plot_path, dpi=300, bbox_inches="tight", transparent=True, format="pdf")


def growth_rate_for_date_range(
    df, start_date, end_date, window_size, col_name="num_nodes"
):
    # Calculate the growth rate for the given date range
    # Get the first and last value for the date range
    df = df[df["window"] == window_size]
    df = df[df["time"] >= start_date]
    df = df[df["time"] <= end_date]

    # Calculate the growth rate
    growth_rate = (max(df[col_name]) - min(df[col_name])) / min(df[col_name]) * 100

    return growth_rate


def calculate_median_for_date_range(
    df, start_date, end_date, window_size, col_name="num_nodes"
):
    # Calculate the growth rate for the given date range
    # Get the first and last value for the date range
    df = df[df["window"] == window_size]
    df = df[df["time"] >= start_date]
    df = df[df["time"] <= end_date]

    # Calculate the growth rate
    median_value = df[col_name].median()

    return median_value


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
    #
    # Keep all the values before 2020 (inclusive)
    df = df[df["time"] <= pd.to_datetime("2018-12-31")]

    # Sort by timestamp column
    df = df.sort_values(by="time")

    for stat_col in ["num_nodes", "num_edges", "avg_degree"]:
        # Growth rate for the 2008-2012 period
        growth_rate = growth_rate_for_date_range(
            df, pd.to_datetime("2008-01-01"), pd.to_datetime("2011-12-31"), "1 year"
        )
        print(f"Growth rate for {stat_col} in the 2008-2012 period: {growth_rate}")

        # Median value for the 2008-2012 period
        median_value = calculate_median_for_date_range(
            df, pd.to_datetime("2008-01-01"), pd.to_datetime("2011-12-31"), "1 year"
        )
        print(f"Median value for {stat_col} in the 2008-2012 period: {median_value}")

        # Growth rate for the 2012-max time period
        growth_rate = growth_rate_for_date_range(
            df, pd.to_datetime("2012-01-01"), max(df["time"]), "1 year"
        )
        print(f"Growth rate for {stat_col} in the 2012-max time period: {growth_rate}")

        # Median value for the 2012-max time period
        median_value = calculate_median_for_date_range(
            df, pd.to_datetime("2012-01-01"), max(df["time"]), "1 year"
        )

    y_col_names = [
        "num_nodes",
        "num_edges",
        "avg_degree",
        # "num_components_more_than_one_node",
        # "largest_connected_component"
    ]
    y_col_legend_names = [
        "Number of nodes",
        "Number of edges",
        "Average degree",
        # "Number of connected components with > 1 node",
        # "Size of the largest connected component (in thousands)"
    ]

    for y_col_name, y_label_name in tqdm(zip(y_col_names, y_col_legend_names)):
        generate_plot(df, y_col_name, y_label_name, args.plot_dir)
