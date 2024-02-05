import argparse
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn.objects as so


def stacked_area_chart(data, x, y, color, **kwargs):
    # Set the style of the plot
    sns.set_style("white")
    sns.set_context("notebook")

    p = so.Plot(
        data=df_topic_interest, x="time", y="degree_centrality", color="top_node_names"
    ).add(so.Area(), so.Stack())
    plt.tight_layout()
    plot_path = os.path.join(args.plot_dir, "topics_growth_stacked.pdf")

    p.save(loc=plot_path, dpi=300, bbox_inches="tight", transparent=True, format="pdf")

    # Show the plot
    p.show()


def percentage_growth_rate(df, topic):
    """
    Calculates the percentage growth rate for a given topic from the minimum to the maximum year.
    """

    df_topic = df[df["top_node_names"] == topic]

    # Get the minimum and maximum year
    min_year = df_topic["time"].min()
    max_year = df_topic["time"].max()

    # Get the degree centrality for the minimum and maximum year
    min_degree_centrality = df_topic[df_topic["time"] == min_year][
        "degree_centrality"
    ].values[0]
    max_degree_centrality = df_topic[df_topic["time"] == max_year][
        "degree_centrality"
    ].values[0]

    # Calculate the percentage growth rate
    percentage_growth_rate = (
        (max_degree_centrality - min_degree_centrality) / min_degree_centrality
    ) * 100

    df_growth_rate = pd.DataFrame(
        {
            "topic": [topic],
            "min_year": [min_year.year],
            "max_year": [max_year.year],
            "min_degree_centrality": [min_degree_centrality],
            "max_degree_centrality": [max_degree_centrality],
            "percentage_growth_rate": [percentage_growth_rate],
        }
    )

    return df_growth_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temp_graph_dir", type=str, default="./../../data/temporal_graph/"
    )
    parser.add_argument("--temp_graph_name", type=str, default="topic_graphs_stats.csv")
    parser.add_argument("--plot_dir", type=str, default="./../../reports/figures/")

    args = parser.parse_args()

    # Load the temporal graph
    temp_graph_path = os.path.join(args.temp_graph_dir, args.temp_graph_name)

    # Save the data to a csv file
    df = pd.read_csv(temp_graph_path)

    # Keep only the values from 1990 onwards
    df["time"] = pd.to_datetime(df["time"])

    # # Keep only the values for the year 1990 onwards (inclusive)
    # df = df[df["time"] >= pd.to_datetime("1990-01-01")]
    #
    # Keep all the values before 2020 (inclusive)
    df = df[df["time"] <= pd.to_datetime("2019-12-31")]

    # Sort by timestamp column
    df = df.sort_values(by="time")

    # Expand the columns with lists
    corrected = []
    for index, row in df.iterrows():
        top_node_names = row["top_node_names"].split(",")
        degree_centrality = row["degree_centrality"].split(",")

        for top_node_name, degree_centrality in zip(top_node_names, degree_centrality):
            top_node_name = (
                top_node_name.replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .replace("\\", "")
                .strip()
                .capitalize()
            )
            degree_centrality = float(
                degree_centrality.replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .replace("\\", "")
                .strip()
            )
            corrected.append(
                [row["time"], row["window"], top_node_name, degree_centrality]
            )

    df_expanded = pd.DataFrame(
        corrected, columns=["time", "window", "top_node_names", "degree_centrality"]
    )

    # Filter for a 1 year window
    df_expanded = df_expanded[df_expanded["window"] == "1 year"]

    # Sort by timestamp column and then by degree centrality
    df_expanded = df_expanded.sort_values(
        by=["time", "degree_centrality"], ascending=[True, False]
    )

    y_col_names = [
        "top_node_names",
        "degree_centrality",
    ]
    y_col_legend_names = [
        "Top node names",
        "Degree centrality",
    ]

    # List of topics of interest
    # topics_interest = [
    #     "Bitcoin",
    #     "Ethereum",
    #     "Smart contract",
    #     "Proof-of-stake",
    #     "Proof-of-work system",
    #     "Scalability",
    #     "Cryptography",
    #     "Public-key cryptography",
    #     "Money",
    # ]

    # List of topics of interest
    topics_interest = [
        "Bitcoin",
        "Ethereum",
        "Smart contract",
        "Proof-of-stake",
        "Proof-of-work system",
        "Cryptography",
        # "Cryptocurrency",
        "Peer-to-peer",
        "Distributed computing",
    ]

    # Create a lineplot for each topic of interest
    df_topic_interest = df_expanded[df_expanded["top_node_names"].isin(topics_interest)]

    # Remove values with degree centrality equal to 0
    df_topic_interest = df_topic_interest[df_topic_interest["degree_centrality"] != 0]

    # Remove mentions of Bitcoin and Ethereum before 2007
    df_topic_interest = df_topic_interest[
        ~(
            (df_topic_interest["top_node_names"] == "Bitcoin")
            & (df_topic_interest["time"] < pd.to_datetime("2007-01-01"))
        )
    ]
    df_topic_interest = df_topic_interest[
        ~(
            (df_topic_interest["top_node_names"] == "Ethereum")
            & (df_topic_interest["time"] < pd.to_datetime("2007-01-01"))
        )
    ]

    print("Calculating the percentage growth rate for each topic of interest...")

    df_growth_rate = pd.concat(
        [percentage_growth_rate(df_topic_interest, topic) for topic in topics_interest]
    )

    print("Saving the dataframe with the percentage growth rate to a csv file...")
    df_growth_rate.to_csv(
        os.path.join(args.temp_graph_dir, "topics_growth_rate.csv"), index=False
    )

    # # Make time to be in year format
    # df_topic_interest["time"] = df_topic_interest["time"].dt.year

    # Set the style of the plot
    sns.set_style("white")
    sns.set_context("notebook")

    plt.figure()

    # Create a scatterplot
    # sns.scatterplot(x="time", y="degree_centrality", data=df_expanded, alpha=0.7)
    sns.lineplot(
        x="time",
        y="degree_centrality",
        hue="top_node_names",
        markers=True,
        dashes=False,
        style="top_node_names",
        data=df_topic_interest,
        sort=True,
        alpha=1,
        linewidth=5,
    )

    # Set the x-limits of the current axes
    # plt.xlim(df_expanded['time'].min(), pd.to_datetime("2019-12-31"))

    sns.despine()

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.ylabel("Degree Centrality", fontsize=18)
    plt.xlabel("Year", fontsize=18)

    # plt.yscale("log")

    # # Create a custom legend
    # legend_elements = [
    #     mlines.Line2D([], [], color=sns.color_palette()[i], label=topic)
    #     for i, topic in enumerate(topics_interest)
    # ]

    plt.legend(
        # bbox_to_anchor=(0.5, 1.25),
        # loc="upper center",
        loc="best",
        ncol=2,
        fontsize=12,
        frameon=False,
        framealpha=0.0,
    )

    plt.tight_layout()

    plot_path = os.path.join(args.plot_dir, "topics_growth.pdf")

    plt.savefig(plot_path, dpi=300, bbox_inches="tight", transparent=True, format="pdf")

    # Show the plot
    plt.show()
