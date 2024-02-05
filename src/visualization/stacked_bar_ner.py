import argparse
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn.objects as so


def pre_process_data(df_filtered):
    # Filter only the columns with source in its name
    source_cols = df_filtered.columns[df_filtered.columns.str.contains("source")]
    df_source = df_filtered[source_cols]

    # Remove duplicated rows based on unique papers
    df_source = df_source.drop_duplicates(subset=["source_title"])

    source_named_entities_density = [
        c
        for c in df_source.columns
        if c.startswith("source_named_entities") and c.endswith("density")
    ]

    # Reshape the dataframe to have the source_named_entities columns as rows under a column named 'named_entity'
    df_source_filtered = df_source.melt(
        id_vars=["source_title", "source_year"],
        value_vars=source_named_entities_density,
        value_name="named_entities_density",
        var_name="named_entity",
    )

    # Correct the named_entity column
    df_source_filtered["named_entity"] = (
        df_source_filtered["named_entity"]
        .str.replace("source_named_entities.", "")
        .str.replace(".density", "")
    )

    # Remove nan values from year
    df_source_filtered = df_source_filtered[df_source_filtered["source_year"].notna()]

    # Convert year to integer ignoring nan values
    df_source_filtered["source_year"] = df_source_filtered["source_year"].astype(int)

    # Replace 0 values with nan
    # df_source_filtered['source_year'].replace(0, np.nan, inplace=True)

    # Reset the index of the dataframe
    df_source_filtered = df_source_filtered.reset_index(drop=True)

    return df_source_filtered


def normalize_df(df):
    # First, we'll group the data by year and named_entity, and sum up the named_entities_density for each group.
    grouped_df = (
        df.groupby(["source_year", "named_entity"])
        .named_entities_density.sum()
        .reset_index()
    )

    # Pivot the data to get named_entities as columns and their densities as values
    pivot_df = grouped_df.pivot(
        index="source_year", columns="named_entity", values="named_entities_density"
    )

    # Normalize the data so that the total for each year is 1
    ndf = pivot_df.div(pivot_df.sum(axis=1), axis=0)

    return ndf


def generate_plot(df, colormap, plot_path):
    # Create a list of colors for the plot in the same order as the columns
    colors = [colormap[col] for col in df.columns]

    # Add the layout
    sns.set_context("notebook")
    sns.set_style("white")
    # sns.set_style("whitegrid")

    # Plot with the specified colors
    plt.figure(figsize=(14, 9))
    df.plot(kind="bar", stacked=True, color=colors, ax=plt.gca(), width=0.7, alpha=1)

    # Remove the spines
    # sns.despine(top=True, left=False)
    sns.despine()

    # plt.title('Normalized Named Entities Density per Year', fontsize=13)
    plt.xlabel("Year", fontsize=20)
    plt.ylabel("Normalized Density", fontsize=20)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Move the legend to the bottom and distribute the labels in two columns
    plt.legend(
        loc="upper center",
        # bbox_to_anchor=(0.5, -0.30),
        bbox_to_anchor=(0.5, 1.25),
        frameon=False,
        fontsize=18,
        # prop=dict(weight='bold'),
        ncol=3,
    )

    plt.tight_layout()  # Adjust layout to avoid clipping

    plt.savefig(plot_path, dpi=300, bbox_inches="tight", transparent=True, format="pdf")

    plt.show()


def generate_plot_sns(normalized_df, colormap, plot_path):
    # # First, we'll group the data by year and named_entity, and sum up the named_entities_density for each group.
    # grouped_df = df.groupby(['source_year', 'named_entity']).named_entities_density.sum().reset_index()
    #
    # # Pivot the data to get named_entities as columns and their densities as values
    # pivot_df = grouped_df.pivot(index='source_year',
    #                             columns='named_entity',
    #                             values='named_entities_density')
    #
    # # Normalize the data so that the total for each year is 1
    # normalized_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)
    #
    # # Create a list of colors for the plot in the same order as the columns
    # colors = [colormap[col] for col in pivot_df.columns]

    # Unpivot the dataframe
    df = normalized_df.reset_index().melt(
        id_vars="source_year",
        value_vars=normalized_df.columns,
        var_name="named_entity",
        value_name="named_entities_density",
    )

    # # Create a list of colors for the plot in the same order as the columns
    colors = [colormap[lb] for lb in df["named_entity"].tolist()]
    # hex_colors = [mpl.colors.rgb2hex(c) for c in colors]

    p = so.Plot(df, y="named_entities_density", x="source_year", color=colors).add(
        so.Bar(), so.Stack()
    )
    p = p.scale(color=None)

    p = p.label(y="Normalized Density", x="Year", color="", legend="Named Entity")

    p.plot()

    p.show()
    # p.save(plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temp_graph_dir", type=str, default="./../../data/temporal_graph/"
    )
    parser.add_argument("--temp_graph_stats_csv", type=str, default="ner_filtered.csv")
    parser.add_argument("--plot_dir", type=str, default="./../../reports/figures/")
    parser.add_argument(
        "--plot_name", type=str, default="NormalizedNamedEntitiesYear.pdf"
    )
    parser.add_argument(
        "--colormap", type=str, default="./../../data/labels_colourmap.json"
    )

    args = parser.parse_args()

    # Load the temporal graph
    temp_graph_path = os.path.join(args.temp_graph_dir, args.temp_graph_stats_csv)

    # Save the data to a csv file
    df = pd.read_csv(temp_graph_path)

    df = pre_process_data(df)

    # Sort values by year
    df = df.sort_values(by="source_year")

    # Load the colormap
    with open(args.colormap) as f:
        colormap = json.load(f)

    labels = df["named_entity"].unique()

    # Filter only for the 2000-2019 period
    df = df[df["source_year"] >= 1990]
    df = df[df["source_year"] <= 2019]

    # Normalize the dataframe
    df = normalize_df(df)

    plot_path = os.path.join(args.plot_dir, args.plot_name)

    generate_plot(df, colormap, plot_path)

    # generate_plot_sns(df, colormap, plot_path)
