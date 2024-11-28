import os
import matplotlib.pyplot as plt
import pandas as pd


def read_csv_files(directory):
    """
    Reads all CSV files in the specified directory and constructs a pandas DataFrame for each.

    :param directory: Path to the directory containing CSV files.
    :return: Dict of filename - df pairs
    """
    dataframes = {}
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            dataframes[filename] = df
    return dataframes


def generate_zipf_law_dataframe(df):
    """
    Generates a DataFrame with Count and Frequency values determined by Zipf's law.
    :param df: DataFrame with columns: 'Rank', 'Word', 'Count', 'Frequency'.
    :return: DataFrame with columns: 'Rank', 'Word', 'Zipf_Count', 'Zipf_Frequency'.
    """
    total_words = df["Count"].sum()
    max_rank = df["Rank"].max()

    # Calculate normalization constant
    normalization_constant = 1 / (sum(1 / rank for rank in range(1, max_rank + 1)))

    zipf_data = []

    for _, row in df.iterrows():
        rank = row["Rank"]
        word = row["Word"]
        zipf_count = total_words / rank
        zipf_frequency = (1 / rank) * normalization_constant
        zipf_data.append((rank, word, zipf_count, zipf_frequency))

    zipf_df = pd.DataFrame(
        zipf_data, columns=["Rank", "Word", "Zipf_Count", "Zipf_Frequency"]
    )
    return zipf_df


def plot_distributions(actual_distribution_df, expected_distribution_df, scale, ax):
    ax.plot(
        actual_distribution_df["Rank"],
        actual_distribution_df["Frequency"],
        label="Actual Distribution",
        color="blue",
        marker="o",
        linestyle="none",
    )

    ax.plot(
        expected_distribution_df["Rank"],
        expected_distribution_df["Zipf_Frequency"],
        label="Zipf's Law Distribution",
        color="red",
        linestyle="-",
    )

    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Word Frequency Distribution ({scale.capitalize()} Scale)")
    ax.legend()
    ax.grid(True)


def main():
    directory = "books_csv"
    dataframes = read_csv_files(directory)

    for filename, df in dataframes.items():
        zipf_df = generate_zipf_law_dataframe(df)

        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Distributions for {filename}")

        plot_distributions(df, zipf_df, scale="linear", ax=axs[0])
        plot_distributions(df, zipf_df, scale="log", ax=axs[1])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


if __name__ == "__main__":
    main()
