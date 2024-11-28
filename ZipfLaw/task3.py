import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from task2 import read_csv_files
from task1 import process_files


def zipf_mandelbrot(r, C, a, b):
    """
    Zipf-Mandelbrot law function.

    :param r: Rank of the word.
    :param C: Normalization constant.
    :param a: Exponent parameter.
    :param b: Offset parameter.
    :return: Frequency of the word with rank r.
    """
    return C * (1 / (r + b) ** a)


def fit_zipf_mandelbrot(df):
    """
    Fits the Zipf-Mandelbrot law to the word distribution in the DataFrame.

    :param df: DataFrame with columns: 'Rank', 'Frequency'.
    :return: Fitted parameters C, a, b.
    """
    ranks = df["Rank"].values
    frequencies = df["Frequency"].values

    # Initial guess for the parameters C, a, b
    initial_guess = [max(frequencies), 1.0, 1.0]

    # Fit the Zipf-Mandelbrot law to the data
    params, _ = curve_fit(zipf_mandelbrot, ranks, frequencies, p0=initial_guess)

    return params


def plot_parameters(results_df):
    """
    Plots the a and b parameters for each language on a scatter plot.

    :param results_df: DataFrame with columns: 'language', 'a', 'b'.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df["a"], results_df["b"], c="blue", marker="o")

    for _, row in results_df.iterrows():
        plt.text(row["a"], row["b"], row["language"], fontsize=9)

    plt.xlabel("a parameter")
    plt.ylabel("b parameter")
    plt.title("Scatter Plot of a and b Parameters for Each Language")
    plt.grid(True)
    plt.show()


def main():
    files_dir = "languages"
    output_dir = "languages_output"
    process_files(files_dir, output_dir)
    dataframes = read_csv_files(output_dir)

    results = []
    for filename, df in dataframes.items():
        _, a, b = fit_zipf_mandelbrot(df)
        language = filename.split("_")[0]
        results.append((language, a, b))

    results_df = pd.DataFrame(results, columns=["language", "a", "b"])
    plot_parameters(results_df)


if __name__ == "__main__":
    main()
