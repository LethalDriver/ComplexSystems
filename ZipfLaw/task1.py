import string
import os
import pandas as pd
from collections import Counter


def read_file_to_word_list(file_path):
    """
    Reads a text file and returns a list of words.

    :param file_path: Path to the text file.
    :return: List of words in the file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    word_list = text.split()
    return word_list


def clean_word_list(word_list):
    """
    Cleans a list of words by removing punctuation and special characters.

    :param word_list: List of words to be cleaned.
    :return: List of cleaned words.
    """
    cleaned_words = []
    for word in word_list:
        cleaned_word = word.strip(string.punctuation + "[]{}()<>").lower()
        cleaned_words.append(cleaned_word)
    return cleaned_words


def construct_word_dataframe(cleaned_word_list):
    """
    Constructs a pandas DataFrame with word rank, word, count, and frequency.

    :param cleaned_word_list: List of cleaned words.
    :return: DataFrame with columns: 'Rank', 'Word', 'Count', 'Frequency'.
    """
    word_counts = Counter(cleaned_word_list)
    total_words = sum(word_counts.values())

    word_data = []
    for rank, (word, count) in enumerate(word_counts.most_common(), start=1):
        frequency = count / total_words
        word_data.append((rank, word, count, frequency))

    df = pd.DataFrame(word_data, columns=["Rank", "Word", "Count", "Frequency"])
    return df


def get_word_count(words):
    return len(words)


def export_dataframe_to_csv(df, file_path):
    """
    Exports a pandas DataFrame to a CSV file.

    :param df: DataFrame to be exported.
    :param file_path: Path to the output CSV file.
    """
    df.to_csv(file_path, index=False)


def process_files(files_dir, output_dir):
    if not os.path.exists(files_dir):
        os.makedirs(files_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(files_dir) if f.endswith(".txt")]

    for file_name in files:
        words = read_file_to_word_list(f"{files_dir}/{file_name}")
        cleaned_words = clean_word_list(words)
        stats_df = construct_word_dataframe(cleaned_words)
        word_count = get_word_count(cleaned_words)
        export_dataframe_to_csv(
            stats_df, f"{output_dir}/{file_name[:-4]}_{word_count}.csv"
        )


def main():
    files_dir = "./books"
    output_dir = "./books_csv"
    process_files(files_dir, output_dir)


if __name__ == "__main__":
    main()
