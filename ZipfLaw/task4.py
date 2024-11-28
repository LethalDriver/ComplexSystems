import matplotlib.pyplot as plt
import pandas as pd
from task2 import read_csv_files, generate_zipf_law_dataframe, plot_distributions
from task1 import process_files
import os


def save_llm_essays(csv_file, output_folder, num_essays=4):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Filter the DataFrame to include only essays generated by the LLM
    llm_essays = df[df["generated"] == 1]

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    essay_nb = 1
    # Save each selected essay to a separate .txt file
    for _, row in llm_essays.iterrows():
        essay_text = row["text"]
        file_path = os.path.join(output_folder, f"{essay_nb}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(essay_text)
        essay_nb += 1


if __name__ == "__main__":
    csv_file = "train_essays.csv"
    essays_txt_folder = "LLM_essays"
    essays_csv_folder = "LLM_essays_csv"

    save_llm_essays(csv_file, essays_txt_folder)
    process_files(essays_txt_folder, essays_csv_folder)

    dataframes = read_csv_files(essays_csv_folder)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Distributions for Essays generated by LLM")

    for ax, (filename, df) in zip(axs.flatten(), dataframes.items()):
        zipf_df = generate_zipf_law_dataframe(df)
        plot_distributions(df, zipf_df, scale="log", ax=ax)
        ax.set_title(f"{filename}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()