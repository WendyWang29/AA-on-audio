import pandas as pd
import random
from collections import defaultdict, Counter

def create_dataset(input_csv, output_csv):
    # Paths
    input_txt = input_csv
    output_csv = output_csv
    base_path = "/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_eval/flac/"

    # Load the TXT fileA into a DataFrame
    df = pd.read_csv(input_txt, sep=" ", header=None, names=["id", "file", "hyphen", "attack_type", "label"])

    # Process labels and file paths
    df["label"] = df["label"].apply(lambda x: 1 if x == "spoof" else 0)
    df["path"] = base_path + df["file"] + ".flac"

    # Group by attack types to ensure equal distribution
    attack_groups = defaultdict(list)
    for _, row in df.iterrows():
        attack_type = row["attack_type"]
        attack_groups[attack_type].append(row)

    # Randomly sample files with a balanced attack distribution
    selected_rows = []
    sample_per_attack = 10000 // len(attack_groups)

    for attack, rows in attack_groups.items():
        # Shuffle rows and select a sample
        random.shuffle(rows)
        selected_rows.extend(rows[:sample_per_attack])

    # If fewer than 10,000 rows were selected, fill the rest randomly
    selected_ids = {row["file"] for row in selected_rows}  # Track selected 'file' IDs
    if len(selected_rows) < 10000:
        remaining_rows = [row for _, row in df.iterrows() if row["file"] not in selected_ids]
        selected_rows.extend(random.sample(remaining_rows, 10000 - len(selected_rows)))

    # Create the output DataFrame
    output_df = pd.DataFrame(selected_rows)
    output_df = output_df.reset_index(drop=True)
    output_df["<anonymous>"] = output_df.index
    output_df = output_df[["<anonymous>", "path", "label"]]  # Reorder columns

    # Save to fileB.csv
    output_df.to_csv(output_csv, index=False)

    # Calculate and print final count for each attack type
    attack_counts = Counter(row["attack_type"] for row in selected_rows)
    print("\nFinal distribution of files by attack type:")
    for attack, count in attack_counts.items():
        print(f"{attack}: {count} files")

    # Print total number of files
    print(f"\nTotal number of files selected: {len(selected_rows)}")

if __name__ == '__main__':
    input_csv = '/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
    output_csv = '/nas/home/wwang/SpecResNet/data/df_eval_19_reduced.csv'
    create_dataset(input_csv, output_csv)


