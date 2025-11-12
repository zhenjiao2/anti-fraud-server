import argparse
import os
from pathlib import Path

import pandas as pd


def merge_excel_with_model(folder: str = "gpt41", model_name: str = "gpt4.1",
                           excel_path: str = "Telecom_Fraud_Texts_human.xlsx", output_path: str = "result.csv"):
    """Merge each sheet in the Excel file with the corresponding model CSV files.

    Args:
        folder: folder where model CSVs live (default: 'gpt41')
        model_name: model identifier used in CSV filenames (default: 'gpt4.1')
        excel_path: path to the input Excel file
        output_path: path to write the merged CSV
    """
    xls = pd.ExcelFile(excel_path)

    df_list = []
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # Keep expected columns and rename
        if 'content' not in df.columns or 'rating' not in df.columns:
            raise ValueError(f"Expected columns 'content' and 'rating' in sheet {sheet_name}")

        df = df[['content', 'rating']]
        df = df.rename(columns={"rating": "human rating"})

        # Build the model CSV path
        csv_name = f"dataset-{sheet_name}_{model_name}_test.csv"
        csv_path = Path(folder) / csv_name

        if not csv_path.exists():
            raise FileNotFoundError(f"Model CSV not found: {csv_path}")

        df2 = pd.read_csv(csv_path, encoding='utf-8')

        if 'rating' not in df2.columns:
            raise ValueError(f"Model CSV {csv_path} missing 'rating' column")

        df2 = df2['rating']

        df_list.append(pd.concat([df.reset_index(drop=True), df2.reset_index(drop=True)], axis=1))

    # Concatenate all sheets
    merged_df = pd.concat(df_list, ignore_index=True)

    # Write output
    merged_df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Merge human-labeled Excel sheets with model CSVs")
    parser.add_argument("--input", type=str, default="gpt41", help="folder containing model CSVs")
    parser.add_argument("--model-name", type=str, default="gpt4.1", help="model name used in CSV filenames")
    parser.add_argument("--excel", type=str, default="Telecom_Fraud_Texts_human.xlsx", help="input Excel file path")
    parser.add_argument("--output", type=str, default="result.csv", help="output CSV path")

    args = parser.parse_args()

    merge_excel_with_model(folder=args.input, model_name=args.model_name,
                           excel_path=args.excel, output_path=args.output)


if __name__ == "__main__":
    main()
