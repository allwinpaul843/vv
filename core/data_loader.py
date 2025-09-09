import pandas as pd
import re

# ===================== DATA LOAD & CLEAN =====================
def load_data():
    df1 = pd.read_csv("data/electronics_dataset_1000.csv")  # Load Amazon product data
    df2 = pd.read_csv("data/electronics.csv")  # Load electronics product data

    required_cols = ['title', 'categories', 'manufacturer', 'brand', 'category_code', 'brand01']  # Only keep the title column
    #required_cols = ['title']
    # Ensure both dataframes have the required column
    for col in required_cols:
        if col not in df1.columns:
            df1[col] = ''
        if col not in df2.columns:
            df2[col] = ''

    df1 = df1[required_cols]  # Subset to required columns
    df2 = df2[required_cols]
    df = pd.concat([df1, df2], ignore_index=True)  # Combine both datasets
    df.fillna('', inplace=True)  # Fill missing values with empty string

    string_columns = df.select_dtypes(include='object').columns  # Get all text-based columns

    # Clean text: lowercase, remove non-alphanumerics, strip spaces
    for col in string_columns:
        df[col + "_clean"] = (
            df[col].astype(str)
            .str.lower()
            .str.replace(r'[^a-z0-9 ]+', ' ', regex=True)
            .str.strip()
        )

    return df, string_columns  # Return cleaned DataFrame and string columns

