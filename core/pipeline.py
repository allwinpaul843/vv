from core.data_loader import load_data
from core.fuzzy_filter import filter_relevant_rows, get_vocab
from core.spell_corrector import correct_query
from core.enrich_suggestions import get_combined_suggestions
import time
import pandas as pd 

# Load dataset and string columns once
_df, _string_columns = load_data()

# Core function for GUI use
def process_spellcorrect_user_query(query):
    relevant_df = filter_relevant_rows(_df, _string_columns, query)
    relevant_df.to_csv("relevant_rows.csv", index=False)  # Save to CSV for debugging

    vocab, token_freq = get_vocab(relevant_df, _string_columns)
    corrected_tokens,typo_found = correct_query(query, vocab, token_freq)

    corrected_text = " ".join(corrected_tokens)
      
    return typo_found, corrected_text, relevant_df, _string_columns

def process_suggestion_user_query(corrected_text, relevant_df, _string_columns):
    
    # Use the correct column name for categories (e.g., 'categories')
    category_column = 'categories'  # Update this based on your dataset structure

    # Check if the column exists
    if category_column not in _df.columns or category_column not in relevant_df.columns:
        raise KeyError(f"'{category_column}' column is missing in the dataset. Please ensure it exists.")

    # Normalize category column to lowercase for case-insensitive comparison
    _df[category_column] = _df[category_column].str.lower()
    relevant_df[category_column] = relevant_df[category_column].str.lower()

    # Extract unique categories from the filtered relevant rows
    categories = relevant_df[category_column].unique()

    # Fetch additional relevant data based on the normalized categories
    additional_relevant_data = _df[_df[category_column].isin(categories)]

    # Combine the filtered relevant rows with the additional relevant data using pd.concat
    combined_relevant_data = pd.concat([relevant_df, additional_relevant_data]).drop_duplicates()

    # Generate suggestions using the combined relevant data
    combined_relevant_data.to_csv("combined_relevant_data.csv", index=False)  # Save to CSV for debugging
    suggestions = get_combined_suggestions(corrected_text, combined_relevant_data, _string_columns,_df)

    return suggestions
