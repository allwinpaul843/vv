from rapidfuzz import fuzz
from collections import Counter
import pandas as pd
import numpy as np
# Define a simple cache
cache = {}

def cached_fuzz_ratio(query, text):
    key = (query, text)
    if key in cache:
        return cache[key]
    score = fuzz.partial_ratio(query, text)
    cache[key] = score
    return score



# ===================== QUERY FILTERING =====================
def filter_relevant_rows(df, string_columns, query, threshold=70):
    query = query.lower()  # Normalize query
    df['max_match_score'] = 0  # Initialize score column

    score_arrays = []  # To hold score arrays for each column
    # Apply fuzzy partial match for each column
    for col in string_columns:
        scores = df[col + "_clean"].apply(lambda x: cached_fuzz_ratio(query, str(x)) if not pd.isna(x) else 0)
        score_arrays.append(scores.values)

    if score_arrays:
        # Stack scores and calculate row-wise maximum
        all_scores = np.vstack(score_arrays)
        df['max_match_score'] = all_scores.max(axis=0)
    else:
        df['max_match_score'] = 0

        #df['max_match_score'] = df['max_match_score'].combine(scores, max)

    '''
    # Apply fuzzy partial match for each column
    for col in string_columns:
        scores = df[col + "_clean"].apply(lambda x: fuzz.partial_ratio(query, x))
        df['max_match_score'] = df['max_match_score'].combine(scores, max)  # Take max score across columns


    '''
    # Filter rows where max fuzzy match score exceeds threshold
    relevant_rows = df[df['max_match_score'] >= threshold].copy()
    relevant_rows.reset_index(drop=True, inplace=True)

    #relevant_rows.to_csv("relevant_rows.csv", index=False)  # Save to CSV for debugging

    return relevant_rows



# In-memory cache for tokenization results
_token_cache = {}

# ===================== VOCAB EXTRACTION =====================
def get_vocab(filtered_df, string_columns):
    all_tokens = []

    for col in string_columns:
        for val in filtered_df[col + "_clean"]:
            if val in _token_cache:
                all_tokens.extend(_token_cache[val])  # Reuse cached tokens
            else:
                words = val.split()
                tokens = []
                tokens.extend(words)  # unigrams
                tokens.extend([' '.join(words[i:i+2]) for i in range(len(words)-1)])  # bigrams
                tokens.extend([' '.join(words[i:i+3]) for i in range(len(words)-2)])  # trigrams

                _token_cache[val] = tokens  # Save result for next time
                all_tokens.extend(tokens)
    negative_words = ['non','not', 'without', 'no', 'never', 'none', 'nothing', 'neither', 'nor', 'nobody', 'nowhere','exclude', 'avoid', 'skip', 'omit', 'disregard', 'ignore', 'reject', 'refuse', 'deny', 'abandon', 'discard', 'eliminate', 'remove','except', 'except for', 'apart from', 'aside from', 'besides', 'beyond', 'outside of', 'other than']
    all_tokens.extend(['phone','iphone','phones','iphones'])
    all_tokens.extend(negative_words)  # Include negative words as tokens
    all_tokens.extend(string_columns)  # Include column names as tokens

    token_freq = Counter(all_tokens)
    
    return set(token_freq), token_freq

