from core.data_loader import load_data
from core.fuzzy_filter import filter_relevant_rows, get_vocab
from core.spell_corrector import correct_query
from core.enrich_suggestions import get_combined_suggestions
import time

# Load dataset and string columns once
_df, _string_columns = load_data()

# Core function for GUI use
def process_spellcorrect_user_query(query):
    relevant_df = filter_relevant_rows(_df, _string_columns, query)
    vocab, token_freq = get_vocab(relevant_df, _string_columns)
    corrected_tokens,typo_found = correct_query(query, vocab, token_freq)

    corrected_text = " ".join(corrected_tokens)
      
    return typo_found, corrected_text, relevant_df, _string_columns

def process_suggestion_user_query(corrected_text, relevant_df, _string_columns):

 
    
    suggestions = get_combined_suggestions(corrected_text, relevant_df, _string_columns)
 

    return suggestions
