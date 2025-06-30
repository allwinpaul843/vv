import streamlit as st
from core.pipeline import process_spellcorrect_user_query, process_suggestion_user_query

st.title("Query Correction and Enrichment Tool")

user_query = st.text_area("Enter your search query:")

if st.button("Correct & Suggest"):
    if user_query.strip():
        typo_found, corrected_text, relevant_df, _string_columns = process_spellcorrect_user_query(user_query.strip())
        if not typo_found:
            st.info("No typos found in the query.")
            suggestions = process_suggestion_user_query(user_query.strip(), relevant_df, _string_columns)
            st.write("Top Suggestions:")
            st.write(suggestions)
        else:
            st.success(f"Corrected Query: {corrected_text}")
            suggestions = process_suggestion_user_query(corrected_text, relevant_df, _string_columns)
            st.write("Top Suggestions:")
            st.write(suggestions)
    else:
        st.warning("Please enter a query.")