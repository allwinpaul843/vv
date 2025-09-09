from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from itertools import chain
import torch
import requests
import json
import time
import ast
import sqlite3
import os
import pickle
from rapidfuzz import fuzz


# ===================== SBERT Setup =====================
sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ===================== Semantic Cache Setup =====================
SEMANTIC_DB_FILE = "semantic_prompt_cache.sqlite"

def init_semantic_cache():
    conn = sqlite3.connect(SEMANTIC_DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS semantic_cache (
            prompt_key TEXT PRIMARY KEY,
            embedding BLOB,
            suggestion TEXT
        )
    """)
    conn.commit()
    return conn

def serialize_tensor(tensor):
    return pickle.dumps(tensor.cpu().numpy())

def deserialize_tensor(blob):
    return torch.tensor(pickle.loads(blob))

def lookup_semantic_cache(conn, semantic_key, threshold=0.95):
    input_embedding = sbert_model.encode([semantic_key], convert_to_tensor=True)[0]
    cur = conn.cursor()
    cur.execute("SELECT prompt_key, embedding, suggestion FROM semantic_cache")
    for row in cur.fetchall():
        cached_key, blob, suggestion = row
        cached_embedding = deserialize_tensor(blob)
        similarity = util.pytorch_cos_sim(input_embedding, cached_embedding).item()
        if similarity >= threshold:
            print(f"[Semantic Cache HIT] Similarity: {similarity:.3f} | Matched: {cached_key}")
            try:
                return ast.literal_eval(suggestion)
            except Exception:
                return None
    return None

def save_to_semantic_cache(conn, semantic_key, suggestion_list):
    embedding = sbert_model.encode([semantic_key], convert_to_tensor=True)[0]
    blob = serialize_tensor(embedding)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO semantic_cache (prompt_key, embedding, suggestion)
        VALUES (?, ?, ?)
    """, (semantic_key, blob, json.dumps(suggestion_list)))
    conn.commit()

# ===================== SBERT Reranking =====================
def rerank_with_sbert(query_text, candidates):
    if not candidates:
        return []
    embeddings = sbert_model.encode([query_text] + candidates, convert_to_tensor=True)
    query_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]
    cosine_scores = util.pytorch_cos_sim(query_embedding, candidate_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=min(5, len(candidates)))
    return [(candidates[i], cosine_scores[i].item()) for i in top_results.indices]

# ===================== Prompt Builder =====================
def build_prompt(string_suggestions_final_list, query=None):
    return f"""
You are an assistant that rewrites long e-commerce product titles into short, well-structured search queries and recommends based on the text "{query}". If the titles are not related to the query, return an empty list.

You are given product titles as a stringified Python list: {string_suggestions_final_list}

Instructions:

- First, determine if the query({query}) is about a main product (e.g., "iPhone"), a product series (e.g., "iPhone headphones"), a part (e.g., "iPhone battery"), or an accessory (e.g., "iPhone case").
- Based on the type of query({query}), suggest relevant products, series, parts, or accessories from the provided list {string_suggestions_final_list}.
- Based on the type of query if brand is present, mostly go with the product of that brand that mostly consumed by the users if it is available in the list{string_suggestions_final_list}.
- For a main product (e.g., "iPhone"):
    1. Suggest other series or models of the same product (e.g., "iPhone 12", "iPhone 13") if present in the list.
    2. Then, suggest related accessories or parts for that product (e.g., "iPhone Case", "iPhone Battery") if present in the list {string_suggestions_final_list}.
    3. Then, suggest similar products from other brands (e.g., "Samsung Phone") if present in the list.
- For a product series or accessory (e.g., "iPhone headphones"):
    1. Suggest other series or models of the same accessory for the same product (e.g., "iPhone Earbuds", "iPhone Wireless Headphones") if present in the list {string_suggestions_final_list}.
    2. Then, suggest similar accessories for other brands (e.g., "Samsung Headphones", "Sony Headphones") if present in the list {string_suggestions_final_list}.
- For a part (e.g., "iPhone battery"):
    1. Suggest other batteries or parts for the same product series (e.g., "iPhone 12 Battery", "iPhone 13 Battery") if present in the list {string_suggestions_final_list}.
    2. Then, suggest similar parts for other brands (e.g., "Samsung Battery") if present in the list {string_suggestions_final_list}.
    3. Then, suggest related accessories for the same product if present in the list {string_suggestions_final_list}.
- Always prioritize suggestions in this order: same product/series first, then other brands, then other types.
- Do not invent or add products, parts, or accessories that are not present in the provided list.
- If the query ({query}) is coming with negative intent, such as "not", "without", or "no", do not suggest any products or accessories related to the query. Based on the query category, suggest only relevant items.
- negative intent words include: ['not', 'without', 'no', 'never', 'none', 'nothing', 'neither', 'nor', 'nobody', 'nowhere','exclude', 'avoid', 'skip', 'omit', 'disregard', 'ignore', 'reject', 'refuse', 'deny', 'abandon', 'discard', 'eliminate', 'remove','except', 'except for', 'apart from', 'aside from', 'besides', 'beyond', 'outside of', 'other than']
- Do not repeat the query itself or suggest titles that are too similar to the query.
- Ensure that all suggestions are unique and meaningful.
- Avoid redundant or repetitive suggestions.
- Focus on generating diverse suggestions that are relevant to the query.
- Rewrite the product title into a short, user-friendly search query (3–4 words).
- Focus on brand, product type, and 1–2 key features.
- Do not include model numbers, years, storage sizes, or marketing terms like "Renewed" or "Latest".
- The output must be a valid Python list of strings, starting with `[` and ending with `]`.
- Only output the final list. Do not include explanations, numbering, or markdown.

**Example:**
If the input list is ["iPhone", "iPhone 12", "iPhone Battery", "iPhone Case", "Samsung Phone", "Samsung Battery", "Sony Headphones"] and the query is "iPhone Battery", suggest ["iPhone Battery", "iPhone 12 Battery", "Samsung Battery", "iPhone Case"] (if all are present).

Return only the Python list of strings as output, nothing else.

Rewrite the following product titles and recommend based on the text "{query}":
{string_suggestions_final_list}
""".strip()

# ===================== Remote Generation =====================
def generate_query_remote(title, query=None):
    api_key = "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"  # Replace with your actual key
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "<YOUR_SITE_URL>",
        "X-Title": "<YOUR_SITE_NAME>",
    }
    prompt = build_prompt(title, query).strip()
    data = {
        "model": "meta-llama/llama-3.3-70b-instruct",
        "messages": [
            {"role": "system", "content": "You are a concise search query generator."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 30,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        content = response.json()["choices"][0]["message"]["content"]
        try:
            print(response.json()["choices"][0]["message"])
            return ast.literal_eval(content)
        except Exception:
            print("[Warning] ast.literal_eval failed. Trying safe repair...")
            return safe_literal_eval(content)
    else:
        raise Exception(f"API Error {response.status_code}: {response.text}")
    


def safe_literal_eval(content):
    """
    Attempts to safely parse and repair a possibly broken LLM output
    into a valid Python list of strings.
    Ensures:
    - Starts with [ and ends with ]
    - All items are quoted properly
    - Removes broken or unterminated strings
    """
    import re

    # Try to find the list structure
    match = re.search(r'\[.*', content.strip(), re.DOTALL)
    if not match:
        print("[Error] Could not find list structure in LLM output.")
        return []

    partial = match.group(0).strip()

    # Step 1: Ensure list starts with `[` and ends with `]`
    if not partial.startswith("["):
        partial = "[" + partial
    if not partial.endswith("]"):
        partial += "]"

    # Step 2: Try basic parse
    try:
        return ast.literal_eval(partial)
    except Exception as e:
        print(f"[Repairing] Basic parse failed: {e}")

    # Step 3: Try manual cleaning
    # Extract everything between the outer brackets
    try:
        inside = re.search(r'\[(.*)\]', partial, re.DOTALL).group(1)
    except:
        print("[Error] Failed to extract contents inside brackets.")
        return []

    # Step 4: Split elements manually, assuming comma-separated
    raw_items = inside.split(",")

    cleaned_items = []
    for item in raw_items:
        item = item.strip()
        # Match items that look like proper quoted strings
        if re.match(r'^"(.*?)"$', item) or re.match(r"^'(.*?)'$", item):
            cleaned_items.append(ast.literal_eval(item))  # unquote safely
        else:
            print(f"[Skipping] Invalid or unterminated string: {item}")

    # Step 5: Return as valid list
    return cleaned_items


# ===================== Exact Match Cache DB =====================
DB_FILE = "remote_generation_cache.sqlite"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS remote_cache (
            phrase TEXT,
            corrected_text TEXT,
            suggestion TEXT,
            PRIMARY KEY (phrase, corrected_text)
        )
    """)
    conn.commit()
    return conn

def lookup_cache(conn, phrase, corrected_text):
    cur = conn.cursor()
    cur.execute("""
        SELECT suggestion FROM remote_cache
        WHERE phrase = ? AND corrected_text = ?
    """, (phrase.strip().lower(), corrected_text.strip().lower()))
    row = cur.fetchone()
    return row[0] if row else None

def save_to_cache(conn, phrase, corrected_text, suggestion):
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO remote_cache (phrase, corrected_text, suggestion)
        VALUES (?, ?, ?)
    """, (phrase.strip().lower(), corrected_text.strip().lower(), suggestion))
    conn.commit()




# ===================== FINAL ENRICHMENT =====================
def get_combined_suggestions(corrected_text, relevant_df, string_columns,_df):
    category_column= 'categories'  # Update this based on your dataset structure
    if not corrected_text.strip():
        return []

    corrected_tokens = set(corrected_text.lower().split())
    all_titles = list(set(
        chain.from_iterable(
            relevant_df[col + "_clean"].dropna().unique().tolist()
            for col in string_columns
        )
    ))
    candidate_phrases = [
        title for title in all_titles
        if any(tok in title.split() for tok in corrected_tokens)
    ]


    print(f"[Debug] Candidate phrases found: {len(candidate_phrases)}")
    print("[Debug] Candidate phrases:",candidate_phrases)  # Show first 5 for brevity
    if not candidate_phrases:
        return []
    
    reranked = rerank_with_sbert(corrected_text, candidate_phrases)
    final_suggestions = []
    suggestion_list = [phrase.strip().lower() for phrase, _ in reranked]
    
    
    #print(f"[Debug] Stringified suggestion list: {stringified_list}")

    # Extract categories for candidate phrases from all relevant columns
    categories = []
    matching=[]
    #for col in string_columns:
    matched_list = relevant_df.loc[
            relevant_df['title' + "_clean"].isin(suggestion_list),
            category_column
        ].dropna().unique().tolist()
    if matched_list:
        for cat in matched_list:
            matching.extend(safe_literal_eval(cat))

    if matching:
        categories = list(set(matching))


    
    print(f"[Debug] Categories before cleaning: {categories}")

  
    '''    # Make sure category list is lowercase and stripped
        categories = [c.strip().lower() for c in categories if c.strip()]

        # Step 1: Filter rows where any category string is found inside the category column
        matched_rows = _df[
        _df[category_column].astype(str).str.lower().apply(
            lambda cell: sum(cat in cell for cat in categories) >= 5
        ) ]'''

    '''matched_rows = _df[
        _df[category_column].astype(str).str.lower().apply(
            lambda cell: any(cat in cell for cat in categories)
        )
    ]'''

    categories = [c.strip().lower() for c in categories if c.strip()]
    print(f"[Debug] Cleaned categories: {len(categories)} found")

    # Step 1: Annotate each row with how many category terms are found in its category column
    _df["match_count"] = _df[category_column].astype(str).str.lower().apply(
        lambda cell: sum(cat in cell for cat in categories)
    )

    print("[Debug] Match count per row:")
    print(_df["match_count"].tolist())

    

    _df.to_csv("matched_rows_cate.csv", index=False)

    # matched_rows = _df[_df["match_count"] >= 4]
    
    # Dynamically select matched_rows based on match_count threshold
    max_match = _df["match_count"].max()
    matched_rows = _df.iloc[0:0]  # create empty DataFra   me with same columns as _df
    if max_match > 0:
        matched_rows = _df[_df["match_count"] == max_match]


    # Reduce threshold until we get at least 10 (max 20) rows, or stop if max_match is 0
    while len(matched_rows) < 10 and max_match > 0:
        max_match -= 1  
        matched_rows = _df[_df["match_count"] == max_match]

    # Limit to maximum 20 rows
    matched_rows = matched_rows.sort_values(by="match_count", ascending=False).head(10)

    # Step 2: Collect clean titles from matched rows
    matched_titles = list(set(
        chain.from_iterable(
            matched_rows[col + "_clean"].dropna().tolist()
            for col in string_columns
        )
    ))

    print(f"[Debug] Matched titles: {matched_rows}")
    print(f"[Debug] Number of matched titles: {len(matched_rows)}")

    # Step 3: Add matched titles to suggestion list
    suggestion_list += [title.strip().lower() for title in matched_titles if title]
    suggestion_list = list(set(suggestion_list))  # Ensure uniqueness


    #suggestion_list+= categories
    stringified_list = str(suggestion_list)
    print(f"[Debug] Stringified suggestion list: {stringified_list}")

    conn = init_db()
    cached = lookup_cache(conn, stringified_list, corrected_text)
    if cached:
        try:
            improved = ast.literal_eval(cached)
        except Exception:
            improved = []
    else:
        # ===================== Semantic Cache Integration =====================
        semantic_key = f"{stringified_list.strip().lower()} | {corrected_text.strip().lower()}"
        semantic_conn = init_semantic_cache()
        improved = lookup_semantic_cache(semantic_conn, semantic_key, threshold=0.95)

        if not improved:
            improved = generate_query_remote(stringified_list, corrected_text.strip().lower())
            if improved and len(improved) > 0:
                save_to_semantic_cache(semantic_conn, semantic_key, improved)

        semantic_conn.close()
        if improved and len(improved) > 0:
            save_to_cache(conn, stringified_list, corrected_text, json.dumps(improved))

    

    for improve in improved:
        if improve not in final_suggestions and improve.lower() != corrected_text.lower():
            final_suggestions.append(improve)
    if not final_suggestions:
        final_suggestions.append("No suggestions found")

    conn.close()
    return final_suggestions
