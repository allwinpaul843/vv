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
You are an assistant that rewrites long e-commerce product titles into short, well-structured search queries and let they recommend based on the text {query}.

right now there are product titles in a list data type which is converted to string , so right now it is in string format - {string_suggestions_final_list}, 

Right now, there are product titles in a list data type (converted to string) - '{string_suggestions_final_list}'and rewrite it into a short, 3 to 4 word query and recommend based on the text - " {query} ".
- Rewrite the title into a short, 3 to 4 word query and recommend based on the text - " {query} ".
- userfriendly search queries needed, like a user would type in a search bar.
- Focus on the most relevant information.
- Include brand, product type, size, key features only
- The output must use proper capitalization and punctuation.
- Always capitalize brand names and proper nouns (e.g., Apple, JBL, iPhone, Bluetooth).
- Insert commas or hyphens only if necessary for clarity (e.g., "Noise-Cancelling", "Over-Ear").
- Do not include model numbers, years, storage sizes, or marketing terms like "Renewed" or "Latest".
- Only keep relevant brand names, product type, and 1–2 key features. If the title is too long, remove less relevant details.
- Only output the final list. Do not include explanations, numbering, or markdown.
- Make sure the response ** starts with a opening square bracket `[` ends with a closing square bracket `]` ** — no incomplete lists.
- I will parse your response using `ast.literal_eval()` — so it must be a valid Python list.

Return the output **only as a Python list of strings**, like this: ["Samsung Electronics", "Samsung Phone", "Samsung TV"]
No other text should be included.

I have provided Example, like the below format i needed so that i can convert to list using - ast.literal_eval() which will be use it in my code:

["Samsung Electronics","Samsung Galaxy S21 Ultra Android Phone","Samsung Galaxy S21 Case","Samsung Dual USB Charger","Samsung TV Wall Mount"]

based on the above example rewrite the following product title into a short search query and recommend based on the text {query}
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
    

'''def safe_literal_eval(content):
    """
    Attempts to safely repair and parse the string response from LLM
    when ast.literal_eval fails due to truncation or syntax issues.
    """
    import re

    # Attempt to extract the first list-like structure from the content
    match = re.search(r'(\[.*)', content.strip(), re.DOTALL)
    if match:
        partial = match.group(1)

        # Repair: if the list is missing a closing bracket
        if not partial.endswith("]"):
            partial += "]"

        try:
            print(f"[Repairing] Attempting to parse: {partial}")
            return ast.literal_eval(partial)
        except Exception as e:
            print(f"ast.literal_eval failed after repair: {e}")
            return []

    print("[Error] Could not find list structure in LLM output.")
    return []

'''
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
def get_combined_suggestions(corrected_text, relevant_df, string_columns):
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
    if not candidate_phrases:
        return []

    reranked = rerank_with_sbert(corrected_text, candidate_phrases)
    final_suggestions = []
    suggestion_list = [phrase.strip().lower() for phrase, _ in reranked]
    stringified_list = str(suggestion_list)

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
            improved = generate_query_remote(suggestion_list, corrected_text.strip().lower())
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
