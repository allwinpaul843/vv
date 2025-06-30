from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from itertools import chain
import torch
import requests
import json
import time

sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#rephrase_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
#rephrase_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")




import sqlite3
import os
from itertools import chain
import time




import pickle
import os
from rapidfuzz import fuzz  # Assuming you are using rapidfuzz





#SBERT_CACHE_FILE = "sbert_cache.pkl"

# Load cache from file if it exists
'''if os.path.exists(SBERT_CACHE_FILE) and os.path.getsize(SBERT_CACHE_FILE) > 0:
    with open(SBERT_CACHE_FILE, "rb") as f:
        rerank_cache = pickle.load(f)
    print(f"Loaded cache with {len(rerank_cache)} entries.")
else:
    rerank_cache = {}
'''



'''
def rerank_with_sbert(query_text, candidates):
    if not candidates:
        return []
    embeddings = sbert_model.encode([query_text] + candidates, convert_to_tensor=True)  # Get embeddings
    query_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]
    cosine_scores = util.pytorch_cos_sim(query_embedding, candidate_embeddings)[0]  # Cosine similarity
    top_results = torch.topk(cosine_scores, k=min(5, len(candidates)))  # Top 5 results
    return [(candidates[i], cosine_scores[i].item()) for i in top_results.indices] 
    '''

#rerank_cache = {}  


'''def make_key(query_text, candidates):
    
    return json.dumps({"query": query_text, "candidates": candidates}, sort_keys=True)'''

def rerank_with_sbert(query_text, candidates):
    if not candidates:
        return []
    
    #key = make_key(query_text, candidates)
    ''' if key in rerank_cache:
            return rerank_cache[key] '''
    
    embeddings = sbert_model.encode([query_text] + candidates, convert_to_tensor=True)
    query_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]
    cosine_scores = util.pytorch_cos_sim(query_embedding, candidate_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=min(5, len(candidates)))
    result = [(candidates[i], cosine_scores[i].item()) for i in top_results.indices]
    
    '''rerank_cache[key] = result
    with open(SBERT_CACHE_FILE, "wb") as f:
        pickle.dump(rerank_cache, f) '''
    
    
    return result







# Prompt builder function
def build_prompt(product_title,query= None):

    return f"""
You are an assistant that rewrites long e-commerce product titles into short, well-structured search queries and let they recommend based on the text {query}.

Your task:
- Rewrite the title into a short, 3 to 4 word query and recommend based on the text - " {query} ".
- userfriendly search queries needed, like a user would type in a search bar.
- Focus on the most relevant information.
- Include brand, product type, size, key features only
- The output must use proper capitalization and punctuation.
- Always capitalize brand names and proper nouns (e.g., Apple, JBL, iPhone, Bluetooth).
- Insert commas or hyphens only if necessary for clarity (e.g., "Noise-Cancelling", "Over-Ear").
- Do not include model numbers, years, storage sizes, or marketing terms like "Renewed" or "Latest".
- Only keep relevant brand names, product type, and 1–2 key features. If the title is too long, remove less relevant details.

Examples:

Product Title: bose quietcomfort 45 bluetooth wireless noise cancelling headphones over ear with mic black
Search Query: Bose Wireless Noise-Cancelling Headphones

Product Title: apple macbook air m2 chip 13 inch laptop 8gb ram 256gb ssd midnight 2023 model
Search Query: Apple MacBook Air

Product Title: jbl flip 6 portable bluetooth speaker with deep bass waterproof design
Search Query: JBL Portable Bluetooth Speaker


based on the above example rewrite the following product title into a short search query and recommend based on the text {query}.:
My Product Title is {product_title} 
And dont provide 'Search Query:' with the output

"""





# Function to generate the search query using requests and measure time
def generate_query_remote(title,query=None):

  #================================Type your openroute apikey====================================

    # Your OpenRouter API key
    api_key = "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"  # Replace with your actual API key
   #================================Type your openroute apikey====================================

    # API endpoint
    url = "https://openrouter.ai/api/v1/chat/completions"

    # Request headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional:
        "HTTP-Referer": "<YOUR_SITE_URL>",
        "X-Title": "<YOUR_SITE_NAME>", }

    prompt = build_prompt(title,query).strip()
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
        response_data = response.json()
        text = response_data["choices"][0]["message"]["content"]
        return text
    else:
        raise Exception(f"API Error {response.status_code}: {response.text}")
    


'''CACHE_FILE = "remote_generation_cache.pkl"


if os.path.exists(CACHE_FILE) and os.path.getsize(CACHE_FILE) > 0:
    with open(CACHE_FILE, "rb") as f:
        remote_generation_cache = pickle.load(f)
    print(f"Loaded cache with {len(remote_generation_cache)} entries.")
else:
    remote_generation_cache = {}'''

DB_FILE = "remote_generation_cache.sqlite"

# One-time DB setup
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

    corrected_tokens = set(corrected_text.lower().split())  # Tokenize corrected query
    all_titles = list(set(
        chain.from_iterable(
            relevant_df[col + "_clean"].dropna().unique().tolist()
            for col in string_columns
        )
    ))  # Flatten all product titles

    # Filter phrases that contain any token from corrected query
    candidate_phrases = [
        title for title in all_titles
        if any(tok in title.split() for tok in corrected_tokens) ]
    if not candidate_phrases:
        return []
    
    

    reranked = rerank_with_sbert(corrected_text, candidate_phrases)  # Rerank candidates with SBERT



    final_suggestions = []

    
        
    conn = init_db()  # Initialize DB connection

    for phrase, score in reranked:
        start_time = time.time()

        cached = lookup_cache(conn, phrase, corrected_text)
        if cached:
            improved = cached
        else:
            try:
                improved = generate_query_remote(phrase.strip().lower(), corrected_text.strip().lower())
                save_to_cache(conn, phrase, corrected_text, improved)
            except Exception as e:
                print(f"Error generating query for '{phrase}': {e}")
                continue

        end_time = time.time()
        print(f"Rephrasing prompt took {end_time - start_time:.2f} seconds")

        if improved in final_suggestions or improved.lower() == corrected_text.lower():
            continue
        final_suggestions.append(improved)

    conn.close()
    return final_suggestions