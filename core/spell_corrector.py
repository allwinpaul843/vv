from spellchecker import SpellChecker

from rapidfuzz import fuzz
from core.fuzzy_filter import get_vocab
import nltk

#nltk.download('stopwords', download_dir='nltk_data')     for downloading locally to save time, run this once


from nltk.corpus import stopwords
nltk.data.path.append("nltk_data")  # Local folder for corpora


'''try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')'''



spell = SpellChecker()  # Initialize spell checker

def precheck_with_spellchecker(token, vocab):
    if not isinstance(token, str) or not token.strip():
        return None  # Skip invalid tokens

    if token in vocab:
        return token  # Token already valid

    try:
        candidates = spell.candidates(token)  # Generate correction candidates
        for cand in candidates:
            if cand in vocab:  # Return first valid candidate found in vocab
                return cand
    except Exception:
        return None

    return None


# ===================== Soundex Suggestions =====================

def soundex(word):
    word = word.lower()
    codes = {
        'a': '', 'e': '', 'i': '', 'o': '', 'u': '', 'y': '', 'h': '', 'w': '',
        'b': '1', 'f': '1', 'p': '1', 'v': '1',
        'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
        'd': '3', 't': '3',
        'l': '4',
        'm': '5', 'n': '5',
        'r': '6'
    }
    if not word:
        return "0000"
    first_letter = word[0].upper()
    encoded = first_letter
    for char in word[1:]:
        encoded += codes.get(char, '')
    encoded = encoded.replace('0', '')

    if not encoded:
        return "0000"

    result = encoded[0]
    for char in encoded[1:]:
        if char != result[-1]:
            result += char
    return (result + '000')[:4]


# Create a cache dictionary outside the loop
score_cache = {}

def score_cached_fuzz_ratio(a, b):
    key = (a, b)            
    if key in score_cache:  
        return score_cache[key]
    score = fuzz.ratio(a, b)
    score_cache[key] = score
    return score

# ===================== QUERY CORRECTION =====================
def correct_query(query, vocab, token_freq, threshold=74):
    original_tokens = query.lower().split()  # Tokenize query
    corrected = []
    
   
    typo_found = False  # Flag if correction happened
    from nltk.corpus import stopwords
    exclude = set(corrected).union(set(stopwords.words('english')))

    for token in original_tokens:
        
        if token in vocab:
            
            corrected.append(token)  # Token is valid
            
        else:
            
            # Use fuzzy match to find candidates above threshold
            candidates = [(w, score_cached_fuzz_ratio(token, w)) for w in vocab if score_cached_fuzz_ratio(token, w) >= threshold]
            
            print(f"Candidates for '{token}': {candidates}")
            if len(candidates)> 0:
                # Pick best based on score and frequency
                best = sorted(candidates, key=lambda x: (-x[1], -token_freq[x[0]]))[0][0]
                
                corrected.append(best)
                typo_found = True
                continue
            
            pre_checked = precheck_with_spellchecker(token, vocab)
            if pre_checked:
                print(token, "prechecked to", pre_checked)
                corrected.append(pre_checked)
                typo_found = True
                continue

            if token in exclude:
                
                corrected.append(token)  # Excluded token, keep original
                continue

            candidates_soundex = [(w, score_cached_fuzz_ratio(token, w)) for w in vocab if(score_cached_fuzz_ratio(token, w) >= 65 and score_cached_fuzz_ratio(token, w) < 74 )]
            soundex_bool = False
            if candidates_soundex:
                
                # Pick best based on score and frequency
                best_soundex = sorted(candidates, key=lambda x: (-x[1], -token_freq[x[0]]))
                query_sdx=soundex(token)
                
                for word in best_soundex:
                      word=word[0]
                      if word in exclude:
                          continue
                      if soundex(word) == query_sdx:
                          corrected.append(word)
                          typo_found = True
                          soundex_bool = True
                          break  # Found a good soundex match

                if soundex_bool:
                    continue

            if not soundex_bool:
                
                corrected.append(token)  # No good match, keep original

    return corrected, typo_found  # Return original, corrected, and flag