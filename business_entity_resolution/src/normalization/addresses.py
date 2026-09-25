"""
Address normalization — preserves structure, standardizes abbreviations.
Handles US, India, and France address formats.
"""
import re
import unicodedata
from typing import Dict, Optional


# US address abbreviations
US_ABBREVIATIONS = {
    # Street types
    'rd': 'road', 'rd.': 'road',
    'st': 'street', 'st.': 'street',
    'ave': 'avenue', 'ave.': 'avenue',
    'blvd': 'boulevard', 'blvd.': 'boulevard',
    'dr': 'drive', 'dr.': 'drive',
    'ct': 'court', 'ct.': 'court',
    'cir': 'circle', 'cir.': 'circle',
    'ln': 'lane', 'ln.': 'lane',
    'pl': 'place', 'pl.': 'place',
    'pkwy': 'parkway', 'pky': 'parkway',
    'hwy': 'highway', 'hwy.': 'highway',
    'trl': 'trail', 'trl.': 'trail',
    'ter': 'terrace', 'terr': 'terrace',
    'cres': 'crescent',
    'expy': 'expressway', 'exp': 'expressway',
    'fwy': 'freeway',
    'mt': 'mount', 'mt.': 'mount',
    'ste': 'suite', 'ste.': 'suite',
    'apt': 'apartment', 'apt.': 'apartment',
    'bldg': 'building', 'bldg.': 'building',
    'fl': 'floor', 'fl.': 'floor', 'flr': 'floor',
    'rm': 'room', 'rm.': 'room',
    'dept': 'department',
    
    # Directional
    'n': 'north', 'n.': 'north',
    's': 'south', 's.': 'south',
    'e': 'east', 'e.': 'east',
    'w': 'west', 'w.': 'west',
    'ne': 'northeast', 'nw': 'northwest',
    'se': 'southeast', 'sw': 'southwest',
    
    # Unit types
    'po': 'po', 'p.o.': 'po',
    'box': 'box',
}

# Indian address abbreviations
INDIA_ABBREVIATIONS = {
    'rd': 'road', 'rd.': 'road',
    'st': 'street', 'st.': 'street',
    'marg': 'marg',
    'nagar': 'nagar', 'ngr': 'nagar',
    'gali': 'gali',
    'mohalla': 'mohalla', 'moh': 'mohalla',
    'chowk': 'chowk', 'chwk': 'chowk',
    'colony': 'colony', 'col': 'colony',
    'sector': 'sector', 'sec': 'sector',
    'block': 'block', 'blk': 'block',
    'phase': 'phase', 'ph': 'phase',
    'dist': 'district', 'distt': 'district',
    'vill': 'village', 'vill.': 'village',
    'kh': 'khasra', 'kh.': 'khasra',
    'h.no': 'house number', 'hno': 'house number',
    'h.no.': 'house number',
    'opp': 'opposite', 'opp.': 'opposite',
    'nr': 'near', 'nr.': 'near',
}

# French address abbreviations
FRENCH_ABBREVIATIONS = {
    'rue': 'rue',
    'av': 'avenue', 'av.': 'avenue',
    'bd': 'boulevard', 'bd.': 'boulevard',
    'pl': 'place', 'pl.': 'place',
    'imp': 'impasse', 'imp.': 'impasse',
    'chem': 'chemin', 'ch': 'chemin',
    'rte': 'route', 'rte.': 'route',
    'allée': 'allee', 'all': 'allee',
    'sq': 'square',
    'quai': 'quai',
    'fbg': 'faubourg',
    'esc': 'escalier',
    'bat': 'batiment', 'bât': 'batiment',
    'appt': 'appartement', 'apt': 'appartement',
    'rez-de-chaussée': 'rdc', 'rdc': 'rdc',
    'cedex': 'cedex',
    'cs': 'cs',
    'bp': 'bp',
}

# Combined abbreviation map (country-agnostic approach per PRD)
ALL_ABBREVIATIONS = {}
ALL_ABBREVIATIONS.update(US_ABBREVIATIONS)
ALL_ABBREVIATIONS.update(INDIA_ABBREVIATIONS)
ALL_ABBREVIATIONS.update(FRENCH_ABBREVIATIONS)

# US state abbreviations to full names
US_STATES = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas',
    'ca': 'california', 'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware',
    'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii', 'id': 'idaho',
    'il': 'illinois', 'in': 'indiana', 'ia': 'iowa', 'ks': 'kansas',
    'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
    'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada',
    'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york',
    'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma',
    'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
    'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah',
    'vt': 'vermont', 'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia',
    'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia',
}

# Indian states
INDIA_STATES = {
    'ap': 'andhra pradesh', 'ar': 'arunachal pradesh', 'as': 'assam',
    'br': 'bihar', 'ct': 'chhattisgarh', 'ga': 'goa', 'gj': 'gujarat',
    'hr': 'haryana', 'hp': 'himachal pradesh', 'jk': 'jammu and kashmir',
    'jh': 'jharkhand', 'ka': 'karnataka', 'kl': 'kerala', 'mp': 'madhya pradesh',
    'mh': 'maharashtra', 'mn': 'manipur', 'ml': 'meghalaya', 'mz': 'mizoram',
    'nl': 'nagaland', 'od': 'odisha', 'or': 'odisha', 'pb': 'punjab',
    'rj': 'rajasthan', 'sk': 'sikkim', 'tn': 'tamil nadu', 'ts': 'telangana',
    'tr': 'tripura', 'up': 'uttar pradesh', 'uk': 'uttarakhand',
    'wb': 'west bengal', 'dl': 'delhi',
}


def normalize_unicode_addr(text: str) -> str:
    """Normalize unicode for address matching."""
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text)
    # Keep all characters but normalize combining marks
    result = []
    for ch in text:
        if unicodedata.category(ch) not in ('Mn',):
            result.append(ch)
    return ''.join(result)


def extract_numbers(text: str) -> list:
    """Extract all numeric components (house numbers, PINs, zip codes)."""
    if not text:
        return []
    return re.findall(r'\b\d+(?:[/-]\d+)*\b', text)


def extract_pin_zip(text: str) -> Optional[str]:
    """Extract PIN/ZIP code from address."""
    if not text:
        return None
    
    # US ZIP (5 digits, optionally +4)
    us_zip = re.search(r'\b(\d{5})(?:-\d{4})?\b', text)
    
    # India PIN (6 digits)
    india_pin = re.search(r'\b(\d{6})\b', text)
    
    # French postal code (5 digits, starting with 0-9)
    # (Same pattern as US ZIP, but context would differentiate)
    
    if india_pin:
        return india_pin.group(1)
    if us_zip:
        return us_zip.group(1)
    return None


def clean_address_punctuation(text: str) -> str:
    """Standardize address punctuation."""
    if not text:
        return ""
    # Remove special characters but keep essential ones
    text = re.sub(r'["\'\[\]{}()\*]+', ' ', text)
    # Normalize ## to #
    text = re.sub(r'#+', '#', text)
    # Replace multiple separators
    text = re.sub(r'[,;]+', ', ', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_address_tokens(text: str) -> str:
    """Normalize address abbreviations."""
    if not text:
        return ""
    tokens = text.lower().split()
    normalized = []
    for t in tokens:
        clean_t = t.rstrip('.,')
        if clean_t in ALL_ABBREVIATIONS:
            normalized.append(ALL_ABBREVIATIONS[clean_t])
        else:
            normalized.append(t.lower())
    return ' '.join(normalized)


def remove_landmarks(text: str) -> str:
    """Remove landmark references (Near X, Opposite Y) — keep for a separate feature."""
    if not text:
        return ""
    # Remove "near X" / "opp X" patterns
    text = re.sub(r'\b(?:near|opp|opposite|behind|beside|next to|adjacent to|in front of)\s+\S+(?:\s+\S+){0,3}', '', text, flags=re.IGNORECASE)
    return text.strip()


def extract_landmarks(text: str) -> str:
    """Extract landmark references for a separate matching feature."""
    if not text:
        return ""
    matches = re.findall(
        r'\b(?:near|opp|opposite|behind|beside|next to|adjacent to|in front of)\s+(\S+(?:\s+\S+){0,3})',
        text, flags=re.IGNORECASE
    )
    return ' '.join(matches)


def normalize_address(address: str) -> dict:
    """
    Full address normalization pipeline.
    Returns dict with multiple normalized forms.
    """
    if not address or (isinstance(address, float) and str(address) == 'nan'):
        return {
            'original': '',
            'cleaned': '',
            'tokens': [],
            'numbers': [],
            'pin_zip': '',
            'landmarks': '',
            'no_landmarks': '',
        }
    
    address = str(address)
    original = address
    
    # Step 1: Unicode normalization
    address = normalize_unicode_addr(address)
    
    # Step 2: Extract PIN/ZIP before cleaning
    pin_zip = extract_pin_zip(address) or ''
    
    # Step 3: Extract numbers
    numbers = extract_numbers(address)
    
    # Step 4: Extract landmarks
    landmarks = extract_landmarks(address)
    
    # Step 5: Clean punctuation
    address = clean_address_punctuation(address)
    
    # Step 6: Normalize abbreviations
    cleaned = normalize_address_tokens(address)
    
    # Step 7: Version without landmarks
    no_landmarks = normalize_address_tokens(remove_landmarks(address))
    
    # Step 8: Tokens
    tokens = cleaned.split()
    
    return {
        'original': original,
        'cleaned': cleaned,
        'tokens': tokens,
        'numbers': numbers,
        'pin_zip': pin_zip,
        'landmarks': landmarks,
        'no_landmarks': no_landmarks,
    }


def normalize_address_simple(address: str) -> str:
    """Simple address normalization for retrieval — single clean string."""
    result = normalize_address(address)
    return result['cleaned']
