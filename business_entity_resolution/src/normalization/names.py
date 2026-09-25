"""
Business name normalization — information-preserving, not lossy.
Handles abbreviations, legal suffixes, punctuation, transliterations.
"""
import re
import unicodedata
from typing import Optional


# Legal suffix normalization map — map variations to canonical forms
LEGAL_SUFFIXES = {
    # Corporation
    'corp': 'corporation', 'corp.': 'corporation', 'corpn': 'corporation',
    # Incorporated
    'inc': 'incorporated', 'inc.': 'incorporated', 'incorp': 'incorporated',
    # Limited
    'ltd': 'limited', 'ltd.': 'limited', 'ltda': 'limited',
    # Limited Liability Company
    'llc': 'llc', 'l.l.c.': 'llc', 'l.l.c': 'llc',
    # Private Limited
    'pvt': 'private', 'pvt.': 'private', 'pvt.ltd': 'private limited',
    'pvt.ltd.': 'private limited', 'pvtltd': 'private limited',
    'private limited': 'private limited', 'privatelimited': 'private limited',
    # Public Limited Company
    'plc': 'plc', 'p.l.c.': 'plc',
    # Company
    'co': 'company', 'co.': 'company', 'comp': 'company',
    # Group
    'grp': 'group', 'grp.': 'group',
    # Associates
    'assoc': 'associates', 'assocs': 'associates',
    # Brothers
    'bros': 'brothers', 'bros.': 'brothers',
    # International
    'intl': 'international', "int'l": 'international', 'intl.': 'international',
    # Services
    'svcs': 'services', 'svc': 'service', 'svcs.': 'services',
    # Industries
    'ind': 'industries', 'inds': 'industries',
    # Enterprise
    'ent': 'enterprise', 'ent.': 'enterprise', 'enterp': 'enterprise',
    # Foundation
    'fdn': 'foundation', 'fdn.': 'foundation', 'fndn': 'foundation',
    # Department
    'dept': 'department', 'dept.': 'department',
    # Manufacturing
    'mfg': 'manufacturing', 'mfg.': 'manufacturing',
    # Solutions
    'soln': 'solutions', 'solns': 'solutions',
    # Technologies
    'tech': 'technologies', 'techn': 'technologies',
    # Laboratories
    'lab': 'laboratories', 'labs': 'laboratories',
    # French equivalents
    'sarl': 'sarl', 's.a.r.l.': 'sarl', 's.a.r.l': 'sarl',
    'sa': 'sa', 's.a.': 'sa', 's.a': 'sa',
    'sas': 'sas', 's.a.s.': 'sas', 's.a.s': 'sas',
    'eurl': 'eurl', 'e.u.r.l.': 'eurl',
    'gie': 'gie', 'g.i.e.': 'gie',
    'sci': 'sci', 's.c.i.': 'sci',
}

# Words to treat as equivalent
WORD_EQUIVALENTS = {
    '&': 'and',
    'n': 'and',  # Hindi 'n' sometimes used for 'and'
    '+': 'and',
    '@': 'at',
}


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters — NFKD decomposition, strip combining marks for comparison."""
    if not text:
        return ""
    # NFKD normalization — decomposes characters
    text = unicodedata.normalize('NFKD', text)
    # Remove combining marks (accents) for matching but keep base characters
    result = []
    for ch in text:
        if unicodedata.category(ch) not in ('Mn',):  # Mn = Mark, Nonspacing
            result.append(ch)
    return ''.join(result)


def clean_punctuation(text: str) -> str:
    """Standardize punctuation for matching."""
    if not text:
        return ""
    # Replace common separators with space
    text = re.sub(r'[/\\|,;:]+', ' ', text)
    # Remove quotes, brackets (keep hyphens and periods for abbreviations)
    text = re.sub(r'["\'\[\]{}()\*#]+', ' ', text)
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_dba(name: str) -> tuple:
    """
    Extract DBA (doing business as) names.
    Returns (primary_name, dba_name_or_None).
    """
    if not name:
        return ("", None)
    
    # Common DBA patterns
    dba_patterns = [
        r'\bdba\b',
        r'\bd/b/a\b',
        r'\bdoing business as\b',
        r'\baka\b',
        r'\ba\.k\.a\.\b',
        r'\btrading as\b',
        r'\bt/a\b',
    ]
    
    for pattern in dba_patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            primary = name[:match.start()].strip()
            dba = name[match.end():].strip()
            return (primary, dba)
    
    # Check for pipe separator (seen in data: "COMPANY | www.company.com")
    if '|' in name:
        parts = name.split('|', 1)
        return (parts[0].strip(), parts[1].strip())
    
    return (name, None)


def normalize_legal_suffix(token: str) -> str:
    """Normalize a single token if it's a legal suffix."""
    lower = token.lower().rstrip('.,')
    if lower in LEGAL_SUFFIXES:
        return LEGAL_SUFFIXES[lower]
    return token.lower()


def extract_legal_suffix(name: str) -> tuple:
    """
    Extract and normalize legal suffixes from the name.
    Returns (name_without_suffix, normalized_suffix_or_empty).
    """
    if not name:
        return ("", "")
    
    tokens = name.split()
    if not tokens:
        return ("", "")
    
    # Check last 1-3 tokens for legal suffix patterns
    suffix_parts = []
    remaining = list(tokens)
    
    for i in range(min(3, len(tokens))):
        token = remaining[-1].lower().rstrip('.,')
        if token in LEGAL_SUFFIXES:
            suffix_parts.insert(0, LEGAL_SUFFIXES[token])
            remaining.pop()
        else:
            break
    
    name_part = ' '.join(remaining)
    suffix_part = ' '.join(suffix_parts)
    
    return (name_part, suffix_part)


def remove_website(name: str) -> str:
    """Remove website URLs from names."""
    # Remove http(s) URLs
    name = re.sub(r'https?://\S+', '', name)
    # Remove www. URLs  
    name = re.sub(r'www\.\S+', '', name)
    # Remove .com/.org/.net etc standalone
    name = re.sub(r'\b\w+\.(com|org|net|co\.in|in|edu)\b', '', name, flags=re.IGNORECASE)
    return name.strip()


def normalize_name(name: str) -> dict:
    """
    Full name normalization pipeline.
    Returns a dict with multiple normalized forms for different matching strategies.
    """
    if not name or (isinstance(name, float) and str(name) == 'nan'):
        return {
            'original': '',
            'cleaned': '',
            'name_only': '',
            'legal_suffix': '',
            'dba_name': '',
            'tokens': [],
            'acronym': '',
        }
    
    name = str(name)
    original = name
    
    # Step 1: Unicode normalization
    name = normalize_unicode(name)
    
    # Step 2: Remove websites
    name = remove_website(name)
    
    # Step 3: Extract DBA
    primary, dba = extract_dba(name)
    if primary:
        name = primary
    
    # Step 4: Clean punctuation
    name = clean_punctuation(name)
    
    # Step 5: Normalize word equivalents
    tokens = name.split()
    normalized_tokens = []
    for t in tokens:
        lower = t.lower()
        if lower in WORD_EQUIVALENTS:
            normalized_tokens.append(WORD_EQUIVALENTS[lower])
        else:
            normalized_tokens.append(lower)
    
    # Step 6: Extract legal suffix
    cleaned = ' '.join(normalized_tokens)
    name_only, legal_suffix = extract_legal_suffix(cleaned)
    
    # Step 7: Generate tokens (for token-level matching)
    name_tokens = name_only.split()
    
    # Step 8: Generate acronym (first letter of each significant word)
    acronym = ''.join(t[0] for t in name_tokens if len(t) > 1 and t.isalpha())
    
    return {
        'original': original,
        'cleaned': cleaned,
        'name_only': name_only,
        'legal_suffix': legal_suffix,
        'dba_name': dba or '',
        'tokens': name_tokens,
        'acronym': acronym,
    }


def normalize_name_simple(name: str) -> str:
    """
    Simple name normalization for retrieval — returns a single clean string.
    Used for building indices.
    """
    result = normalize_name(name)
    return result['cleaned']
