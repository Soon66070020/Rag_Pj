"""Thai text processing utilities using pythainlp.

This module provides Thai-specific text normalization, tokenization, and
keyword extraction for both queries and knowledge base documents.

The system primarily supports Thai language with mixed Thai-English documents.
"""

import re
import unicodedata
from typing import List, Dict, Any
from collections import Counter


def correct_ocr_errors(text: str) -> str:
    """Correct common PDF/OCR extraction errors in Thai text using regex.

    Fixes common issues found in Thai PDF extraction:
    - Numbers replacing Thai characters (e.g., "2" instead of "อ")
    - OCR errors like "โา" instead of "า"
    - Duplicate vowels (e.g., "าา" instead of "า")
    - Unicode private use area characters
    - Replacement characters (U+FFFD)
    - Abnormal spacing between Thai characters

    Args:
        text: Text extracted from PDF/OCR.

    Returns:
        Corrected text with common extraction errors fixed.

    Example:
        >>> correct_ocr_errors("น้ำ2ัดลม")
        'น้ำอัดลม'
        >>> correct_ocr_errors("จ ำ หน่ าย")
        'จำหน่าย'
    """
    if not text:
        return text

    # 1. Single character substitutions (common OCR errors)
    single_char_substitutions = {
        '2': 'อ',  # "น้ำ2ัดลม" → "น้ำอัดลม"
    }
    text = text.translate(str.maketrans(single_char_substitutions))

    # 2. Fix "โา" OCR error → "า"
    text = re.sub(r"โา", "า", text)  # เช่น "จโาหน่าย" → "จำหน่าย"

    # 3. Fix duplicate vowels
    text = re.sub(r"าา", "า", text)  # "าา" → "า"
    text = re.sub(r"ำา", "ำ", text)  # "ำา" → "ำ" (เช่น "ทำา" → "ทำ")

    # 4. Remove Unicode private use area characters (often from PDF fonts)
    text = re.sub(r"[\uf700-\uf74f]", "", text)

    # 5. Remove replacement characters
    text = re.sub(r"[�\ufffd]", "", text)

    # 6. Normalize newlines, tabs, and multiple spaces
    text = re.sub(r"[\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    # 7. Remove abnormal spacing between Thai characters
    # Range [ก-์] includes consonants, vowels, and tone marks
    # "จ ำ" → "จำ", "ท ำ" → "ทำ", "ค ว า ม" → "ความ"
    text = re.sub(r"(?<=[ก-์])\s(?=[ก-์])", "", text)

    return text.strip()


def normalize_thai_text(text: str) -> str:
    """Normalize Thai text for processing.

    Performs comprehensive text normalization for Thai language:
    - Correct OCR/PDF extraction errors
    - Unicode NFC normalization (preserves Thai characters)
    - pythainlp text normalization
    - Remove excessive whitespace
    - Handle mixed Thai-English text

    Args:
        text: Input Thai or mixed Thai-English text.

    Returns:
        Normalized text ready for further processing.

    Example:
        >>> text = "  ฉันควร  กิน อะไร  "
        >>> normalize_thai_text(text)
        'ฉันควร กิน อะไร'
    """
    # First, correct OCR/PDF extraction errors
    text = correct_ocr_errors(text)

    # Unicode normalization (NFC preserves Thai characters)
    text = unicodedata.normalize('NFC', text)

    # pythainlp normalization
    try:
        from pythainlp.util import normalize as thai_normalize
        text = thai_normalize(text)
    except ImportError:
        # Fall back to basic normalization if pythainlp not available
        pass

    # Whitespace cleanup - remove extra spaces while preserving Thai spacing
    text = ' '.join(text.split())

    return text


def tokenize_thai(text: str, engine: str = "newmm") -> List[str]:
    """Tokenize Thai text into words.

    Uses pythainlp's newmm (maximum matching) algorithm by default,
    which works well for medical terminology and mixed language text.

    Args:
        text: Thai text to tokenize.
        engine: Tokenization engine ("newmm", "longest", "icu", "attacut").
            - "newmm": Maximum matching (default, fast and accurate)
            - "longest": Longest matching
            - "icu": ICU library tokenizer
            - "attacut": Deep learning-based (more accurate but slower)

    Returns:
        List of Thai words/tokens.

    Example:
        >>> tokenize_thai("ฉันควรกินอะไรหลังผ่าตัด")
        ['ฉัน', 'ควร', 'กิน', 'อะไร', 'หลัง', 'ผ่าตัด']
    """
    try:
        from pythainlp import word_tokenize
        tokens = word_tokenize(text, engine=engine)
        return tokens
    except ImportError:
        # Fallback: return text split by spaces if pythainlp not available
        return text.split()


def extract_thai_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract important Thai keywords from text.

    Tokenizes Thai text, removes stopwords, and returns the most
    frequent keywords. Useful for category inference and understanding
    query intent.

    Args:
        text: Thai text to extract keywords from.
        top_n: Number of top keywords to extract.

    Returns:
        List of important Thai keywords sorted by frequency.

    Example:
        >>> text = "ฉันควรกินอาหารอะไรหลังผ่าตัดฟัน อาหารอ่อนดีไหม"
        >>> extract_thai_keywords(text, top_n=3)
        ['อาหาร', 'ผ่าตัด', 'ฟัน']
    """
    # Tokenize text
    tokens = tokenize_thai(text)

    # Remove stopwords (common Thai words that don't carry much meaning)
    try:
        from pythainlp.corpus import thai_stopwords
        stopwords = thai_stopwords()
        keywords = [t for t in tokens if t not in stopwords and len(t) > 1]
    except ImportError:
        # Basic filtering if pythainlp not available
        basic_stopwords = {'ที่', 'นี้', 'นั้น', 'ของ', 'มี', 'เป็น', 'ได้', 'จะ', 'ใน', 'ไหม'}
        keywords = [t for t in tokens if t not in basic_stopwords and len(t) > 1]

    # Count keyword frequencies
    keyword_counts = Counter(keywords)

    # Return top N keywords
    return [kw for kw, count in keyword_counts.most_common(top_n)]


def infer_category_from_thai_keywords(query: str) -> str:
    """Infer query category based on Thai medical keywords.

    Analyzes Thai keywords to determine the most appropriate category
    for metadata filtering in retrieval.

    Categories:
        - "Emergency": Urgent medical situations
        - "Medication": Medicine and drug-related queries
        - "Nutrition": Food and eating-related queries
        - "Post-op Care": General post-operative care (default)

    Args:
        query: Thai query text.

    Returns:
        Inferred category string.

    Example:
        >>> infer_category_from_thai_keywords("มีเลือดออกมากมาก")
        'Emergency'
        >>> infer_category_from_thai_keywords("กินยาแก้ปวด")
        'Medication'
        >>> infer_category_from_thai_keywords("กินอาหารอะไรได้บ้าง")
        'Nutrition'
    """
    # Extract keywords from query
    keywords = extract_thai_keywords(query, top_n=10)
    query_lower = query.lower()

    # Define Thai medical keyword sets for each category
    medication_keywords = [
        'ยา', 'แก้ปวด', 'ปฏิชีวนะ', 'ยาปฏิชีวนะ', 'แก้อักเสบ',
        'ยาแก้', 'ทาน', 'ยากิน', 'ยาเม็ด', 'แคปซูล'
    ]

    emergency_keywords = [
        'เลือดออก', 'ปวดมาก', 'บวมมาก', 'ไข้สูง', 'อันตราย',
        'เจ็บมาก', 'ทนไม่ได้', 'รุนแรง', 'ฉุกเฉิน', 'ด่วน'
    ]

    nutrition_keywords = [
        'อาหาร', 'กิน', 'ดื่ม', 'รับประทาน', 'เครื่องดื่ม',
        'ทาน', 'กินข้าว', 'น้ำ', 'เมนู', 'งด'
    ]

    # Check keywords for emergency first (highest priority)
    if any(kw in keywords or kw in query_lower for kw in emergency_keywords):
        return "Emergency"

    # Check for medication
    elif any(kw in keywords or kw in query_lower for kw in medication_keywords):
        return "Medication"

    # Check for nutrition
    elif any(kw in keywords or kw in query_lower for kw in nutrition_keywords):
        return "Nutrition"

    # Default to general post-op care
    else:
        return "Post-op Care"


def infer_categories_from_taxonomy(text: str, taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    """Infer multiple categories from text using a 2-level taxonomy.

    Scans text for keyword matches at both L1 (main category) and L2
    (subcategory) levels. A single chunk can match multiple categories.

    Args:
        text: Text content to classify.
        taxonomy: Taxonomy dict loaded from config/taxonomy.yaml.
            Expected structure: {category: {keywords_th: [...], subcategories: {name: {keywords_th: [...]}}}}

    Returns:
        Dict with:
            - primary_category: L1 with most keyword matches (fallback "Post-op Care")
            - categories: List of all matched L1 categories
            - subcategories: List of all matched L2 subcategory names
    """
    tokens = set(tokenize_thai(text.lower()))
    text_lower = text.lower()

    l1_match_counts: Dict[str, int] = {}
    matched_categories: set = set()
    matched_subcategories: set = set()

    for l1_name, l1_data in taxonomy.items():
        l1_name_str = str(l1_name)
        l1_count = 0

        # Check L1 keywords
        l1_keywords: List[str] = [str(k) for k in l1_data.get('keywords_th', [])]
        for kw in l1_keywords:
            if kw in tokens or kw in text_lower:
                l1_count += 1

        # Check L2 subcategories
        subcategories = l1_data.get('subcategories', {})
        for l2_name, l2_data in subcategories.items():
            l2_name_str = str(l2_name)
            l2_keywords: List[str] = [str(k) for k in l2_data.get('keywords_th', [])]
            for kw in l2_keywords:
                if kw in tokens or kw in text_lower:
                    l1_count += 1
                    matched_categories.add(l1_name_str)
                    matched_subcategories.add(l2_name_str)
                    break  # Count each L2 once

        if l1_count > 0:
            l1_match_counts[l1_name_str] = l1_count
            matched_categories.add(l1_name_str)

    # Determine primary category (highest match count)
    if l1_match_counts:
        primary = max(l1_match_counts.keys(), key=lambda k: l1_match_counts[k])
    else:
        primary = "Post-op Care"
        matched_categories.add("Post-op Care")

    return {
        "primary_category": primary,
        "categories": sorted(matched_categories),
        "subcategories": sorted(matched_subcategories),
    }


def clean_thai_text_for_embedding(text: str) -> str:
    """Clean and prepare Thai text for embedding generation.

    Performs all necessary preprocessing steps to prepare Thai text
    for BGE-M3 embedding generation:
    - Normalization
    - Remove excessive punctuation
    - Preserve Thai-English mixed content

    Args:
        text: Raw Thai or mixed text.

    Returns:
        Cleaned text ready for embedding.

    Example:
        >>> clean_thai_text_for_embedding("ฉันควรกิน!!!อะไร???")
        'ฉันควรกิน อะไร'
    """
    # Normalize text
    text = normalize_thai_text(text)

    # Remove excessive punctuation but keep some for context
    import re
    # Replace multiple punctuation with single
    text = re.sub(r'([!?।]+)', r'\1', text)
    text = re.sub(r'([.]+)', '.', text)

    # Remove extra whitespace again
    text = ' '.join(text.split())

    return text
