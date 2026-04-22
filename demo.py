# translation.py
from collections import defaultdict
import re
from datetime import datetime

import fitz
import requests
from lingua import LanguageDetectorBuilder, Language

API_KEY = "API_Key"
BASE_URL = "https://translation.googleapis.com/language/translate/v2"

_lingua_detector = None

# ─────────────────────────────────────────
# REGEX PATTERNS
# ─────────────────────────────────────────
_CASE_ID_RE  = re.compile(r'^[A-Z]{2,3}-[A-Z0-9]+-[A-Z0-9]+$', re.IGNORECASE)
_DOSAGE_RE   = re.compile(
    r'^(mg|mcg|ug|ml|auc|kg|g|mmol|nmol'
    r'|milligram|microgram|liter|litre'
    r'|tablet|capsule|vial|ampoule'
    r'|dose|unit|units|interval|days|weeks|months|years)'
    r'[\s\(\)s]*$',
    re.IGNORECASE
)
_ORG_NAME_RE    = re.compile(r'^[A-Z]{2,6}\s*[-–]\s*.+$')
MEDDRA_CLEAN_RE = re.compile(
    r"\(?['\"]?\b(?:MedDRA|code|version)\b['\"]?\)?[:']?",
    re.IGNORECASE
)
IGNORE_TOKENS = {
    "lipase", "glucose", "albumin", "protein", "plasma",
    "insulin", "cholesterol", "bilirubin", "carcinoma", "paxlovid", "pembrolizumab",
    "patient", "paclitaxel",
}

# ─────────────────────────────────────────
# LINGUA DETECTOR
# ─────────────────────────────────────────
def get_lingua_detector():
    global _lingua_detector
    if _lingua_detector is None:
        _lingua_detector = (
            LanguageDetectorBuilder
            .from_languages(Language.FRENCH, Language.ENGLISH)
            .with_minimum_relative_distance(0.11)
            .build()
        )
    return _lingua_detector


# ─────────────────────────────────────────
# STEP 1 — Document Parsing + BBox
# ─────────────────────────────────────────
def extract_words_with_bbox(pdf_path):
    doc = fitz.open(pdf_path)
    words_data = []
    for page_num, page in enumerate(doc):
        for w in page.get_text("words"):
            words_data.append({
                "text": w[4],
                "bbox": (w[0], w[1], w[2], w[3]),
                "page": page_num
            })
    doc.close()
    return words_data


# ─────────────────────────────────────────
# STEP 2 — Pre-filter
# ─────────────────────────────────────────
def is_date_format(text):
    for fmt in ["%d-%b-%Y", "%d/%b/%Y"]:
        try:
            datetime.strptime(text.strip(), fmt)
            return True
        except ValueError:
            continue
    return False


def pre_filter_words(words_data):
    skip_words = []
    api_words  = []
    lingua_detector = get_lingua_detector()  # init once

    for word in words_data:
        text = word["text"]

        # Empty or ICH prefix
        if not text or text.startswith("ICH"):
            skip_words.append({**word, "language": "en", "confidence": 0.99, "isReliable": True, "source": "pre-filter"})
            continue

        # Single accented char → send to API
        if len(text) == 1 and not text.isascii():
            api_words.append(word)
            continue

        # Single plain char or digits/symbols only
        if len(text) <= 1 or not any(c.isalpha() for c in text):
            skip_words.append({**word, "language": "en", "confidence": 0.99, "isReliable": True, "source": "pre-filter"})
            continue

        # Regex filters
        if _DOSAGE_RE.match(text) or MEDDRA_CLEAN_RE.match(text):
            skip_words.append({**word, "language": "en", "confidence": 0.99, "isReliable": True, "source": "regex"})
            continue

        if _CASE_ID_RE.match(text) or text.lower().strip() in IGNORE_TOKENS:
            skip_words.append({**word, "language": "en", "confidence": 0.99, "isReliable": True, "source": "regex"})
            continue

        if _ORG_NAME_RE.match(text) or is_date_format(text):
            skip_words.append({**word, "language": "en", "confidence": 0.99, "isReliable": True, "source": "regex"})
            continue

        # Lingua detection
        lang = lingua_detector.detect_language_of(text)
        if lang is None:
            api_words.append(word)
            continue

        conf = lingua_detector.compute_language_confidence(text, lang)

        if lang.iso_code_639_1.name.lower() == "en":
            skip_words.append({**word, "language": "en", "confidence": float(conf), "isReliable": True, "source": "lingua"})
        else:
            api_words.append(word)

    return skip_words, api_words


# ─────────────────────────────────────────
# STEP 3 — Batch Detect via Google API
# ─────────────────────────────────────────
BATCH_SIZE = 50

def batch_detect_language(texts):
    try:
        response = requests.post(
            f"{BASE_URL}/detect",
            params={"key": API_KEY},
            json={"q": texts}
        )
        result = response.json()

        if "error" in result:
            raise Exception(f"API Error {result['error']['code']}: {result['error']['message']}")

        return [
            {
                "language":   d[0]["language"],
                "confidence": d[0]["confidence"],
                "isReliable": d[0]["isReliable"]
            }
            for d in result["data"]["detections"]
        ]

    except Exception as e:
        print(f"Batch detection failed: {e}")
        return [{"language": "unknown", "confidence": 0.0, "isReliable": False}] * len(texts)


def tag_words_with_language(api_words):
    tagged = []
    total  = len(api_words)

    print(f"\n   📤 Words going to Google API: {total}")  # ✅ API word count

    for i in range(0, total, BATCH_SIZE):
        batch      = api_words[i:i + BATCH_SIZE]
        texts      = [w["text"] for w in batch]
        detections = batch_detect_language(texts)

        for word, detection in zip(batch, detections):
            # ✅ Only keep foreign (non-English) words
            if detection["language"] != "en":
                tagged.append({
                    **word,
                    "language":   detection["language"],
                    "confidence": detection["confidence"],
                    "isReliable": detection["isReliable"],
                    "source":     "google-api"
                })

    print(f"   🌍 Foreign words found: {len(tagged)} / {total}")  # ✅ foreign vs total
    return tagged


# ─────────────────────────────────────────
# STEP 4 — Highlight Foreign Words Only
# ─────────────────────────────────────────
TranslatedDict = {
    "tous":          "all",
    "les":           "the",
    "jours":         "days",
    "KEYTRUDA":      "pembrolizumab brand name",
    "posologie":     "dosage",
    "voie":          "route",
    "orale":         "oral",
    "comprimé":      "tablet",
    "perfusion":     "infusion",
    "semaines":      "weeks",
    "traitement":    "treatment",
    "patients":      "patients",
    "dose":          "dose",
    "J1":            "Day 1",
    "J8":            "Day 8",
    "J15":           "Day 15",
    "négatif":       "negative",
    "segnalazioni":  "reports",
    "perchè":        "because",
    "riferimento":   "reference",
    "cancro":        "cancer",
}

def highlight_words(tagged_words, input_pdf, output_pdf):
    doc = fitz.open(input_pdf)

    translated_data     = defaultdict(list)
    non_translated_data = defaultdict(list)

    # Normalize lookup dict
    translated_dict_normalized = {
        k.strip().lower(): v for k, v in TranslatedDict.items()
    }

    for word in tagged_words:
        page_no = word["page"]
        page    = doc[page_no]
        bbox    = fitz.Rect(word["bbox"])
        annot   = page.add_rect_annot(bbox)

        word_lookup = word["text"].strip().lower()

        if word_lookup in translated_dict_normalized:
            translation = translated_dict_normalized[word_lookup]
            annot.set_colors(stroke=(1, 1, 0))           # 🟡 Yellow — translated
            annot.set_info(content=f"{translation}")
            translated_data[page_no].append({
                "word":        word["text"],
                "translation": translation,
                "bbox":        word["bbox"],
                "language":    word["language"],
                "source_type": word.get("source", "google-api"),
            })
        else:
            annot.set_colors(stroke=(1, 0, 0))           # 🔴 Red — no translation
            non_translated_data[page_no].append({
                "word":        word["text"],
                "translation": "",
                "bbox":        word["bbox"],
                "language":    word["language"],
                "source_type": word.get("source", "google-api"),
            })

        annot.update()

    doc.save(output_pdf)
    doc.close()

    print(f"\nHighlighted PDF saved → {output_pdf}")
    print(f"  🟡 Yellow (translated)    : {sum(len(v) for v in translated_data.values())} words")
    print(f"  🔴 Red    (not translated) : {sum(len(v) for v in non_translated_data.values())} words")


# ─────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────
def run_pipeline(pdf_path):
    print("=" * 60)
    print(f"📄 Processing: {pdf_path}")
    print("=" * 60)

    print("\n🔍 Step 1: Extracting words + BBox...")
    words_data = extract_words_with_bbox(pdf_path)
    print(f"   Found {len(words_data)} words")

    print("\n⚡ Step 2: Pre-filtering...")
    skip_words, api_words = pre_filter_words(words_data)
    print(f"   ✅ Skipped (English/regex) : {len(skip_words)} words")
    print("skipppp words", skip_words)
    print(f"   📤 Sending to API          : {len(api_words)} words")
    print("api words", api_words)

    print("\n🌐 Step 3: Detecting languages via Google API...")
    tagged_api_words = tag_words_with_language(api_words)
    print("tagged words", tagged_api_words)

    print("\n🎨 Step 4: Highlighting foreign words...")
    highlight_words(
        tagged_api_words,
        input_pdf=pdf_path,
        output_pdf=r"D:\TMS\output_highlighted.pdf"
    )

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    run_pipeline(r"D:\TMS\static\French\report_03_FR.pdf")