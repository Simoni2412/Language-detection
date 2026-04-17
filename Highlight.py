import fitz
from collections import defaultdict

# from markdown_it.rules_block import reference
# # #
from lang_detection import (
    process_file,
    detect_document_language,
    extract_foreign_content,
    segment_all_foreign_records,
    enrich_segments_with_word_bboxes, keep_primary_language_only,
)
#
from main import (
    process_file,
    detect_document_language,
    extract_foreign_content,
    segment_all_foreign_records,
    enrich_segments_with_word_bboxes,
)

#
# from demo import (
#     process_file,
#     detect_document_language,
#     extract_foreign_content,
#     segment_all_foreign_records,
#     enrich_segments_with_word_bboxes,
# )


# ---------------------------------------------------------------------------
# Translation dictionary  (replace with DB lookup when ready)
# ---------------------------------------------------------------------------

TranslatedDict = {
    "tous":         "all",
    "les":          "the",
    "jours":        "days",
    "KEYTRUDA":     "pembrolizumab brand name",
    "posologie":    "dosage",
    "voie":         "route",
    "orale":        "oral",
    "comprimé":     "tablet",
    "perfusion":    "infusion",
    "semaines":     "weeks",
    "traitement":   "treatment",
    "patients":     "patients",
    "dose":         "dose",
    "J1":           "Day 1",
    "J8":           "Day 8",
    "J15":          "Day 15",
    "négatif" : "negative",
    "segnalazioni" : "reports",
    "perchè" : "because",
    "riferimento" : "reference",
    "cancro" : "cancer",
    # "recuperada" :"Recovered",
    # "resuelto":"Resolved"


}

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
# #
input_pdf  = r"D:\TMS\static\FR-AFSSAPS-BX2024002066.pdf"
output_pdf = r"D:\TMS\static\FR-AFSSAPS-BX2024002066_highlighted.pdf"
# # #
# input_pdf  = r"D:\TMS\static\ES-AEMPS-325884.pdf"
# output_pdf = r"D:\TMS\static\ES-AEMPS-325884_highlighted.pdf"

# input_pdf  = r"D:\TMS\static\pmc_9922741 (2).pdf"
# output_pdf = r"D:\TMS\static\pmc_9922741 (2)_highlighted.pdf"
model_path = r"D:\TMS\Models\lid.176.bin"

# ---------------------------------------------------------------------------
# Build segments from text_extraction pipeline
# ---------------------------------------------------------------------------

parsed        = process_file(input_pdf)
# document_lang = detect_document_language(parsed, model_path)
# print(f"Detected document language : {document_lang}")

foreign_records = extract_foreign_content(parsed, base_language="en")
print(f"Foreign phrases found      : {len(foreign_records)}")
# for r in foreign_records:
#     print("BEFORE SEG:", r)

foreign_records, top_lang, lang_counts = keep_primary_language_only(foreign_records)
print("Top language:", top_lang)
print("Counts:", dict(lang_counts))
print("After filter:", len(foreign_records))
# for r in foreign_records:
#     print("BEFORE SEG:", r)

segments = segment_all_foreign_records(foreign_records)
segments = enrich_segments_with_word_bboxes(segments, input_pdf)
print(f"Word-level segments        : {len(segments)}")

# ---------------------------------------------------------------------------
# Highlight
# ---------------------------------------------------------------------------

doc = fitz.open(input_pdf)

translated_data     = defaultdict(list)   # page_no → [...]
non_translated_data = defaultdict(list)   # page_no → [...]

for seg in segments:
    # Skip segments without a word-level bbox (e.g. paragraph/txt sources)
    if seg.get("x0") is None:
        continue

    page_no = seg["page_no"]
    word    = seg["word"]
    rect    = fitz.Rect(seg["x0"], seg["y0"], seg["x1"], seg["y1"])
    page    = doc[page_no - 1]

    annot = page.add_highlight_annot(rect)

    translated_dict_normalized = {
        k.lower(): v for k, v in TranslatedDict.items()
    }

    word_lookup = word.lower()

    if word_lookup in translated_dict_normalized:
        translation = translated_dict_normalized[word_lookup]
        annot.set_colors(stroke=(1, 1, 0))   # Yellow — translation found
        annot.set_info(content=f"{translation}")
        translated_data[page_no].append({
            "word":        word,
            "translation": translation,
            "bbox":        (seg["x0"], seg["y0"], seg["x1"], seg["y1"]),
            "language":    seg["source_language"],
            "source_type": seg["source_type"],
        })
    else:
        annot.set_colors(stroke=(1, 0, 0))   # Red — no translation yet
        non_translated_data[page_no].append({
            "word":        word,
            "translation": "",
            "bbox":        (seg["x0"], seg["y0"], seg["x1"], seg["y1"]),
            "language":    seg["source_language"],
            "source_type": seg["source_type"],
        })

    annot.update()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

doc.save(output_pdf)
doc.close()

print(f"\nHighlighted PDF saved → {output_pdf}")
print(f"  Yellow (translated)     : {sum(len(v) for v in translated_data.values())} words")
print(f"  Red    (not translated) : {sum(len(v) for v in non_translated_data.values())} words")