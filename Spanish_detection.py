import fitz  # PyMuPDF
import re
import unicodedata
from lingua import Language, LanguageDetectorBuilder
import enchant
from datetime import datetime
import os

def process_spenish(file_path):
    input_pdf = file_path
    filename = os.path.basename(file_path)

    output_pdf = os.path.join(
        os.path.dirname(file_path),
        filename.replace(".pdf", "_highlighted.pdf")
    )

    # -----------------------------
    # Initialize Lingua detector
    # -----------------------------
    languages = [Language.ENGLISH, Language.SPANISH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    english_dict = enchant.Dict("en_US")

    # -----------------------------
    # Clean word
    # -----------------------------
    def clean_word(word):
        return re.sub(r"[^\w]", "", word.lower())

    # -----------------------------
    # Normalize accents
    # -----------------------------
    def normalize(word):
        return "".join(
            c for c in unicodedata.normalize("NFD", word.lower())
            if unicodedata.category(c) != "Mn"
        )

    # -----------------------------
    # Check Spanish characters
    # -----------------------------
    def has_spanish_chars(word):
        return any(c in word.lower() for c in "áéíóúñ")

    # -----------------------------
    # Check Date Format
    # -----------------------------
    def is_date_format(text):
        formats = [
            "%d-%b-%Y",
            "%d/%b/%Y",
        ]
        for fmt in formats:
            try:
                datetime.strptime(text.strip(), fmt)
                return True
            except ValueError:
                continue
        return False

    # -----------------------------
    # Check if word is Spanish
    # -----------------------------
    def is_spanish_word(word):
        clean = clean_word(word)

        # Ignore date
        if is_date_format(word):
            return False

        # Ignore English words
        if clean and len(clean) > 2 and english_dict.check(clean):
            return False

        # Ignore short/empty
        if not clean or len(clean) < 2:
            return False

        # Spanish special characters
        if has_spanish_chars(clean):
            return True

        try:
            confidences = detector.compute_language_confidence_values(clean)

            spanish_score = 0.0
            english_score = 0.0

            for conf in confidences:
                if conf.language == Language.SPANISH:
                    spanish_score = conf.value
                elif conf.language == Language.ENGLISH:
                    english_score = conf.value

            # Threshold logic (UNCHANGED)
            if spanish_score >= 0.65:
                return True

            if (
                spanish_score >= 0.50
                and spanish_score > english_score
                and len(clean) > 2
            ):
                return True

        except Exception as e:
            print("Detection error:", e)

        return False

    # -----------------------------
    # Highlight Spanish words
    # -----------------------------
    def highlight_spanish_words(input_pdf, output_pdf):
        doc = fitz.open(input_pdf)

        for page in doc:
            words = page.get_text("words")

            for w in words:
                word_text = w[4]

                if is_spanish_word(word_text):
                    rect = fitz.Rect(w[:4])
                    highlight = page.add_highlight_annot(rect)
                    highlight.update()

        doc.save(output_pdf)
        doc.close()

        print(f"✅ Highlighted Spanish words saved to: {output_pdf}")

    # ✅ IMPORTANT: Call function
    try:
        highlight_spanish_words(input_pdf, output_pdf)
    except Exception as e:
        print("Processing error:", e)
        return None

    # ✅ Return filename only (Flask expects this)
    return os.path.basename(output_pdf)