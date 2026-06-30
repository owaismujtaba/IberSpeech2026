"""
Semantic-category label map for the Words experiment.

`Sem Categories.xlsx` (sheet 'Seleccion final') lists 6 semantic categories, each
with 5 high-frequency and 5 low-frequency real Spanish words (60 words total).
The sheet is laid out as repeating blocks:

    <Category name>      <NaN>
    high_word_1          low_word_1
    ...                  ...
    high_word_5          low_word_5

i.e. a category header is a row whose 'High frequency' cell holds the category name
and whose 'Low frequency' cell is empty. Word event labels in the EEG annotations
match these words exactly after normalising with ``.strip().lower()``.
"""
import pandas as pd

# Stable, sorted category order → category index is reproducible across runs.
CATEGORIES = sorted([
    "Animals",
    "Body parts",
    "Clothing",
    "Food",
    "Kitchen utensils",
    "Music instruments",
])

_CATEGORY_SET = set(CATEGORIES)


def _norm(word) -> str:
    """Normalise a word/label to its canonical form for matching."""
    return str(word).strip().lower()


def load_word_categories(xlsx: str = "Sem Categories.xlsx") -> dict:
    """
    Parse the semantic-categories spreadsheet.

    Returns a dict ``{normalised_word: category}`` with 60 entries.
    """
    df = pd.read_excel(xlsx, sheet_name="Seleccion final")
    hi_col, lo_col = df.columns[0], df.columns[1]

    word_to_cat: dict = {}
    current_cat = None
    for _, row in df.iterrows():
        hi = row[hi_col]
        lo = row[lo_col]
        hi_is_str = isinstance(hi, str) and hi.strip() != ""
        lo_is_str = isinstance(lo, str) and lo.strip() != ""

        # Category header: a non-empty high cell, empty low cell, matching a known name.
        if hi_is_str and not lo_is_str and hi.strip() in _CATEGORY_SET:
            current_cat = hi.strip()
            continue

        if current_cat is None:
            continue
        if hi_is_str:
            word_to_cat[_norm(hi)] = current_cat
        if lo_is_str:
            word_to_cat[_norm(lo)] = current_cat

    return word_to_cat
