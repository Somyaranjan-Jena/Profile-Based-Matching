import re

try:
    import spacy
except ModuleNotFoundError:
    spacy = None

if spacy is not None:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
else:
    nlp = None

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def clean_text(text):
    if text is None:
        return ""

    text = text.lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if nlp is None:
        return " ".join(
            word
            for word in text.split()
            if word not in STOP_WORDS
        )

    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
    ]

    return " ".join(tokens)


if __name__ == "__main__":

    sample = """
    Data Scientist with 5 years of experience in Machine Learning,
    Python, SQL and NLP. I enjoy mentoring juniors and solving
    real-world healthcare problems.
    """

    print(clean_text(sample))
