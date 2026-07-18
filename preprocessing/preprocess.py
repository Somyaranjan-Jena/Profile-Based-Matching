import re
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    if text is None:
        return ""

    text = text.lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

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