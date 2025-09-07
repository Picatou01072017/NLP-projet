import re
import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

class TextPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self,
                 text_columns,
                 spacy_model="fr_core_news_sm",
                 stopwords=None,
                 lemma = True,
                 batch_size=1000,
                 n_process=1,
                 lazy_load=True):

        self.text_columns = text_columns or []
        self.spacy_model = spacy_model
        self.stopwords = stopwords
        self.batch_size = batch_size
        self.lemma = lemma
        self.n_process = n_process
        self.lazy_load = lazy_load
        self._nlp = self._get_model()

    def _get_model(self):
        "Get NLP Model"
        
        try:
            nlp = spacy.load(self.spacy_model, disable=["parser", "ner", "textcat"])
        except OSError:
            from spacy.cli import download
            download(self.spacy_model)
            nlp = spacy.load(self.spacy_model, disable=["parser", "ner", "textcat"])

        return nlp
    
    
    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()
        for col in self.text_columns:
            series = X[col].fillna("").astype(str)
            series = series.apply(self._basic_clean)
            if self.lemma:
              X[col] = self._lemmatize(series)
            else :
              X[col] = series


        return X

    # ---------- utils ----------

    def _basic_clean(self, text: str) -> str:
        # lower + nettoyages légers
        text = text.lower()
        text = re.sub(r"<.*?>", " ", text)         # HTML
        text = re.sub(r"(http|www)\S*", " ", text) # URLs
        text = re.sub(r"\S*@\S*\s*", " ", text)    # emails
        text = re.sub(r"\d+", " ", text)           # chiffres
        text = re.sub(r'[^\w\s]', '', text)        # Caractères spéciaux
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _lemmatize(self, s: pd.Series) -> pd.Series:
        texts = s.tolist()
        out = []

        for doc in self._nlp.pipe(texts, batch_size=self.batch_size, n_process=self.n_process):
            lemmas = []
            for t in doc:
                if t.is_space or t.is_punct:
                    continue
                lemma = (t.lemma_ or "").strip().lower()
                if not lemma:
                    continue
                # garder seulement des lemmes "mots"
                if not lemma.replace("’","'").replace("-", "").isalpha():
                    continue
                if self.stopwords is not None and lemma in self.stopwords:
                    continue
                lemmas.append(lemma)
            out.append(" ".join(lemmas))
        return pd.Series(out, index=s.index)