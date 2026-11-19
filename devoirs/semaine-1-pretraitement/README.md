# Semaine 1 — Prétraitement de texte en français

Chaîne de prétraitement classique appliquée à un court texte en français.

| Étape | Outil |
|---|---|
| Tokenisation | spaCy |
| Suppression des mots vides | liste de stopwords français |
| Racinisation (stemming) | NLTK `SnowballStemmer("french")` |
| Pondération des termes | scikit-learn `TfidfVectorizer` |

**Notebook :** [`pretraitement_texte.ipynb`](pretraitement_texte.ipynb) (exécuté, avec résultats)

**Limite :** l'étape « lemmatisation » réutilise les tokens sans lemmatiseur. Une vraie lemmatisation nécessite un modèle spaCy entraîné, par exemple `fr_core_news_sm`.
