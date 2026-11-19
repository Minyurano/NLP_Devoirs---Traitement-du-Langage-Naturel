# Semaine 2 — Introduction au RAG

Premier système **RAG** (Retrieval-Augmented Generation) construit sur des articles du Code de la route marocain (infractions, amendes, niveau de gravité).

| Étape | Outil |
|---|---|
| Données | 10 articles structurés (`article_id`, infraction, amende min/max, type, gravité, mots-clés) |
| Embeddings | Sentence-Transformers `all-MiniLM-L6-v2` |
| Indexation | FAISS (`IndexFlatL2`) |
| Recherche | top-k articles les plus proches d'une question |

**Notebook :** [`rag_introduction.ipynb`](rag_introduction.ipynb) (exécuté, avec résultats)
