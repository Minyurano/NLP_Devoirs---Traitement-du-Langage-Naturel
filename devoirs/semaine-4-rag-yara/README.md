# Semaine 4 — Comparaison d'architectures RAG pour les règles YARA

Prototype qui compare **six architectures RAG** pour générer des règles YARA à partir d'une description de malware. C'est la première version du [projet final](../../projet-yara-rag/).

| Architecture | Idée |
|---|---|
| Classic RAG | Recherche dense (FAISS) puis génération |
| RAG with Re-ranking | Re-classement des documents récupérés |
| Hybrid RAG | Combinaison recherche dense + BM25 |
| Multi-hop RAG | Reformulation de la requête et recherche en plusieurs étapes |
| Graph RAG | Graphe de connaissances (NetworkX) entre types de malwares et indicateurs |
| Agentic RAG | Un agent choisit l'action (recherche par type, multi-étapes…) |

- **Données :** base personnelle de 20 entrées (description, règle YARA, type de malware, indicateurs), créée manuellement puis enrichie synthétiquement.
- **Évaluation :** confiance moyenne, qualité de la récupération, qualité de la règle.

**Notebook :** [`rag_yara_architectures.ipynb`](rag_yara_architectures.ipynb) (exécuté, avec résultats)
