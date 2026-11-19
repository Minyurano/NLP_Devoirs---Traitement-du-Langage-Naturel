# Traitement du Langage Naturel (NLP) — Devoirs et projet

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?logo=spacy&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF)
![Gradio](https://img.shields.io/badge/Gradio-UI-F97316?logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

Travaux réalisés dans le cadre du module **Traitement du Langage Naturel** du Master **Intelligence Artificielle & Sciences des Données (IASD)**, FST de Tanger. Le dépôt suit une progression : du prétraitement de texte classique jusqu'à un système **RAG** (Retrieval-Augmented Generation) complet appliqué à la cybersécurité.

---

## Contenu

| Partie | Sujet | Techniques |
|---|---|---|
| [Semaine 1](devoirs/semaine-1-pretraitement/) | Prétraitement de texte en français | Tokenisation spaCy, stopwords, stemming Snowball, TF-IDF |
| [Semaine 2](devoirs/semaine-2-rag-introduction/) | Premier RAG sur des infractions du Code de la route | Sentence-Transformers, index FAISS, recherche sémantique |
| [Semaine 3](devoirs/semaine-3-extraction-juridique/) | Extraction d'informations depuis un texte juridique (PDF scanné) | OCR Tesseract, regex, TF-IDF + KMeans, NER spaCy (EntityRuler) |
| [Semaine 4](devoirs/semaine-4-rag-yara/) | Comparaison d'architectures RAG pour générer des règles YARA | RAG classique, re-ranking, hybride, multi-hop, graph RAG, agentic |
| [**Projet final**](projet-yara-rag/) | **Génération automatique de règles YARA par RAG + LLM** | BM25 + FAISS, Cross-Encoder, TinyLlama / Phi-2 / Mistral-7B, interface Gradio |

Présentation du projet : [`docs/presentation-rag-yara.pptx`](docs/presentation-rag-yara.pptx)

## Projet final en bref

À partir d'une description en langage naturel d'un comportement malveillant (ex. *« ransomware chiffrant les fichiers avec AES »*), le système retrouve les règles YARA les plus proches parmi ~30 000 règles validées, puis demande à un LLM d'en rédiger une nouvelle. Trois stratégies de récupération sont comparées : **hybride**, **re-ranking** et **agentic**.
→ Détails, architecture et résultats : [projet-yara-rag/README.md](projet-yara-rag/README.md)

## Structure du dépôt

```
.
├── projet-yara-rag/                 # Projet final : génération de règles YARA par RAG
│   ├── yara_rag_system.py           # Code du système (script exécutable)
│   ├── yara_rag_demo.ipynb          # Notebook d'exécution avec résultats
│   ├── requirements.txt
│   └── README.md
├── devoirs/
│   ├── semaine-1-pretraitement/
│   ├── semaine-2-rag-introduction/
│   ├── semaine-3-extraction-juridique/
│   └── semaine-4-rag-yara/
├── docs/
│   └── presentation-rag-yara.pptx
├── LICENSE
└── README.md
```

## Exécution des devoirs

Les notebooks des devoirs ont été exécutés sur Google Colab ou Jupyter ; leurs résultats sont conservés dans les fichiers. Pour les relancer, ouvrir le notebook et exécuter les cellules dans l'ordre (les cellules `!pip install` installent les dépendances nécessaires).

## Auteur

**Aloys MINYURANO** — Master IASD, FST de Tanger (Université Abdelmalek Essaâdi)
[GitHub](https://github.com/Minyurano) · [LinkedIn](https://www.linkedin.com/in/aloys-minyurano-871717402)

Distribué sous licence [MIT](LICENSE).
