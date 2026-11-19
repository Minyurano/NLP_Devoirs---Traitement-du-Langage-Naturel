# YARA RAG — Génération automatique de règles YARA par RAG et LLM

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-dense%20retrieval-0467DF)
![BM25](https://img.shields.io/badge/BM25-sparse%20retrieval-6B7280)
![Transformers](https://img.shields.io/badge/Transformers-LLM-FFD21E?logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-interface-F97316?logo=gradio&logoColor=white)

Générer une **règle YARA** (signature de détection de malware) à partir d'une **simple description textuelle** d'un comportement malveillant, en combinant une recherche dans ~30 000 règles existantes et un modèle de langage.

---

## Contexte

Écrire des règles YARA demande une expertise pointue (familles de malwares, chaînes caractéristiques, motifs hexadécimaux, appels système). Face au volume croissant de fichiers suspects, la rédaction manuelle devient un goulot d'étranglement. Ce projet explore l'idée suivante : **s'appuyer sur un corpus de règles validées pour guider un LLM** dans la rédaction d'une nouvelle règle.

## Objectifs
- Générer une règle YARA à partir d'une description en langage naturel.
- Réutiliser un corpus de règles existantes et validées comme base de connaissances.
- Comparer trois stratégies RAG : **hybride**, **re-ranking** et **agentic**.
- Proposer une interface interactive (Gradio) pour tester le système.

## Architecture

```
 Description du malware (texte)
            │
            ▼
 ┌──────────────────────────────────────────────┐
 │ Récupération                                 │
 │  • FAISS  (dense, BAAI/bge-base-en-v1.5)     │
 │  • BM25   (sparse, termes techniques / API)  │
 │  • Fusion : score = 0.6·dense + 0.4·sparse   │
 │  • [Rerank] Cross-Encoder ms-marco-MiniLM    │
 └──────────────────────────────────────────────┘
            │  top-k règles proches (strings + condition)
            ▼
 ┌──────────────────────────────────────────────┐
 │ Génération : LLM (Mistral-7B / Phi-2 /       │
 │ TinyLlama / distilgpt2 selon la mémoire)     │
 │ [Agentic] jusqu'à 3 essais si syntaxe        │
 │ invalide                                     │
 └──────────────────────────────────────────────┘
            │
            ▼
 Post-traitement (nom de règle, meta:, accolades)
            │
            ▼
 Validation par compilation yara-python → règle YARA
```

### Base de connaissances
- ~30 000 règles YARA issues de sources publiques, filtrées sur `is_valid = True`.
- Colonnes attendues : `rule_name`, `description`, `rule_body`, `predicted_type` (ou `malware_type`), `is_valid`, `source_url`.
- Embeddings : `BAAI/bge-base-en-v1.5` (768 dimensions), similarité cosinus via FAISS.

### Sélection automatique du LLM
| Ressources disponibles | Modèle utilisé |
|---|---|
| GPU ≥ 8 Go de VRAM | Mistral-7B-Instruct (quantifié 4 bits) |
| GPU 4–8 Go de VRAM | Phi-2 (2,7 B, quantifié 4 bits) |
| Sans GPU (ou < 4 Go de VRAM) | TinyLlama-1.1B-Chat |
| Sans GPU et RAM libre < 2,5 Go | distilgpt2 (repli) |

### Les trois architectures
| Architecture | Principe |
|---|---|
| **Hybride** | Fusion linéaire des scores FAISS et BM25 (`alpha = 0.6`) |
| **Rerank** | Les 15 meilleurs candidats sont re-classés par un Cross-Encoder |
| **Agentic** | Récupération re-rankée, puis génération vérifiée par `yara-python` et relancée si la syntaxe est invalide (3 essais maximum) |

## Évaluation

Quatre métriques, calculées sur 5 requêtes de test (ransomware, keylogger, rootkit, backdoor, cryptominer) :

| Métrique | Description |
|---|---|
| Syntaxe valide | La règle générée compile avec `yara-python` |
| Similarité | Cosinus entre l'embedding de la règle générée et celui des règles récupérées |
| Hallucination | Absence de `meta:` / `strings:` ou incohérence entre `condition` et `strings` |
| Pertinence de la récupération | Part des règles récupérées ayant une description informative et un type spécifique |

### Résultats obtenus

Score global = 0,30 × syntaxe valide + 0,30 × similarité + 0,20 × (1 − hallucination) + 0,20 × pertinence.


| Architecture | Syntaxe valide | Similarité | Hallucination ↓ | Pertinence | Score global |
|---|---:|---:|---:|---:|---:|
| **Agentic RAG** | 0.00 | **0.716** | **0.00** | **0.80** | **0.575** |
| RAG Rerank | 0.00 | 0.715 | 0.06 | **0.80** | 0.562 |
| RAG Hybride | 0.00 | 0.709 | **0.00** | 0.72 | 0.557 |

### Analyse
- L'architecture **Agentic** obtient le meilleur score global : similarité la plus élevée et aucune hallucination détectée.
- **Aucune règle n'est syntaxiquement valide** sur ce test. L'exécution a eu lieu sur une machine avec seulement 0,7 Go de RAM libre : le système a donc basculé sur **distilgpt2**, un petit modèle généraliste qui n'est pas entraîné à produire du code YARA. La récupération, elle, fonctionne (règles pertinentes retrouvées, pertinence de 0,72 à 0,80).
- Prochaine étape : relancer l'évaluation avec TinyLlama ou Phi-2 (≥ 2,5 Go de RAM ou un GPU), puis envisager un fine-tuning léger (LoRA) sur des paires description → règle.

## Installation et exécution

```bash
git clone https://github.com/Minyurano/NLP_Devoirs---Traitement-du-Langage-Naturel.git
cd NLP_Devoirs---Traitement-du-Langage-Naturel/projet-yara-rag
pip install -r requirements.txt
python yara_rag_system.py
```

- Le fichier de données **`yara_step2.csv`** (règles YARA, volumineux) doit être placé dans ce dossier ; il n'est pas versionné.
- Au premier lancement, les modèles (embedding, re-ranker, LLM) sont téléchargés depuis Hugging Face et les index FAISS / BM25 sont calculés puis mis en cache.
- Les index sont **recalculés automatiquement si le dataset change** (empreinte SHA-256 enregistrée dans `yara_index_meta.json`).
- L'interface Gradio s'ouvre sur `http://127.0.0.1:7860` (ou le port libre suivant).

Le notebook [`yara_rag_demo.ipynb`](yara_rag_demo.ipynb) contient une exécution complète avec ses sorties.

## Exemples de requêtes

| Requête | Architecture conseillée |
|---|---|
| `ransomware encrypt files AES` | Hybride |
| `keylogger GetAsyncKeyState` | Rerank |
| `rootkit SSDT hooking` | Agentic |

## Interface

L'application Gradio permet de :
- saisir la description d'un malware ;
- choisir l'architecture RAG et le LLM ;
- générer la règle et voir si sa syntaxe est valide ;
- obtenir une explication ligne par ligne de la règle ;
- consulter les règles sources utilisées (URL).

## Fichiers

| Fichier | Rôle |
|---|---|
| `yara_rag_system.py` | Système complet : données, index, LLM, 3 architectures RAG, métriques, interface |
| `yara_rag_demo.ipynb` | Exécution de référence avec sorties |
| `requirements.txt` | Dépendances Python |

Les modèles téléchargés et les index générés sont exclus du dépôt (voir `.gitignore`).
