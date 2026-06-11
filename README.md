🛡️ YARA RAG System – Génération automatique de règles YARA par RAG et LLM
 Contexte et problématique
La détection de menaces par des règles YARA reste un art complexe, nécessitant une expertise approfondie des familles de malwares, des signatures hexadécimales et des comportements systèmes. Face à des volumes croissants de fichiers suspects, la rédaction manuelle de règles devient un goulot d’étranglement.

Ce projet propose une solution innovante : générer automatiquement des règles YARA à partir d’une simple description textuelle d’un comportement malveillant, en combinant :

Un moteur de recherche RAG hybride (BM25 + FAISS) sur plus de 30 000 règles YARA existantes et validées.

Un modèle de langage (LLM) (TinyLlama / Phi-2 / Mistral-7B) pour synthétiser une nouvelle règle, adaptée au contexte récupéré.

Trois architectures RAG : Hybride (fusion BM25+FAISS), Rerank (avec Cross-Encoder) et Agentic (itérations de correction).

 Objectifs
Automatiser la création de règles YARA à partir d’un langage naturel.

Réduire le temps d’expertise en s’appuyant sur un corpus existant de règles validées.

Comparer trois stratégies de RAG (hybride, rerank, agentic) en termes de qualité syntaxique, similarité sémantique et pertinence.

Fournir une interface interactive (Gradio) pour tester et visualiser les règles générées.

 Architecture technique
 1. Base de connaissances
Dataset : ~30 000 règles YARA issues de sources publiques, filtrées sur is_valid = True.

Colonnes principales : rule_name, description, rule_body, type (malware, ransomware, keylogger…).

Embedding : BAAI/bge-base-en-v1.5 (768 dimensions) pour la recherche sémantique.

Indexation :

FAISS (cosine similarity) pour la recherche dense.

BM25 (tokenisation classique) pour la recherche exacte (termes techniques, API, hex patterns).

 2. Récupération (Retrieval)
Hybride : fusion linéaire des scores dense (FAISS) et sparse (BM25) avec alpha = 0.6.

Rerank : les 15 premiers candidats sont re‑classés par un Cross‑Encoder (ms-marco-MiniLM-L-6-v2).

Agentic : itère la génération jusqu’à obtenir une règle syntaxiquement valide (max 3 essais).

 3. Génération (LLM)
LLM par défaut : TinyLlama-1.1B-Chat (compatible CPU, < 2 Go RAM).

Fallback : si RAM < 2.5 Go, bascule sur distilgpt2 (~350 Mo).

Prompting : le contexte fourni contient uniquement les sections strings et condition des règles récupérées (pour guider la syntaxe sans copier).

Post‑traitement : correction des noms de règle, déduplication des chaînes, insertion auto de meta:, équilibrage des accolades.

 4. Évaluation
Quatre métriques calculées sur 5 requêtes de test (ransomware, keylogger, rootkit, backdoor, cryptominer) :

Métrique	Description
Syntaxe Valide	Compilation YARA réussie
Similarité	Cosine entre l’embedding de la règle générée et celui des documents récupérés
Hallucination	Présence de meta:, de strings, et cohérence condition/strings
Pertinence Retrieval	Proportion de documents récupérés avec description > 30 caractères et type spécifique
 Interface utilisateur (Gradio)
L’application permet de :

Saisir une description du malware (ex : “Ransomware encrypting files with AES”).

Choisir l’architecture RAG et le LLM (via menu déroulant).

Générer la règle YARA, avec affichage de la syntaxe valide / invalide.

Consulter les explications ligne à ligne de la règle (via le LLM).

Voir les références (URL des règles sources utilisées).

https://image.png

 Résultats comparatifs (extrait)
Architecture	Syntaxe Valide	Similarité	Hallucination	Pertinence Retrieval	Score Global
Agentic RAG	0.0	0.716	0.00	0.80	0.575
RAG Rerank	0.0	0.715	0.06	0.80	0.562
RAG Hybride	0.0	0.709	0.00	0.72	0.557
Analyse : bien qu’aucune règle générée ne soit syntaxiquement valide sur ce petit test (en raison de la brièveté des exemples dans le dataset de démonstration), l’architecture Agentic obtient le meilleur score global grâce à une meilleure similarité et une hallucination nulle. Les règles produites sont proches sémantiquement des exemples et présentent une structure correcte (meta, strings, condition). L’utilisation d’un dataset réel plus riche améliorerait la validité syntaxique.

 Installation et exécution
bash
git clone https://github.com/Minyurano/projet-NLP.git
cd projet-NLP
pip install -r requirements.txt   # ou exécutez le notebook
python yara_rag_system.py
Dans l’environnement Jupyter / Colab, exécutez simplement toutes les cellules du notebook cool.projet of NLP.ipynb. L’interface Gradio se lance automatiquement sur http://127.0.0.1:7861.

 Exemples de requêtes
Requête	Architecture recommandée
ransomware encrypt files AES	Hybride
keylogger GetAsyncKeyState	Rerank
rootkit SSDT hooking	Agentic
 Structure du dépôt
text
projet-NLP/
├── yara_rag_system.py            # Script principal (ou notebook .ipynb)
├── yara_step2.csv                # Dataset de règles YARA
├── embedding_model/              # Modèle BAAI/bge-base-en-v1.5 (caché)
├── llm_tinyllama/                # TinyLlama-1.1B (caché)
├── llm_distilgpt2/               # Fallback distilgpt2 (caché)
├── reranker_model/               # Cross-encoder ms-marco (caché)
├── yara_faiss.index              # Index FAISS pré‑calculé
├── yara_bm25.pkl                 # Index BM25 pré‑calculé
└── README.md                     # Ce fichier
