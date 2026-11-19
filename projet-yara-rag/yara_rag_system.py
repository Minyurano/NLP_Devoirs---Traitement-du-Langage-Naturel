"""
Système RAG YARA — Génération de Règles YARA
Domaine       : Cybersécurité
Dataset       : ~30K règles YARA validées (yara-python)
Architectures : Hybride | Rerank | Agentic
LLMs          : Phi-2 | TinyLlama | Mistral-7B

Pipeline : User décrit malware → RAG Hybride (BM25+FAISS) → LLM génère règle YARA

Usage :
    pip install -r requirements.txt
    python yara_rag_system.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. DÉPENDANCES : installées au préalable via `pip install -r requirements.txt`
# ─────────────────────────────────────────────────────────────────────────────
import os


# ─────────────────────────────────────────────────────────────────────────────
# 2. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import re
import json
import hashlib
import time
import pickle
import warnings
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
import torch
import yara
import gradio as gr

print('Imports OK')
print(f'GPU disponible : {torch.cuda.is_available()}')


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHEMINS (relatifs au dossier du script)
# ─────────────────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # script Python classique
except NameError:
    BASE_DIR = os.getcwd()  # Jupyter / Colab : utilise le dossier courant

DATASET_PATH    = os.path.join(BASE_DIR, 'yara_step2.csv')
EMBED_MODEL_DIR = os.path.join(BASE_DIR, 'embedding_model')
FAISS_PATH      = os.path.join(BASE_DIR, 'yara_faiss.index')
BM25_PATH       = os.path.join(BASE_DIR, 'yara_bm25.pkl')
DOCS_PATH       = os.path.join(BASE_DIR, 'yara_docs.pkl')
EMBED_PATH      = os.path.join(BASE_DIR, 'yara_embeddings.pkl')
INDEX_META_PATH = os.path.join(BASE_DIR, 'yara_index_meta.json')  # empreinte du dataset indexé
LLM_DIR_MISTRAL = os.path.join(BASE_DIR, 'llm_mistral')
LLM_DIR_PHI2    = os.path.join(BASE_DIR, 'llm_phi2')
LLM_DIR_TINY    = os.path.join(BASE_DIR, 'llm_tinyllama')
RERANKER_DIR    = os.path.join(BASE_DIR, 'reranker_model')
EVAL_CSV_PATH   = os.path.join(BASE_DIR, 'rag_evaluation.csv')


# ─────────────────────────────────────────────────────────────────────────────
# 4. CHARGEMENT ET PRÉPARATION DU DATASET
# ─────────────────────────────────────────────────────────────────────────────
print('Chargement du dataset...')
df_raw = pd.read_csv(DATASET_PATH, dtype=str).fillna('')
print(f'  Total chargé : {len(df_raw):,} règles')

# Garder uniquement les règles valides (déjà vérifiées par yara-python)
df = df_raw[df_raw['is_valid'].str.lower().isin(['true', '1', 'yes'])].copy()
print(f'  Après filtre is_valid=True : {len(df):,} règles')

# Normaliser la colonne rule_body
if 'rule_body' not in df.columns and 'rule_text' in df.columns:
    df.rename(columns={'rule_text': 'rule_body'}, inplace=True)
if 'rule_body' not in df.columns and 'rule' in df.columns:
    df.rename(columns={'rule': 'rule_body'}, inplace=True)

# Utiliser predicted_type si disponible, sinon malware_type
if 'predicted_type' in df.columns:
    df['type'] = df['predicted_type']
elif 'malware_type' in df.columns:
    df['type'] = df['malware_type']
else:
    df['type'] = 'malware'

# Texte enrichi pour l'embedding (rule_name + type + description)
df['embed_text'] = (
    df['rule_name'].str.replace('_', ' ') + ' ' +
    df['type'] + ' ' +
    df['description']
).str.strip()

# Source URL pour le bouton Référence
if 'source_url' not in df.columns:
    df['source_url'] = ''

df = df.reset_index(drop=True)
documents = df.to_dict('records')

print(f'  Documents prêts : {len(documents):,}')
print(f'  Colonnes        : {list(df.columns)}')
print('\n  Distribution des types :')
print(df['type'].value_counts().head(10).to_string())
print('\n  Exemple embed_text :')
print(f'  {df["embed_text"].iloc[0][:120]}')


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODÈLE D'EMBEDDING (téléchargé une seule fois)
# ─────────────────────────────────────────────────────────────────────────────
# Modèle orienté code / cybersécurité — remplace all-MiniLM-L6-v2
EMBED_MODEL_NAME = 'BAAI/bge-base-en-v1.5'

def _load_embedding_model(model_path: str, model_name: str) -> SentenceTransformer:
    """Charge le modèle d'embedding avec gestion d'erreur et re-téléchargement si corrompu."""
    if os.path.exists(model_path):
        print(f'Modèle embedding déjà téléchargé — chargement depuis {model_path}...')
        try:
            model = SentenceTransformer(model_path, device='cpu')
            # Test rapide pour vérifier que le modèle n'est pas corrompu
            _ = model.encode(['test'], convert_to_numpy=True)
            return model
        except Exception as e:
            print(f'  Modèle local corrompu ({e}) → re-téléchargement...')
            import shutil
            shutil.rmtree(model_path, ignore_errors=True)

    print(f'Téléchargement du modèle embedding : {model_name}...')
    try:
        model = SentenceTransformer(model_name, device='cpu', trust_remote_code=True)
        model.save(model_path)
        print(f'Modèle sauvegardé → {model_path}')
        return model
    except Exception as e:
        # Fallback sur un modèle encore plus léger si le principal échoue
        fallback = 'sentence-transformers/all-MiniLM-L6-v2'
        print(f'  Échec ({e}) → fallback sur {fallback}...')
        model = SentenceTransformer(fallback, device='cpu')
        model.save(model_path)
        print(f'Modèle fallback sauvegardé → {model_path}')
        return model

embedding_model = _load_embedding_model(EMBED_MODEL_DIR, EMBED_MODEL_NAME)

EMBED_DIM = embedding_model.get_sentence_embedding_dimension()
print(f'Modèle prêt — dimension : {EMBED_DIM}')


# ─────────────────────────────────────────────────────────────────────────────
# 6. INDEXATION FAISS + BM25 (calculée une seule fois)
# ─────────────────────────────────────────────────────────────────────────────
embed_texts = df['embed_text'].tolist()

# Empreinte du dataset : si le CSV change (règles ajoutées/supprimées), les
# index en cache ne correspondent plus et doivent être recalculés.
DATASET_FINGERPRINT = hashlib.sha256(
    '\n'.join(embed_texts).encode('utf-8')
).hexdigest()


def _cache_is_valid() -> bool:
    paths = (FAISS_PATH, BM25_PATH, DOCS_PATH, EMBED_PATH, INDEX_META_PATH)
    if not all(os.path.exists(p) for p in paths):
        return False
    with open(INDEX_META_PATH, encoding='utf-8') as f:
        meta = json.load(f)
    return (
        meta.get('fingerprint') == DATASET_FINGERPRINT
        and meta.get('n_documents') == len(embed_texts)
        and meta.get('embedding_dim') == EMBED_DIM
    )


if _cache_is_valid():
    print('Index en cache valides (même dataset) — chargement...')
    faiss_index = faiss.read_index(FAISS_PATH)
    with open(BM25_PATH, 'rb') as f:
        bm25 = pickle.load(f)
    with open(DOCS_PATH, 'rb') as f:
        documents = pickle.load(f)
    with open(EMBED_PATH, 'rb') as f:
        doc_embeddings = pickle.load(f)
    print(f'  FAISS : {faiss_index.ntotal:,} vecteurs')
    print(f'  BM25  : {len(documents):,} documents')
else:
    print(f'Calcul des embeddings pour {len(embed_texts):,} règles...')
    doc_embeddings = embedding_model.encode(
        embed_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=64,
    )
    print(f'  Shape embeddings : {doc_embeddings.shape}')

    # Index FAISS (Inner Product après normalisation = cosine)
    print('Construction index FAISS...')
    faiss_index = faiss.IndexFlatIP(EMBED_DIM)
    norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    normalized = doc_embeddings / (norms + 1e-10)
    faiss_index.add(normalized.astype('float32'))

    # Index BM25
    print('Construction index BM25...')
    tokenized = [t.lower().split() for t in embed_texts]
    bm25 = BM25Okapi(tokenized)

    # Sauvegardes
    faiss.write_index(faiss_index, FAISS_PATH)
    with open(BM25_PATH, 'wb') as f:
        pickle.dump(bm25, f)
    with open(DOCS_PATH, 'wb') as f:
        pickle.dump(documents, f)
    with open(EMBED_PATH, 'wb') as f:
        pickle.dump(doc_embeddings, f)

    with open(INDEX_META_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'fingerprint': DATASET_FINGERPRINT,
            'n_documents': len(embed_texts),
            'embedding_dim': EMBED_DIM,
        }, f, indent=2)

    print(f'Index sauvegardés → {FAISS_PATH}, {BM25_PATH}')

print('Indexation terminée.')


# ─────────────────────────────────────────────────────────────────────────────
# 7. MODÈLE LLM (téléchargé une seule fois)
#    Décommenter l'option souhaitée, commenter les autres.
# ─────────────────────────────────────────────────────────────────────────────

# ── Sélection automatique du LLM selon la VRAM disponible ───────────────────
#   - GPU avec >= 8 Go VRAM  → Mistral-7B  (meilleure qualité)
#   - GPU avec 4-8 Go VRAM   → Phi-2 2.7B  (bon compromis)
#   - CPU / < 4 Go VRAM      → TinyLlama   (le plus léger, ~2 Go RAM)

def _free_vram_gb() -> float:
    """Retourne la VRAM libre en Go (0 si pas de GPU)."""
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / 1024 ** 3

_vram = _free_vram_gb()
print(f'VRAM libre détectée : {_vram:.1f} Go')

if _vram >= 8:
    LLM_NAME = 'mistralai/Mistral-7B-Instruct-v0.1'
    LLM_DIR  = LLM_DIR_MISTRAL
elif _vram >= 4:
    LLM_NAME = 'microsoft/phi-2'
    LLM_DIR  = LLM_DIR_PHI2
else:
    # CPU ou VRAM insuffisante → TinyLlama (fonctionne même sans GPU)
    LLM_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    LLM_DIR  = LLM_DIR_TINY

# 4-bit uniquement si GPU disponible avec assez de VRAM
USE_4BIT = torch.cuda.is_available() and _vram >= 4

# ── Vérification RAM disponible ──────────────────────────────────────────────
# TinyLlama nécessite ~2 Go de RAM libre. Si insuffisant, on bascule sur
# distilgpt2 (~350 Mo) qui fonctionne même avec très peu de RAM.
import psutil
_ram_free_gb = psutil.virtual_memory().available / 1024**3
print(f'RAM libre disponible : {_ram_free_gb:.1f} Go')

if _ram_free_gb < 2.5 and _vram == 0.0:
    print('  RAM insuffisante pour TinyLlama → bascule sur distilgpt2 (350 Mo)')
    LLM_NAME = 'distilgpt2'
    LLM_DIR  = os.path.join(BASE_DIR, 'llm_distilgpt2')

print(f'LLM sélectionné : {LLM_NAME}')


def _load_llm(model_path: str) -> AutoModelForCausalLM:
    """
    Charge le LLM avec la stratégie adaptée à la VRAM disponible.
    - USE_4BIT=True  : 4-bit NF4 + CPU offload si VRAM partiellement insuffisante
    - USE_4BIT=False : float32 sur CPU (lent mais universel, pas besoin de GPU)
    """
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,  # autorise l'offload CPU
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map='auto',
                trust_remote_code=True,
            )
        except ValueError:
            # VRAM encore insuffisante même avec offload → CPU float32
            print('  Bascule sur CPU (VRAM insuffisante)...')
            for dtype in [torch.float16, torch.bfloat16, torch.float32]:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map='cpu',
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                    )
                    print(f'  Chargement CPU réussi avec dtype={dtype}')
                    break
                except (OSError, RuntimeError):
                    continue
    else:
        # Essayer float16 d'abord (2x moins de RAM que float32)
        # puis bfloat16, puis float32 en dernier recours
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            try:
                print(f'  Chargement CPU avec dtype={dtype}...')
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map='cpu',
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                print(f'  Chargement réussi avec dtype={dtype}')
                break
            except (OSError, RuntimeError) as e:
                print(f'  Échec avec dtype={dtype} : {e}')
                continue
        else:
            raise RuntimeError(
                'Impossible de charger le modèle : RAM insuffisante.\n'
                'Solutions :\n'
                '  1. Augmenter la mémoire virtuelle Windows (Paramètres → Performances → Fichier d\'échange)\n'
                '  2. Fermer les autres applications pour libérer de la RAM\n'
                '  3. Utiliser un modèle encore plus petit (ex: gpt2)'
            )
    return model


if os.path.exists(LLM_DIR):
    print(f'LLM déjà téléchargé — chargement depuis {LLM_DIR}...')
    tokenizer = AutoTokenizer.from_pretrained(LLM_DIR)
    llm = _load_llm(LLM_DIR)
else:
    print(f'Téléchargement LLM : {LLM_NAME}...')
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, trust_remote_code=True)
    llm = _load_llm(LLM_NAME)
    tokenizer.save_pretrained(LLM_DIR)
    llm.save_pretrained(LLM_DIR)
    print(f'LLM sauvegardé → {LLM_DIR}')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm_pipeline = pipeline(
    'text-generation',
    model=llm,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False,
)

ACTIVE_LLM = LLM_NAME.split('/')[-1]
print(f'LLM actif : {ACTIVE_LLM}')


def generate(prompt: str, max_tokens: int = 400) -> str:
    """
    Génère du texte avec le LLM et post-traite la sortie pour
    produire une règle YARA syntaxiquement correcte.
    """
    try:
        # distilgpt2 / gpt2 : limiter les tokens car modèle petit
        effective_tokens = min(max_tokens, 300) if 'gpt2' in LLM_NAME.lower() else max_tokens
        out = llm_pipeline(prompt, max_new_tokens=effective_tokens, return_full_text=False)
        text = out[0]['generated_text'].strip()

        # 1. Supprimer les préfixes numériques type "1 {"
        text = re.sub(r'^\d+\s*\{', 'rule Detect_Malware {', text)

        # 2. Ajouter "rule " si absent
        if not re.search(r'^rule\s+\w+', text, re.IGNORECASE):
            match = re.search(r'^([A-Za-z_][A-Za-z0-9_]*)\s*\{', text, re.MULTILINE)
            if match:
                rule_name = match.group(1)
                text = re.sub(
                    rf'^{re.escape(rule_name)}\s*{{',
                    f'rule {rule_name} {{',
                    text,
                    flags=re.MULTILINE,
                )
            else:
                text = 'rule Detect_Malware {\n' + text

        # 3. Supprimer le faux "rule Detect_Malware {" suivi d'un autre rule
        text = re.sub(
            r'rule\s+Detect_Malware\s*\{\s*\n\s*(?=\w+\s*\{)',
            '',
            text,
            flags=re.IGNORECASE,
        )

        # 4. Corriger les noms de règle commençant par un chiffre
        match = re.search(r'rule\s+([0-9][A-Za-z0-9_]*)\s*\{', text)
        if match:
            text = text.replace(f'rule {match.group(1)}', f'rule Rule_{match.group(1)}')

        # 5. Supprimer les appels Python (string.find, import, def)
        lines = text.split('\n')
        text = '\n'.join([
            l for l in lines
            if 'string.find(' not in l
            and '.find(' not in l
            and 'import ' not in l
            and 'def ' not in l
        ])

        # 6. Nettoyer les strings hex (supprimer les commentaires // dans les blocs hex)
        def clean_hex_line(m):
            content = re.sub(r'\s*//.*', '', m.group(1))
            return f'{{ {content.strip()} }}'
        text = re.sub(r'\{\s*([^}]*?)\s*\}', clean_hex_line, text, flags=re.DOTALL)

        # 7. Déduplication des strings (par valeur)
        lines = text.split('\n')
        new_lines = []
        seen_values: set = set()
        in_strings = False
        for line in lines:
            if re.search(r'^\s*strings\s*[:{]', line, re.IGNORECASE):
                in_strings = True
                new_lines.append(line)
                continue
            if in_strings and re.search(r'^\s*condition\s*[:{]', line, re.IGNORECASE):
                in_strings = False
            if in_strings:
                m = re.match(
                    r'\s*(\$\w+)\s*=\s*(\{[^}]+\}|"[^"]+")'
                    r'(?:\s+(?:ascii|wide|fullword))*',
                    line,
                )
                if m:
                    _, val = m.groups()
                    if val in seen_values:
                        continue
                    seen_values.add(val)
            new_lines.append(line)
        text = '\n'.join(new_lines)

        # 8. Insérer "meta:" si absent juste après l'accolade ouvrante
        if re.search(r'rule\s+\w+\s*\{\s*(?!meta:)', text, re.IGNORECASE):
            text = re.sub(
                r'(rule\s+\w+\s*\{\s*)',
                r'\1meta:\n    author = "RAG-System"\n',
                text,
                flags=re.IGNORECASE,
            )

        # 9. Corriger les doubles accolades
        text = text.replace('{ {', '{').replace('}}', '}')

        # 10. Corriger les sections mal formatées (meta/strings/condition avec {)
        text = re.sub(r'meta\s*\{',      'meta:',      text, flags=re.IGNORECASE)
        text = re.sub(r'strings\s*\{',   'strings:',   text, flags=re.IGNORECASE)
        text = re.sub(r'condition\s*\{', 'condition:', text, flags=re.IGNORECASE)

        # 11. Nettoyer la condition (supprimer les références à des strings non définies)
        defined_strings = set(re.findall(r'\$(\w+)\s*=', text))

        def clean_condition(m):
            cond = m.group(1)
            for s in re.findall(r'\$(\w+)', cond):
                if s not in defined_strings:
                    cond = re.sub(rf'\${re.escape(s)}\b', '', cond)
            cond = re.sub(r'\s+and\s+and\s+', ' and ', cond)
            cond = re.sub(r'\s+or\s+or\s+',   ' or ',  cond)
            cond = re.sub(r'\s+and\s*$',       '',      cond)
            cond = re.sub(r'^\s*and\s+',       '',      cond)
            if not cond.strip():
                cond = 'true'
            return f'condition: {cond}'

        text = re.sub(
            r'condition\s*:\s*(.*?)(?=\n\S|\Z)',
            clean_condition,
            text,
            flags=re.DOTALL,
        )

        # 12. Fermer les accolades manquantes
        open_b  = text.count('{')
        close_b = text.count('}')
        if open_b > close_b:
            text += '}' * (open_b - close_b)

        return text.strip()

    except Exception as e:
        return f'Erreur génération: {e}'


# ─────────────────────────────────────────────────────────────────────────────
# 8. RE-RANKER (téléchargé une seule fois)
# ─────────────────────────────────────────────────────────────────────────────
RERANKER_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

if os.path.exists(RERANKER_DIR):
    print('Re-ranker déjà téléchargé — chargement...')
    reranker = CrossEncoder(RERANKER_DIR)
else:
    print(f'Téléchargement re-ranker : {RERANKER_NAME}...')
    reranker = CrossEncoder(RERANKER_NAME)
    reranker.save(RERANKER_DIR)
    print(f'Re-ranker sauvegardé → {RERANKER_DIR}')

print('Re-ranker prêt.')


# ─────────────────────────────────────────────────────────────────────────────
# 9. FONCTIONS DE RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_hybrid(query: str, k: int = 5, alpha: float = 0.6) -> List[Dict]:
    """
    RAG Hybride : BM25 (sparse) + FAISS (dense) fusionnés.
    alpha=0.6 → favorise légèrement FAISS (sémantique).
    BM25 capture les strings YARA exacts (hex, noms d'API).
    FAISS capture le sens général de la requête.
    """
    # Dense — FAISS
    q_emb = embedding_model.encode([query], convert_to_numpy=True)
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)
    scores_faiss, indices_faiss = faiss_index.search(q_norm.astype('float32'), k * 3)
    dense = {int(idx): float(score) for idx, score in zip(indices_faiss[0], scores_faiss[0])}

    # Sparse — BM25
    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    top_bm25 = np.argsort(bm25_scores)[::-1][:k * 3]
    max_bm25 = bm25_scores[top_bm25[0]] if bm25_scores[top_bm25[0]] > 0 else 1
    sparse = {int(i): float(bm25_scores[i]) / max_bm25 for i in top_bm25}

    # Fusion
    max_dense = max(dense.values()) if dense else 1
    all_ids = set(dense.keys()) | set(sparse.keys())
    combined = {
        idx: alpha * (dense.get(idx, 0) / max_dense) + (1 - alpha) * sparse.get(idx, 0)
        for idx in all_ids
    }

    top_ids = sorted(combined, key=combined.get, reverse=True)[:k]
    results = []
    for idx in top_ids:
        doc = documents[idx].copy()
        doc['retrieval_score'] = round(combined[idx], 4)
        doc['dense_score']     = round(dense.get(idx, 0), 4)
        doc['sparse_score']    = round(sparse.get(idx, 0), 4)
        results.append(doc)
    return results


def retrieve_with_rerank(
    query: str, k_candidates: int = 15, k_final: int = 5
) -> List[Dict]:
    """
    RAG Rerank : Hybride → CrossEncoder pour re-classer.
    Récupère 15 candidats, re-classe, garde le top 5.
    """
    candidates = retrieve_hybrid(query, k=k_candidates)
    pairs = [[query, doc['embed_text']] for doc in candidates]
    rerank_scores = reranker.predict(pairs)
    for doc, score in zip(candidates, rerank_scores):
        doc['rerank_score'] = float(score)
    return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)[:k_final]


print('Fonctions retrieval définies.')

# Test rapide
test = retrieve_hybrid('ransomware encrypt files AES', k=3)
print('Test retrieval hybride — top 3 :')
for d in test:
    print(f"  [{d['type']}] {d['rule_name']} (score: {d['retrieval_score']})")


# ─────────────────────────────────────────────────────────────────────────────
# 10. PROMPT BUILDER ET 3 ARCHITECTURES RAG
# ─────────────────────────────────────────────────────────────────────────────

def build_context(docs: List[Dict], n: int = 2) -> str:
    """
    Construit le contexte syntaxique à partir des règles récupérées.
    On envoie uniquement le squelette (strings + condition) pour guider
    la syntaxe sans copier la règle entière.
    """
    parts = []
    for i, doc in enumerate(docs[:n], 1):
        rule_body = str(doc.get('rule_body', ''))
        strings_match   = re.search(r'strings\s*:(.*?)(?=condition|\})', rule_body, re.DOTALL | re.IGNORECASE)
        condition_match = re.search(r'condition\s*:(.*?)(?=\}|$)',        rule_body, re.DOTALL | re.IGNORECASE)
        strings_part   = strings_match.group(1).strip()[:300]  if strings_match   else ''
        condition_part = condition_match.group(1).strip()[:150] if condition_match else ''
        parts.append(
            f'// Example {i} [{doc.get("type", "unknown")}]\n'
            f'// strings:\n{strings_part}\n'
            f'// condition: {condition_part}'
        )
    return '\n\n'.join(parts)


def build_prompt(
    query: str,
    context: str,
    ref_url: str = '',
    ref_desc: str = '',
    rule_name: str = 'Detect_Malware',
) -> str:
    ref_line = f'reference = "{ref_url[:80]}"' if ref_url else ''

    # Prompt court pour les petits modèles (distilgpt2, gpt2)
    if 'gpt2' in LLM_NAME.lower():
        return f"""rule {rule_name} {{
    meta:
        author = "RAG-System"
        description = "Detects {query[:60]}"
        {ref_line}
    strings:
        $a = "malware" ascii
        $b = "encrypt" ascii
    condition:
        $a and $b
}}"""

    # Prompt complet pour les modèles instruct (TinyLlama, Phi-2, Mistral)
    return f"""<s>[INST] Generate a YARA rule that detects: {query}

Examples (do not copy, only inspire):
{context}

You MUST follow this template exactly, replace CAPITALIZED text:

rule {rule_name} {{
    meta:
        author = "RAG-System"
        description = "A short description of what the rule detects"
        date = "2025-01-01"
        {ref_line}
    strings:
        $a = "first string" ascii
        $b = "second string" ascii
    condition:
        $a and $b
}}

CONSTRAINTS:
- Every string must be properly quoted (").
- The condition must only refer to strings that exist.
- Output ONLY the rule, no extra text.

Now write the rule: [/INST]
rule {rule_name} {{
    meta:
        author = "RAG-System"
        description = \""""


def is_valid_yara(rule_text: str) -> bool:
    """Vérifie si la règle YARA est syntaxiquement valide."""
    try:
        yara.compile(source=rule_text)
        return True
    except Exception:
        return False


# ── Architecture 1 : RAG Hybride ─────────────────────────────────────────────
def rag_hybrid(query: str) -> Tuple[str, List[Dict]]:
    docs      = retrieve_hybrid(query, k=5)
    ref_url   = docs[0].get('source_url',  '') if docs else ''
    ref_desc  = docs[0].get('description', '') if docs else ''
    rule_name = docs[0].get('rule_name', 'Detect_Malware') if docs else 'Detect_Malware'
    if rule_name and rule_name[0].isdigit():
        rule_name = 'Rule_' + rule_name
    context = build_context(docs, n=2)
    prompt  = build_prompt(query, context, ref_url, ref_desc, rule_name)
    rule    = generate(prompt)
    return rule, docs


# ── Architecture 2 : RAG Rerank ──────────────────────────────────────────────
def rag_rerank(query: str) -> Tuple[str, List[Dict]]:
    docs      = retrieve_with_rerank(query, k_candidates=15, k_final=5)
    ref_url   = docs[0].get('source_url',  '') if docs else ''
    ref_desc  = docs[0].get('description', '') if docs else ''
    rule_name = docs[0].get('rule_name', 'Detect_Malware') if docs else 'Detect_Malware'
    if rule_name and rule_name[0].isdigit():
        rule_name = 'Rule_' + rule_name
    context = build_context(docs, n=2)
    prompt  = build_prompt(query, context, ref_url, ref_desc, rule_name)
    rule    = generate(prompt)
    return rule, docs


# ── Architecture 3 : Agentic RAG ─────────────────────────────────────────────
def rag_agentic(query: str, max_iterations: int = 3) -> Tuple[str, List[Dict]]:
    docs      = retrieve_with_rerank(query, k_candidates=15, k_final=5)
    ref_url   = docs[0].get('source_url',  '') if docs else ''
    ref_desc  = docs[0].get('description', '') if docs else ''
    rule_name = docs[0].get('rule_name', 'Detect_Malware') if docs else 'Detect_Malware'
    if rule_name and rule_name[0].isdigit():
        rule_name = 'Rule_' + rule_name
    context = build_context(docs, n=2)
    prompt  = build_prompt(query, context, ref_url, ref_desc, rule_name)
    rule    = generate(prompt)
    for iteration in range(max_iterations):
        if is_valid_yara(rule):
            break
        print(f'  [Agentic] Itération {iteration + 1} — correction...')
        fix_prompt = build_prompt(query, context, ref_url, ref_desc, rule_name)
        rule = generate(fix_prompt)
    return rule, docs


# Mapping pour l'interface
RAG_FUNCTIONS = {
    'RAG Hybride': rag_hybrid,
    'RAG Rerank' : rag_rerank,
    'Agentic RAG': rag_agentic,
}

print('Architectures RAG définies : Hybride | Rerank | Agentic')


# ─────────────────────────────────────────────────────────────────────────────
# 11. MÉTRIQUES D'ÉVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def metric_syntax_valid(rule: str) -> bool:
    """Règle structurellement valide (meta + strings + condition)."""
    return is_valid_yara(rule)


def metric_similarity(generated: str, retrieved_docs: List[Dict]) -> float:
    """Similarité cosine entre règle générée et règles récupérées."""
    if not retrieved_docs:
        return 0.0
    gen_emb   = embedding_model.encode([generated], convert_to_numpy=True)
    ref_texts = [d.get('embed_text', '') for d in retrieved_docs]
    ref_embs  = embedding_model.encode(ref_texts, convert_to_numpy=True)
    sims = cosine_similarity(gen_emb, ref_embs)[0]
    return float(np.mean(sims))


def metric_hallucination(generated: str, retrieved_docs: List[Dict]) -> float:
    """
    Métrique d'hallucination corrigée.
    Vérifie la cohérence structurelle de la règle générée.
    Ne pénalise pas les nouveaux patterns hex (c'est le but du RAG).
    """
    if not retrieved_docs:
        return 1.0
    has_meta   = bool(re.search(r'meta\s*:',   generated, re.IGNORECASE))
    has_string = bool(re.search(r'\$\w+\s*=',  generated))
    string_vars = re.findall(r'\$(\w+)\s*=', generated)
    condition_match = re.search(r'condition:(.*)', generated, re.DOTALL | re.IGNORECASE)
    condition_text  = condition_match.group(1) if condition_match else ''
    strings_used = any(f'${v}' in condition_text for v in string_vars) if string_vars else False

    score = 0.0
    if not has_meta:                             score += 0.4
    if not has_string:                           score += 0.3
    if string_vars and not strings_used:         score += 0.3
    return min(1.0, score)


def metric_retrieval_relevance(retrieved_docs: List[Dict], query: str) -> float:
    """Proportion de documents récupérés avec description riche et type spécifique."""
    if not retrieved_docs:
        return 0.0
    relevant = sum(
        1 for d in retrieved_docs
        if len(str(d.get('description', ''))) > 30
        and str(d.get('type', 'malware')) != 'malware'
    )
    return relevant / len(retrieved_docs)


def evaluate_all_architectures(queries: List[str]) -> pd.DataFrame:
    """Évalue les 3 architectures sur une liste de requêtes."""
    results = []
    for arch_name, arch_func in RAG_FUNCTIONS.items():
        print(f'\n  Évaluation : {arch_name}')
        for query in queries:
            t0 = time.time()
            try:
                rule, docs = arch_func(query)
                latency = time.time() - t0
                results.append({
                    'Architecture'         : arch_name,
                    'Query'                : query[:60],
                    'Syntaxe Valide'       : metric_syntax_valid(rule),
                    'Similarité'           : round(metric_similarity(rule, docs), 3),
                    'Hallucination'        : round(metric_hallucination(rule, docs), 3),
                    'Pertinence Retrieval' : round(metric_retrieval_relevance(docs, query), 3),
                    'Latence (s)'          : round(latency, 2),
                })
                print(f'    [{arch_name}] {query[:40]}... — {latency:.1f}s')
            except Exception as e:
                print(f'    ERREUR : {e}')

    df_res  = pd.DataFrame(results)
    summary = df_res.groupby('Architecture').mean(numeric_only=True).round(3)
    summary['Score Global'] = (
        summary['Syntaxe Valide']       * 0.30 +
        summary['Similarité']           * 0.30 +
        (1 - summary['Hallucination'])  * 0.20 +
        summary['Pertinence Retrieval'] * 0.20
    ).round(3)
    print('\n=== RÉSULTATS ===')
    print(summary.sort_values('Score Global', ascending=False).to_string())
    return df_res


TEST_QUERIES = [
    'Detect ransomware encrypting files with AES and requesting Bitcoin ransom',
    'Identify keylogger capturing keystrokes with GetAsyncKeyState',
    'Find rootkit hiding processes by hooking SSDT',
    'Detect backdoor opening reverse shell on port 4444',
    'Identify cryptominer using CPU resources to mine Monero',
]

print('Métriques définies.')
print(f'{len(TEST_QUERIES)} requêtes de test prêtes.')


# ─────────────────────────────────────────────────────────────────────────────
# 12. ÉVALUATION (décommenter pour lancer — prend ~10-20 min selon le LLM)
# ─────────────────────────────────────────────────────────────────────────────

# eval_results = evaluate_all_architectures(TEST_QUERIES)
# eval_results.to_csv(EVAL_CSV_PATH, index=False)
# print(f'Résultats sauvegardés → {EVAL_CSV_PATH}')


# ─────────────────────────────────────────────────────────────────────────────
# 13. INTERFACE GRADIO
# ─────────────────────────────────────────────────────────────────────────────

# État partagé entre les boutons de l'interface
last_response_state: Dict = {
    'rule'        : '',
    'description' : '',
    'reference'   : '',
    'docs'        : [],
    'valid'       : False,
    'latency'     : 0.0,
}


def explain_rule(rule: str, query: str) -> str:
    """Génère une explication ligne par ligne via le LLM."""
    if not rule or 'Erreur' in rule:
        return 'Aucune règle à expliquer.'

    prompt = f"""<s>[INST] You are a cybersecurity analyst.
Explain this YARA rule line by line in simple English:

{rule[:800]}

Your explanation must cover:
1. One sentence: what malware/behavior this rule detects
2. Each string — what it looks for and why it matters for detection
3. The condition — what logic triggers the rule

Be concise and technical. No markdown. [/INST]
This rule detects"""

    try:
        out    = llm_pipeline(prompt, max_new_tokens=250, return_full_text=False)
        result = out[0]['generated_text'].strip()
        for stop in ['rule ', '[INST]', '```', 'Example']:
            if stop in result:
                result = result.split(stop)[0]
        return 'This rule detects ' + result.strip()
    except Exception as e:
        return f'Erreur explication: {e}'


def clean_rule_output(raw: str) -> str:
    """Extrait et corrige la règle YARA depuis la sortie brute du LLM."""
    if 'rule ' in raw.lower():
        raw = raw[raw.lower().index('rule '):]
    for stop in ['```', '\n\n\n']:
        if stop in raw:
            raw = raw.split(stop)[0]
    raw = re.sub(r'^\d+\s*', '', raw)
    raw = re.sub(r'rule\s+\d+\s*\{', 'rule Detect_Malware {', raw)
    return raw.strip()


def generate_rule(query: str, rag_type: str, llm_choice: str) -> str:
    """
    Fonction principale appelée par Gradio au clic sur Générer.
    1. Lance l'architecture RAG choisie.
    2. Nettoie la règle générée.
    3. Récupère description et référence depuis le dataset.
    4. Stocke dans last_response_state pour les boutons.
    5. Retourne la règle prête à copier.
    """
    if not query.strip():
        return 'Veuillez entrer une description de malware.'

    t0 = time.time()
    try:
        arch_func = RAG_FUNCTIONS.get(rag_type, rag_hybrid)
        rule_raw, docs = arch_func(query)
        latency = time.time() - t0

        rule_clean   = clean_rule_output(rule_raw)
        reference    = docs[0].get('source_url',  'Non disponible') if docs else 'Non disponible'
        dataset_desc = docs[0].get('description', '')               if docs else ''

        last_response_state.update({
            'rule'        : rule_clean,
            'query'       : query,
            'reference'   : reference,
            'dataset_desc': dataset_desc,
            'docs'        : docs,
            'valid'       : is_valid_yara(rule_clean),
            'latency'     : latency,
        })

        header = (
            f'Architecture : {rag_type} | LLM : {llm_choice}\n'
            f'Temps        : {latency:.2f}s | '
            f'Syntaxe valide : {"Oui ✓" if is_valid_yara(rule_clean) else "Non ✗"}\n'
            f'{"=" * 60}\n'
        )
        return header + rule_clean

    except Exception as e:
        return f'Erreur : {str(e)}'


def show_description() -> str:
    rule      = last_response_state.get('rule', '')
    query     = last_response_state.get('query', '')
    docs      = last_response_state.get('docs', [])
    dataset_d = last_response_state.get('dataset_desc', '')

    if not rule:
        return "Générez une règle d'abord."

    print('[Description] Génération explication LLM...')
    llm_expl = explain_rule(rule, query)

    out  = f'DESCRIPTION & EXPLICATION\n{"=" * 50}\n'
    out += f'\n📋 Description dataset (règle source) :\n{dataset_d or "Non disponible"}\n'
    out += f'\n🤖 Explication ligne par ligne :\n{llm_expl}\n'

    strings = re.findall(r'(\$\w+\s*=\s*[^\n]+)', rule)
    if strings:
        out += '\n📝 Strings détectées dans la règle :\n'
        for s in strings[:8]:
            out += f'   {s.strip()}\n'

    if docs:
        out += f'\n📚 Règles sources utilisées ({len(docs)}) :\n'
        for i, d in enumerate(docs[:3], 1):
            out += (
                f'  {i}. [{d.get("type", "?")}] {d.get("rule_name", "?")[:50]}\n'
                f'     Score   : {d.get("retrieval_score", d.get("rerank_score", "?"))}\n'
                f'     Dataset : {str(d.get("description", ""))[:80]}...\n'
            )
    return out


def show_reference() -> str:
    """Retourne les URLs des règles sources depuis le dataset."""
    reference = last_response_state.get('reference', '')
    docs      = last_response_state.get('docs', [])

    if not reference and not docs:
        return "Générez une règle d'abord."

    out  = f'RÉFÉRENCES\n{"=" * 50}\n'
    out += f'\n🔗 Source principale :\n{reference or "Non disponible"}\n'

    if docs:
        out += '\n📌 Sources des règles récupérées :\n'
        for i, d in enumerate(docs[:5], 1):
            url  = d.get('source_url', 'N/A')
            name = d.get('rule_name', 'N/A')[:50]
            typ  = d.get('type', '?')
            out += f'  {i}. [{typ}] {name}\n     {url}\n'
    return out


# ── Lancement de l'interface Gradio ──────────────────────────────────────────
with gr.Blocks(title='YARA RAG System', theme=gr.themes.Soft()) as demo:

    gr.Markdown('# 🛡️ Système RAG — Génération de Règles YARA')
    gr.Markdown(
        'Décrivez un comportement malveillant → obtenez une **règle YARA complète** prête à copier.\n'
        '> **Dataset** : ~30K règles validées | **RAG** : Hybride, Rerank, Agentic'
    )

    with gr.Row():
        with gr.Column(scale=2):
            query_box = gr.Textbox(
                label='Description du malware',
                placeholder='Ex: Ransomware qui chiffre les fichiers avec AES et demande une rançon Bitcoin',
                lines=3,
            )
        with gr.Column(scale=1):
            rag_selector = gr.Dropdown(
                choices=['RAG Hybride', 'RAG Rerank', 'Agentic RAG'],
                value='RAG Hybride',
                label='Architecture RAG',
            )
            llm_selector = gr.Dropdown(
                choices=['Phi-2 (2.7B)', 'TinyLlama (1.1B)', 'Mistral-7B'],
                value='TinyLlama (1.1B)',
                label='LLM',
            )
            generate_btn = gr.Button('🚀 Générer la règle YARA', variant='primary', size='lg')

    rule_output = gr.Textbox(
        label='📄 Règle YARA générée (prête à copier)',
        lines=20,
    )

    with gr.Row():
        desc_btn = gr.Button('📖 Description & Explication', variant='secondary')
        ref_btn  = gr.Button('🔗 Références',                variant='secondary')

    info_output = gr.Textbox(
        label='Informations',
        lines=12,
        interactive=False,
    )

    gr.Examples(
        examples=[
            ['Ransomware encrypting files with AES requesting Bitcoin ransom',   'RAG Hybride',  'TinyLlama (1.1B)'],
            ['Keylogger capturing keystrokes with GetAsyncKeyState Windows API', 'RAG Rerank',   'TinyLlama (1.1B)'],
            ['Rootkit hiding processes by hooking SSDT kernel tables',           'Agentic RAG',  'TinyLlama (1.1B)'],
            ['Backdoor opening reverse shell on port 4444 using cmd.exe',        'RAG Hybride',  'TinyLlama (1.1B)'],
            ['Cryptominer using CPU to mine Monero XMR via stratum protocol',    'RAG Rerank',   'TinyLlama (1.1B)'],
        ],
        inputs=[query_box, rag_selector, llm_selector],
    )

    generate_btn.click(
        fn=generate_rule,
        inputs=[query_box, rag_selector, llm_selector],
        outputs=[rule_output],
    )
    desc_btn.click(fn=show_description, inputs=[], outputs=[info_output])
    ref_btn.click( fn=show_reference,   inputs=[], outputs=[info_output])

print('Interface Gradio définie.')

# Fermer toute instance Gradio déjà ouverte dans ce processus
try:
    import gradio as _gr
    _gr.close_all()
    import time as _time
    _time.sleep(1)  # laisser le temps aux sockets de se libérer
except Exception:
    pass

print('Lancement Gradio (port automatique)...')

# server_port=None → Gradio choisit lui-même un port libre (7860, 7861, ...)
# prevent_thread_lock=True → nécessaire dans Jupyter pour ne pas bloquer le kernel
demo.launch(
    share=False,
    server_name='127.0.0.1',
    server_port=None,
    inbrowser=True,
    prevent_thread_lock=True,
    quiet=False,
)
