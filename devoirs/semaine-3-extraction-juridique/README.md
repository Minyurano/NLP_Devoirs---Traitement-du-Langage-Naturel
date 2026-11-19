# Semaine 3 — Extraction d'informations depuis un texte juridique

Extraction structurée des infractions et amendes à partir d'un document juridique scanné (Bulletin Officiel), puis analyse.

| Étape | Outil |
|---|---|
| Conversion PDF → image | pypdfium2 |
| OCR | Tesseract (`fra`) |
| Nettoyage et segmentation par article | expressions régulières |
| Extraction des amendes (min / max) et des classes | expressions régulières |
| Regroupement des infractions | TF-IDF + KMeans (5 clusters) |
| Reconnaissance d'entités | spaCy `fr_core_news_sm` + `EntityRuler` personnalisé |
| Export | `export_final.csv` |

**Notebook :** [`extraction_juridique.ipynb`](extraction_juridique.ipynb) (exécuté, avec résultats)

**Pistes restant à traiter** (liste « TO DO » de l'énoncé, en fin de notebook) : nouveaux patterns regex, colonnes supplémentaires (localisation, catégorie de véhicule…), comparaison approche à base de règles vs BERT.
