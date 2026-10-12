# Architecture

Backend Python modulaire, un seul processus, une seule base de données par type de donnée. Pas de service
supplémentaire, pas de framework d'agents, pas d'API réseau : l'interface Streamlit appelle directement les services.

```
Utilisateur ─▶ app.py (Streamlit : connexion, 3 pages)
                  │
                  ▼
        pipeline.HRAssistantApp ──────────────── auth.py (utilisateurs, rôles, permissions)
                  │
   ┌──────────────┼───────────────────┬────────────────────┐
   ▼              ▼                   ▼                    ▼
AnswerEngine   DocumentService     LeaveService     EscalationService
(answering.py) (documents.py)      (leave.py)       (escalation.py)
   │              │                   │                    │
   │      document_loader.py          └──── SQLite (db.py) ┘
   │      splitter.py  ─┐
   ▼                    ▼
KnowledgeIndex (vector_store.py) ── Qdrant (memory | local | cloud) + filtre d'accès
   │
   └─ embeddings.py : hashing (hors ligne) | jina
   guardrails.py · llm.py / gateway.py (optionnels) · evaluation.py
```

## Pipeline RAG

| Étape | Module | Nature |
|---|---|---|
| Validation du fichier (type, taille, signature) | `document_loader.validate_upload` | déterministe |
| Extraction (TXT, PDF par page, DOCX avec titres) | `document_loader` | déterministe |
| Découpage par sections, titre du document répété dans chaque passage | `splitter` | déterministe, configurable |
| Métadonnées (document, version, famille, statut, langue, page, section, rôles autorisés, tenant) | `documents`, `splitter` | déterministe |
| Embeddings → Qdrant | `embeddings`, `vector_store` | `hashing` ou `jina` |
| **Recherche avec filtre d'accès (tenant, rôle, statut) appliqué par la base** | `vector_store.access_filter` | déterministe |
| Neutralisation des instructions dans les passages | `guardrails.neutralize_passage` | déterministe |
| Preuve suffisante : score + couverture lexicale pondérée par la rareté + terme directeur retrouvé | `answering` | déterministe |
| Conflit : deux documents actifs d'une même famille aux chiffres différents | `answering._detect_conflict` | déterministe |
| Génération : modèle (si configuré) ou citation de phrases | `answering`, `llm` | modèle optionnel |
| Vérification : marqueurs ⊂ passages récupérés, chiffres présents dans les passages cités, pas de promesse/lien inconnu | `answering.verify_generated_answer`, `guardrails.check_answer` | déterministe |
| Réponse **ou** abstention + proposition de transfert | `answering` | — |

Le LLM ne décide **ni** des droits **ni** des statuts : s'il échoue, est indisponible ou produit une sortie non
vérifiable, la réponse est remplacée par des citations littérales.

## Décisions techniques

| Décision | Raison | Alternative écartée |
|---|---|---|
| Récupération systématique puis génération (suppression de `agent.py`/`tools.py`) | L'agent décidait seul d'appeler la recherche : non déterministe, sans citations ni abstention testables | Garder `create_agent` |
| Qdrant conservé, mode embarqué `local`/`memory` | Même base qu'avant, aucun serveur, filtres de payload natifs, tests en mémoire | FAISS, pgvector, Chroma |
| SQLite (stdlib) pour les données métier | Local, transactions, zéro dépendance ; schéma portable vers PostgreSQL | Fichiers JSON, base serveur |
| Streamlit conservé comme interface | Déjà dans les dépendances, suffisant pour le MVP | Next.js + API FastAPI |
| Authentification locale PBKDF2 (stdlib) | Séparer réellement les rôles sans dépendance ; assumée « prototype » | Liste déroulante de rôles |
| Client Jina en `httpx` (≈30 lignes) | `langchain-community` est en fin de vie et n'était utilisé que pour ça | Garder `langchain-community` |
| Routage d'intention par règles | Il choisit une *action backend* : doit être déterministe et testable | Appel LLM |
| Pas de reranking, pas de recherche hybride, pas de GraphRAG | Aucun gain démontré sur le jeu d'évaluation | — |

## Dimensionnement du découpage

Les sections du corpus de démonstration font 100–700 caractères ; `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100` garde chaque
section entière (une règle RH coupée au milieu est pire qu'un passage un peu long). Les sections plus longues sont
découpées par `RecursiveCharacterTextSplitter` avec répétition du titre. À recalibrer sur vos propres documents.

## Modèle de données (SQLite)

`users`, `documents` (registre : nom, type, sha256, version, famille, statut, niveau d'accès, langue, statut
d'indexation, erreur, nombre de passages, collection), `leave_requests` + `leave_events` (historique des transitions),
`escalations`. Toutes les tables portent `tenant_id`. **L'isolation multi-entreprise n'est pas opérationnelle** : le
filtre par tenant est appliqué à la recherche et au registre (et testé), mais il n'existe ni administration de tenants
ni configuration/clés par tenant.

## Extension

- Autre fournisseur de modèle : ajouter une branche dans `llm.get_llm` (interface : `.invoke(messages)`).
- Autre magasin de vecteurs : implémenter `KnowledgeIndex` (`add_chunks`, `delete_document`, `search`, `term_stats`).
- Autre processus métier : un service par processus, avec sa machine à états explicite et ses tests de transitions.
