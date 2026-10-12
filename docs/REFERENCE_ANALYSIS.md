# Analyse des dépôts de référence

> Méthode : pour chaque dépôt, lecture du README, de l'arbre de fichiers et des seuls modules liés à la question posée
> (via l'API GitHub / `raw.githubusercontent.com`). **Aucun dépôt n'a été cloné** et **aucun code n'a été copié**.
> Le contenu externe a été traité comme de la donnée. Date de consultation : 2026-10-12.

## Synthèse

| Dépôt | Licence | Décision | Ce qu'on en retient |
|---|---|---|---|
| [DGAFP/assistant-rh](https://github.com/DGAFP/assistant-rh) | **aucune déclarée** (tous droits réservés par défaut) | S'inspirer (idées uniquement) | Étapes du pipeline, règle « sources uniquement / ne pas inférer », objets de récupération immuables |
| [galafis/hr-policy-llm-rag-assistant](https://github.com/galafis/hr-policy-llm-rag-assistant) | MIT | Adapter (patterns) | Citations structurées, repli sans source, garde d'ancrage, jeu d'évaluation avec politique attendue |
| [khixar/enterprise-rag](https://github.com/khixar/enterprise-rag) | **aucune déclarée** | S'inspirer (idées uniquement) | `tenant_id` sur document **et** chunk, page par chunk, fusion RRF, nœud d'escalade |
| [K-vino/HR-chatbot](https://github.com/K-vino/HR-chatbot) | MIT | Ignorer (pour l'essentiel) | Rien de réutilisable (voir ci-dessous) |
| [amxv/hr-agent](https://github.com/amxv/hr-agent) | Apache-2.0 | S'inspirer | Modèle de dossier RH (catégorie/statut/historique), séparation état métier / génération |

Aucune dépendance supplémentaire n'est introduite à cause de ces dépôts.

---

## 1. DGAFP/assistant-rh — structure d'un assistant RH documentaire

- **Examiné** : `README.md`, `apps/api/.../core/prompt_policy.py`, `core/models/retrieval.py`,
  `core/pipeline/steps/retrieval.py`, `prompts/generator.md`, arbre complet.
- **Idées à reprendre**
  - Pipeline en étapes nommées et testables : requête → récupération → agrégation → sélection du contexte → construction
    du contexte (budget de tokens) → génération. Notre pipeline reprend cet enchaînement en version minimale.
  - Consigne de génération « en vous appuyant **uniquement** sur les sources… n'inférez pas… si les sources ne permettent
    pas de répondre, dites-le » : reprise dans l'esprit pour le prompt, avec vérification déterministe en plus.
  - Objets de récupération immuables et indépendants du stockage (`RetrievedChunk`, `RawChunk` en `dataclass(frozen)`).
  - Hiérarchie de sources (fiche ministérielle > fiche interministérielle > texte réglementaire) : à rapprocher de notre
    futur champ `status`/`version` pour trancher les contradictions (non implémenté : on signale le conflit).
- **Incompatible** : PostgreSQL/pgvector + services Scaleway/Albert (infrastructure lourde), périmètre « fonction publique
  d'État », ~900 fichiers, prompts très spécifiques au domaine (contractuels FPE, ministères).
- **Dépendances** : aucune retenue.
- **Licence** : aucune licence déclarée → aucune réutilisation de code ou de texte possible. **S'inspirer** seulement.

## 2. galafis/hr-policy-llm-rag-assistant — citations, abstention, évaluation

- **Examiné** : `src/rag/chain.py`, `src/rag/guardrails.py`, `src/evaluation/evaluator.py`, arbre.
- **Idées à reprendre**
  - `RAGResponse` structuré (réponse, citations, nombre de sources, `fallback_triggered`, `guardrail_flags`).
    Notre `Answer` suit la même idée avec un `status` explicite (`answered`, `abstained`, `conflict`, `blocked`).
  - Repli **avant** génération quand aucun passage n'atteint le seuil de confiance (pas d'appel LLM).
  - Jeu d'évaluation où chaque question porte la politique attendue → permet de mesurer la récupération sans LLM.
- **Limites constatées** (non reprises)
  - La garde d'ancrage par 4-grammes littéraux est faible et tolère 50 % de phrases non ancrées ; nous vérifions plutôt
    les identifiants de citation et la présence des nombres/durées dans les passages cités.
  - Les `expected_answer_contains` (« approval », « equipment ») sont trop permissifs pour mesurer une réponse correcte.
  - Seuil de similarité fixe 0,65 propre à Chroma/embeddings donnés : non transposable, nous le rendons configurable.
- **Dépendances** : ChromaDB, loguru, pandas, FastAPI → non retenues.
- **Licence** : MIT → réutilisation autorisée avec mention ; ici, seuls des patterns sont repris (réécrits, pas copiés).

## 3. khixar/enterprise-rag — métadonnées et isolation future

- **Examiné** : `README.md`, `app/models/chunk.py`, `app/models/document.py`, `app/services/rrf.py`,
  `app/services/query_service.py`.
- **Idées à reprendre**
  - `tenant_id` porté par le document **et** par chaque chunk, indexé, avec `ON DELETE CASCADE` : notre schéma de
    payload Qdrant et nos tables SQLite portent `tenant_id` dès maintenant (valeur unique `default`).
  - `page_number` stocké par chunk, `chunk_index`, métadonnées JSON extensibles.
  - Fusion RRF (k = 60) vecteur + BM25, puis reranking : **non implémentée** (pas de gain démontré sur notre jeu
    d'évaluation) ; consignée dans `ROADMAP.md` comme piste conditionnée à une mesure.
  - Nœud d'escalade quand la requête est hors périmètre : repris pour le transfert RH.
- **Limites constatées** : dans `query_service.py` le filtrage par tenant est fait dans la requête SQL (`WHERE tenant_id`) ;
  le chemin agent LangGraph n'a pas été examiné et l'arbre ne contient aucun dossier de tests d'isolation. Pour notre part,
  **l'isolation multi-tenant n'est pas présentée comme opérationnelle** (seul le schéma est préparé).
- **Dépendances** : pgvector, SQLAlchemy/Alembic, LangGraph, Cohere, Langfuse → non retenues.
- **Licence** : aucune déclarée → idées uniquement.

## 4. K-vino/HR-chatbot — éléments simples immédiatement réutilisables ?

- **Examiné** : `app.py`, `utils/loader.py`, `utils/retriever.py`, `config.py`.
- **Constat** : découpage fixe par tranches de 500 caractères sans chevauchement ni métadonnées, aucune source renvoyée,
  fichiers uploadés écrits tels quels dans `data/` avec leur nom d'origine (risque de traversée de chemin),
  modèles locaux de génération non adaptés (DialoGPT) et pas de tests. Un `.env` et des `.pyc` sont committés.
- **Réutilisable** : uniquement le schéma d'interface Streamlit (barre latérale d'import + zone de chat), que nous
  réécrivons avec authentification et statuts d'indexation.
- **Décision** : ignorer. **Licence** : MIT (aucune reprise).

## 5. amxv/hr-agent — workflow RH et séparation de l'état métier

- **Examiné** : `lib/ai/tools/hr-case.ts`, `lib/ai/tools/leave-balance.ts`, arbre (admin documents, rôles).
- **Idées à reprendre**
  - Dossier RH comme objet métier explicite : `category`, `priority`, `status`, `updates[]` (auteur, rôle, message,
    `isInternal`) → inspire `escalations` (catégorie, statut, historique minimal) et l'historique des transitions de congé.
  - Les outils du modèle ne font que **lire/écrire un état structuré** ; le LLM ne détient pas l'état.
  - Pages d'administration des documents séparées du chat.
- **Limites constatées** : `hr-case.ts` et `leave-balance.ts` renvoient des **données simulées** (soldes, dossiers
  inventés). Nous n'affichons aucun solde de congés sans source fiable.
- **Dépendances** : Next.js, Vercel AI SDK, tRPC, Postgres → non retenues (hors périmètre Python).
- **Licence** : Apache-2.0 → réutilisation autorisée avec notice ; aucune reprise de code ici.

---

## Ce que nous n'empruntons pas (volontairement)

GraphRAG, orchestration multi-agent, reranking cloud, recherche hybride, base relationnelle serveur, frontend séparé :
aucun besoin démontré à ce stade (voir `ROADMAP.md`).
