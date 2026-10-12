# Audit initial du dépôt HRagentAssistant

> Audit ciblé réalisé avant toute modification de code (état du dépôt : branche `main`, commit `471ce09`,
> working tree propre). Les constats marqués ✔ ont été vérifiés par exécution, les autres par lecture du code.

## 1. Architecture existante

Pipeline RAG linéaire en petits modules « une étape = un fichier », construit avec LangChain :

```
data/hr_policy.txt ─ document_loader (TextLoader) ─ splitter (RecursiveCharacterTextSplitter 500/60)
   ─ embeddings (JinaEmbeddings, jina-embeddings-v2-base-en) ─ vector_store (Qdrant Cloud)
   ─ tools (@tool search_hr_policy) ─ agent (langchain.agents.create_agent + system prompt)
   ─ llm/gateway (ChatOpenAI → Portkey → Groq "openai/gpt-oss-20b")
   ─ guardrails (ChatGroq "gpt-oss-safeguard-20b", entrée + sortie) ─ pipeline.ask()
   ─ evaluation (LangSmith dataset + openevals LLM-judge) ; tracing (LangSmith, log seulement)
```

| Fichier | Rôle | Verdict |
|---|---|---|
| `config.py` | Constantes + variables d'environnement | Réutilisable, à corriger (clés obligatoires, valeurs en dur) |
| `document_loader.py` | `TextLoader` sur un seul .txt | Incomplet (pas de PDF/DOCX, pas de métadonnées) |
| `splitter.py` | Découpage récursif 500/60 | Réutilisable, métadonnées à enrichir |
| `embeddings.py` | `JinaEmbeddings` | Réutilisable ; modèle EN uniquement |
| `vector_store.py` | Qdrant **Cloud uniquement** | À étendre (local/mémoire, filtres) |
| `tools.py` / `agent.py` | Tool de recherche + agent LangChain | À remplacer (voir §4) |
| `llm.py` / `gateway.py` | Groq via Portkey (slug `@hrpolicy` en dur) | Réutilisable, à rendre optionnel |
| `guardrails.py` | Garde LLM en entrée/sortie | Utile en défense en profondeur, pas comme autorité |
| `pipeline.py` | Assemblage + `ask()` | À réécrire (flux déterministe) |
| `evaluation.py` | LangSmith + LLM-judge, 10 questions EN | Non exécutable localement |
| `logger.py` | Logging fichier + console | À corriger (effets de bord à l'import, contenu loggé) |
| `tracing.py` | Log de l'état LangSmith | OK, inchangé |
| `main.py`, `rag.py`, `README.md` | Placeholder / fichier vide / fichier vide | Sans contenu |
| `rag.ipynb` | Prototype pas-à-pas avec FAISS | Doublon des modules, dérive (voir P2) |

## 2. Fonctionnalités réellement opérationnelles

**Aucune de bout en bout dans l'état initial.**

- ✔ `config.check_api_keys()` échoue avec l'environnement fourni (`.env` contient `GROQ_API_KEY` et `JINA_API_KEY`, mais pas
  `QDRANT_URL`/`QDRANT_API_KEY`/`PORTKEY_API_KEY`) : `build_hr_assistant()` ne peut pas démarrer.
- ✔ Le venv `venv/` ne contient que `numpy` : aucune dépendance du projet n'est installée.
- Il n'existe ni point d'entrée utilisable (`main.py` affiche « Hello »), ni interface (`pipeline.py` mentionne un
  `app.py` Streamlit qui n'existe pas), ni test.

## 3. Bugs et risques

### P0 — bloquants (fonctionnement / sécurité)

| # | Constat | Preuve |
|---|---|---|
| P0-1 | Impossible de démarrer sans Qdrant Cloud **et** Portkey (clés exigées sans condition). | ✔ `check_api_keys()` |
| P0-2 | `guardrails.py` instancie `ChatGroq` **à l'import** : importer `pipeline` sans clé Groq plante. | ✔ `GroqError` |
| P0-3 | Aucun contrôle d'accès : pas d'utilisateur, pas de rôle, pas de filtrage de documents. Toute personne qui atteint l'agent voit tout le corpus. | lecture |
| P0-4 | Le contenu récupéré est concaténé sans séparation ni métadonnées : une instruction cachée dans un document est traitée comme du contexte « de confiance », et aucune source n'est exploitable. | `tools.py` |
| P0-5 | Les questions et réponses complètes (données RH potentiellement personnelles) sont écrites en clair dans `logs/*.log`. | `pipeline.ask`, `guardrails` |

### P1 — indispensables au MVP

| # | Constat |
|---|---|
| P1-1 | Pas de réponses sourcées vérifiables : le tool renvoie du texte brut, aucune citation, aucune vérification. |
| P1-2 | Pas d'abstention déterministe : l'unique garde-fou est « dis que tu ne sais pas » dans le prompt, déléguée au LLM. |
| P1-3 | L'agent décide lui-même d'appeler (ou non) l'outil de recherche : comportement non testable hors ligne. |
| P1-4 | Un seul fichier en dur, un seul index ; `vector_store_exists()` réutilise l'index sans vérifier qu'il correspond au document (index périmé silencieux). Pas de gestion de documents (statut, version, suppression, réindexation). |
| P1-5 | PDF/DOCX non pris en charge ; pas de métadonnées (page, section, version, langue, statut). |
| P1-6 | Aucun workflow métier (demande de congé, transfert RH), aucune persistance des données métier. |
| P1-7 | Aucun test ; `evaluation.py` dépend de LangSmith + Portkey + openevals (non installé par défaut), 10 cas EN seulement, pas de métrique déterministe ni de cas négatifs. |
| P1-8 | Deux sources de dépendances incohérentes (`pyproject.toml: dependencies = []` vs `requirements.txt`) ; `jupyter` en dépendance d'exécution ; `langchain` (méta-paquet) listé alors que seuls les sous-paquets servent. |
| P1-9 | Modèle d'embedding `jina-embeddings-v2-base-en` (anglais) pour des utilisateurs francophones. |
| P1-10 | `_check_safety` fait `json.loads` sans gestion d'erreur : une réponse non-JSON du garde fait planter la requête. |
| P1-11 | Pas de `.env.example`, README vide, pas d'instructions d'installation. |

### P2 — améliorations utiles

- `logger.py` crée `logs/` et ouvre un fichier **à l'import**, relatif au répertoire courant (✔ vérifié).
- Dérive notebook/modules : le notebook utilise FAISS (non déclaré), `gpt-oss-120b`, `chunk_overlap=50` contre 60 dans `config.py`.
- Slug Portkey `@hrpolicy` en dur dans `gateway.py` ; modèle de juge identique au modèle principal.
- `rag.py` vide et non référencé (conservé, voir ROADMAP).
- Corpus de démonstration en anglais, mono-document, non marqué comme exemple dans les métadonnées.

### P3 — évolutions futures

Multi-entreprise (tenant), SSO, notifications, intégration SIRH, recherche hybride/reranking si les évaluations le justifient,
multilingue avancé, GraphRAG (non justifié à ce stade).

## 4. Composants réutilisables et décisions

| Composant | Décision | Raison |
|---|---|---|
| `RecursiveCharacterTextSplitter` | **Conserver**, enrichir les métadonnées | Fonctionne, configurable |
| `JinaEmbeddings` | **Conserver** (option `jina`) | Intégration existante ; modèle rendu configurable |
| Qdrant | **Conserver**, ajouter mode local/mémoire | Même base, zéro service externe pour le prototype ; filtres de payload natifs |
| Portkey → Groq | **Conserver** (option `portkey`), ajouter Groq direct et mode extractif | Aucune dépendance obligatoire à un fournisseur payant |
| Prompts de `guardrails.py` | **Conserver** comme défense en profondeur (optionnelle) | Les règles dures passent côté backend |
| `agent.py` + `tools.py` | **Retirer** | L'agent choisit seul d'appeler la recherche → ni déterminisme, ni citations vérifiables, ni abstention testable. Récupération systématique puis génération. |
| `evaluation.py` (LangSmith) | **Conserver** (imports paresseux) + ajouter une évaluation locale déterministe | Le juge LLM reste utile, mais ne peut pas être la seule mesure |
| `rag.ipynb` | **Conserver tel quel** | Support pédagogique, hors périmètre MVP |

## 5. Dépendances

- **Conservées** : `python-dotenv`, `numpy`, `langchain-core`, `langchain-community` (JinaEmbeddings),
  `langchain-text-splitters`, `langchain-qdrant`, `qdrant-client`, `langchain-groq`, `langchain-openai`, `portkey-ai`, `streamlit`.
- **Ajoutées** : `pypdf` (PDF avec n° de page), `python-docx` (DOCX).
- **Retirées du socle** : `langchain` (méta-paquet, plus d'agent), `jupyter`/`ipykernel` (extra `notebook`).
- **Optionnelles** : `openevals`, `langsmith` (extra `eval`).
- Source unique : `pyproject.toml` ; `requirements.txt` ne fait que renvoyer vers lui.

## 6. Sécurité

| Sujet | État initial | Cible MVP |
|---|---|---|
| Secrets | `.env` ignoré par git ✔, aucune clé committée ✔ ; pas de `.env.example` | `.env.example` sans secret |
| Authentification / rôles | Inexistants | Comptes locaux (mots de passe hachés PBKDF2), rôles `employee` / `hr_manager`, contrôle côté backend |
| Isolation des documents | Inexistante | Filtre de payload (rôles autorisés, statut, tenant) appliqué **avant** la génération |
| Injection via documents | Non traitée | Contexte délimité et traité comme donnée, citations vérifiées, détection d'instructions, pas d'outil exécutant du code |
| Logs | Contenu complet des échanges | Identifiants et métriques uniquement, pas de texte de question/réponse |
| Erreurs | Exceptions brutes | Messages génériques côté UI, détail technique en log |

## 7. Écarts par rapport au MVP et priorités de correction

1. **Socle** (P0-1, P0-2, P1-8, P1-11) : config souple, dépendances cohérentes, installation reproductible.
2. **RAG fiable** (P0-4, P1-1 à P1-5, P1-9) : ingestion PDF/DOCX/TXT, métadonnées, filtrage, citations vérifiées, abstention.
3. **Sécurité** (P0-3, P0-5, P1-10) : auth locale, rôles, logs sans contenu, garde-fous déterministes.
4. **Workflows** (P1-6) : demande de congé (machine à états), transfert RH.
5. **Tests & évaluation** (P1-7) : tests unitaires/intégration/sécurité, jeu d'évaluation ≥ 30 cas.
