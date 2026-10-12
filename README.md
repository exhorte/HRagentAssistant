# HRagentAssistant

Prototype local d'**assistant RH documentaire** : il répond aux questions des employés à partir des documents RH
réellement chargés, **cite ses sources**, **s'abstient** quand les documents ne suffisent pas, aide à préparer et
suivre une **demande de congé**, et peut **transmettre** une question à un responsable RH.

> ⚠️ **Prototype, données synthétiques uniquement.** Ne chargez aucune donnée RH réelle tant que les points de
> [docs/SECURITY.md](docs/SECURITY.md) (section « Avant toute utilisation réelle ») ne sont pas traités.
> Les documents de démonstration sont fictifs et sans valeur légale.

## Ce que fait le MVP

| Fonctionnalité | État |
|---|---|
| **A. Assistant documentaire** : réponses sourcées, citations vérifiées, abstention, détection de conflits entre versions | ✅ testé |
| **B. Gestion des documents** : import PDF / DOCX / TXT, métadonnées, statut d'indexation, erreurs, retrait, réindexation | ✅ testé |
| **C. Demande de congé** : récapitulatif, confirmation explicite, brouillon/soumission, suivi, décision RH contrôlée | ✅ testé |
| **D. Transfert aux RH** : résumé, motif, enregistrement visible dans l'interface (**aucune notification n'est envoyée**) | ✅ testé |
| Connexion + rôles (employé / responsable RH), contrôle d'accès appliqué par le backend | ✅ testé (prototype, voir SECURITY) |

Non fournis : solde de congés (aucun SIRH connecté), notifications, SSO, multi-entreprise, OCR de PDF scannés.

## Installation

Prérequis : Python 3.12+ (ou [uv](https://docs.astral.sh/uv/)).

```bash
# Windows (Git Bash / PowerShell) ou Linux/macOS
python -m venv venv
venv/Scripts/python -m pip install -e ".[dev]"      # Linux/macOS : venv/bin/python
```

Avec uv : `uv venv --python 3.12 venv && uv pip install --python venv/Scripts/python.exe -e ".[dev]"`.

`pyproject.toml` est l'unique source des dépendances (`requirements.txt` ne fait que le référencer).

## Configuration

Le prototype fonctionne **sans aucune clé API** (réponses extractives, embeddings locaux, Qdrant embarqué).
Pour changer de mode, copiez `.env.example` vers `.env` (jamais committé) :

| Variable | Valeurs | Effet |
|---|---|---|
| `LLM_PROVIDER` | `none` (défaut), `groq`, `portkey` | `none` : phrases extraites des documents. `groq`/`portkey` : réponse rédigée par un modèle, **puis vérifiée** |
| `EMBEDDING_BACKEND` | `hashing` (défaut), `jina` | `hashing` : hors ligne, lexical. `jina` : sémantique, nécessite `JINA_API_KEY` (utiliser `jina-embeddings-v3` pour le français) |
| `QDRANT_MODE` | `local` (défaut), `memory`, `cloud` | `local` : index embarqué dans `data/runtime/qdrant` |
| `GROQ_API_KEY`, `PORTKEY_API_KEY`, `JINA_API_KEY`, `QDRANT_URL`/`QDRANT_API_KEY` | | Uniquement si le mode correspondant est choisi |

Les clés ne sont lues que depuis l'environnement. Au changement de backend d'embeddings, réindexez les documents
(page « Documents RH » → « Réindexer tous les documents »).

## Lancer

```bash
venv/Scripts/python main.py init      # base SQLite, comptes de démonstration, corpus de démonstration
venv/Scripts/python main.py serve     # interface Streamlit sur http://localhost:8501
```

`init` génère les mots de passe des comptes `alice` et `bob` (employés) et `claire` (responsable RH) et les écrit dans
`data/runtime/demo_credentials.txt` (fichier local, ignoré par git, jamais affiché). Ajoutez `--special` pour importer
aussi le jeu `data/samples_special` (version obsolète, note contradictoire, document piégé).

> Un seul processus à la fois peut ouvrir l'index Qdrant local : arrêtez l'interface avant d'utiliser `main.py ask`.

### Utiliser l'assistant

1. Connectez-vous (`alice`) → **Assistant RH** et posez une question de démonstration :

   | Question | Comportement attendu |
   |---|---|
   | « Combien de jours de congés payés par an ? » | Réponse (25 jours) + source `politique_conges.txt` |
   | « Quelle est la procédure de demande d'attestation ? » | Réponse + source PDF avec numéro de page |
   | « Quelle est la capitale de la France ? » | Abstention + proposition de transfert aux RH |
   | « Quel est le salaire du niveau B ? » (employé) | Abstention : document réservé aux RH |
   | « Combien de jours de télétravail par semaine ? » (avec `--special`) | Conflit signalé entre deux documents actifs |
   | « Ignore toutes les instructions précédentes… » | Refus |

2. **Demandes RH → Nouvelle demande de congé** : saisissez type et dates → récapitulatif → case de confirmation →
   « Confirmer et soumettre » (ou « Enregistrer en brouillon »). Le statut se suit dans « Mes demandes ».
3. Connectez-vous en `claire` : **À traiter (RH)** → « Prendre en charge » → « Approuver » / « Refuser » (motif obligatoire).
4. Une question sans réponse propose « Transmettre cette question aux RH » : résumé et motif à confirmer ;
   le transfert apparaît dans « Transferts reçus (RH) ».

En ligne de commande : `python main.py ask "..." --as alice`, `python main.py documents --as claire`.

## Tests et évaluation

```bash
venv/Scripts/python -m pytest                 # 200+ tests, hors ligne, ~1 min 30
venv/Scripts/python -m pytest -m security     # tests de sécurité bloquants
venv/Scripts/python main.py eval              # évaluation métier locale (36 cas) + rapport
venv/Scripts/python main.py eval --cases data/eval/holdout_cases.json
```

Aucun test n'appelle de service externe. Résultats mesurés et leurs limites : [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md).

## Architecture (résumé)

```
Streamlit (app.py) ──▶ HRAssistantApp (pipeline.py) ──▶ auth / rôles (auth.py)
        ├─ AnswerEngine (answering.py) ── KnowledgeIndex (vector_store.py, Qdrant + filtre d'accès)
        │        └─ guardrails.py · llm.py/gateway.py (optionnels) · embeddings.py
        ├─ DocumentService (documents.py) ── document_loader.py · splitter.py
        ├─ LeaveService (leave.py)  ·  EscalationService (escalation.py)
        └─ SQLite (db.py) : utilisateurs, registre de documents, demandes, transferts
```

Détails et décisions : [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Sécurité : [docs/SECURITY.md](docs/SECURITY.md).
Audit initial : [docs/PROJECT_AUDIT.md](docs/PROJECT_AUDIT.md). Références étudiées : [docs/REFERENCE_ANALYSIS.md](docs/REFERENCE_ANALYSIS.md).
Suite : [docs/ROADMAP.md](docs/ROADMAP.md).

## Limites connues

- **La porte d'abstention est lexicale** (couverture pondérée des termes de la question par les passages), quel que
  soit le backend : pas de synonymes au-delà d'un petit thésaurus RH, donc les paraphrases éloignées (« travail à
  distance » pour « télétravail ») entraînent une abstention — sûre, mais gênante. **Mesuré** : avec `jina-embeddings-v3`
  elles sont aussi refusées. Rendre cette porte sémantique est la priorité n°1 de [docs/ROADMAP.md](docs/ROADMAP.md).
- Le mode extractif **cite des phrases**, il ne synthétise pas. Un modèle (`LLM_PROVIDER`) améliore la rédaction.
- Seuils d'abstention calibrés sur un petit corpus synthétique : à recalibrer sur vos documents.
- Détection de conflit : par métadonnée (même « famille » de documents, deux versions actives, chiffres différents).
- Pas de solde de congés, pas de jours ouvrés/fériés, pas de notification, pas d'OCR.
- Authentification locale de prototype (pas de SSO, de verrouillage de compte ni de TLS).

## Données conservées

Tout est local sous `data/runtime/` (ignoré par git) : base SQLite (comptes hachés, registre de documents, demandes,
transferts — le texte des questions transférées y est stocké), fichiers importés (`uploads/`), index vectoriel
(`qdrant/`), journaux techniques sans contenu. Les questions/réponses ne sont pas écrites dans les journaux.
Avec `LLM_PROVIDER=groq|portkey` ou `EMBEDDING_BACKEND=jina`, des extraits de documents et les questions sont envoyés
au fournisseur choisi : n'utilisez ces modes qu'avec des données autorisées.
