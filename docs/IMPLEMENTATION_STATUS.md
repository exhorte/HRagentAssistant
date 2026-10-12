# État d'implémentation

_Dernière mise à jour : 2026-10-12 · branche `feat/mvp-hr-assistant`_

## Objectif actuel

MVP local fonctionnel : assistant documentaire RH sourcé, workflow de congé, transfert aux RH.
**Statut : MVP livré, avec les limites listées ci-dessous.** Rien n'est présenté comme prêt pour la production.

## Terminé

- Audit ciblé et étude des 5 dépôts de référence (`PROJECT_AUDIT.md`, `REFERENCE_ANALYSIS.md`).
- Socle : configuration par environnement (défauts hors ligne), dépendances unifiées dans `pyproject.toml`, journaux sans contenu.
- RAG : import TXT/PDF/DOCX, découpage par sections, métadonnées, index Qdrant avec filtre d'accès, citations vérifiées,
  abstention, conflits de versions, neutralisation des injections, mode LLM optionnel avec vérification.
- Workflows : demande de congé (machine à états, historique), transfert RH (aucune notification envoyée).
- Sécurité : authentification locale, rôles, contrôle côté backend (voir `SECURITY.md`).
- Interface Streamlit (connexion, assistant, documents, demandes) ; CLI (`init`, `ask`, `documents`, `eval`, `serve`).
- Corpus de démonstration synthétique (6 documents + 3 documents « spéciaux »), jeux d'évaluation (36 + 15 cas).
- Documentation : README, ARCHITECTURE, SECURITY, ROADMAP.

## Tests exécutés (2026-10-12, Windows 11, Python 3.12.13)

| Suite | Résultat |
|---|---|
| `pytest` (unitaires, intégration, sécurité, UI headless) | **211 passés**, 0 échec, ~75 s, hors ligne |
| dont `-m security` | tests de sécurité bloquants : tous passés ; contrôle par mutation (filtre de rôle retiré → tests en échec) |
| UI Streamlit | `AppTest` (7 scénarios) + vérification manuelle dans un navigateur (connexion, chat, conflit, documents RH, demandes) |
| Chemin LLM réel (Groq, `openai/gpt-oss-20b`) | 2 questions, `mode=llm`, JSON valide, citations vérifiées — **test manuel ponctuel, non automatisé** |
| Client Jina réel (`jina-embeddings-v3`) | embeddings obtenus (dim 1024) — **test manuel ponctuel, non automatisé** |
| Chemin Portkey | **non testé** (pas de clé `PORTKEY_API_KEY`) |
| Qdrant Cloud | **non testé** (pas de `QDRANT_URL`) |

## Évaluation métier (`python main.py eval`) — à lire avec prudence

| Jeu | Backend | Réussite | Recall@k | Abstentions correctes | Fausses abstentions |
|---|---|---:|---:|---:|---:|
| `hr_eval_cases.json` (36 cas) | hashing (hors ligne) | 36/36 | 1,0 | 5/5 | 0 |
| `holdout_cases.json` (15 cas) | hashing | 15/15 | 1,0 | 5/5 | 0 |
| `hr_eval_cases.json` | jina-v3 | 36/36 | 1,0 | 5/5 | 0 |
| `holdout_cases.json` | jina-v3 | 15/15 | 1,0 | 5/5 | 0 |

**Ces 100 % ne sont pas une mesure de qualité.** Les cas sont écrits par le développeur sur un corpus synthétique
de 9 petits documents, après calibrage sur des questions voisines. Historique honnête :

1. Première exécution du jeu de 36 cas : 34/36 (extraction trop restrictive sur les questions multi-éléments ;
   question « Combien de jours ? » répondue). Corrigé.
2. Sonde de 16 paraphrases non vues : 9/16 (6 abstentions à tort, 1 fausse réponse). Normalisation lexicale ajustée.
3. Jeu `holdout_cases.json` écrit *avant* ce réglage, exécuté une fois : 13/15, dont **2 fausses réponses**
   (« préavis », « démission » : le terme directeur est absent des documents mais des mots génériques franchissaient le
   seuil). Règle ajoutée (le terme le plus spécifique de la question doit être retrouvé). Le jeu n'est plus vierge ensuite.
4. Mesure avec Jina : les 4 paraphrases testées (« travail à distance », « garder mes congés pour l'an prochain »,
   « souffrant », « justificatif de salaire ») sont **toujours refusées** : la porte d'abstention est lexicale.

## En cours / à faire

- Porte d'abstention sémantique (ROADMAP n°1) ; recalibrage des seuils sur de vrais documents autorisés.
- Évaluation par un juge LLM (`run_langsmith_evaluation`, extra `eval`) : conservée mais **non exécutée** ici.
- Sécurité de niveau production (SECURITY.md, « Avant toute utilisation réelle »).

## Problèmes connus

- Paraphrases éloignées refusées (voir ci-dessus) — comportement sûr mais gênant.
- Mode extractif : phrases citées, pas de synthèse ; l'ordre des puces suit le classement des passages.
- Un seul processus peut ouvrir l'index Qdrant local (verrou) : arrêter l'UI avant d'utiliser la CLI.
- Les dates sont en jours calendaires ; pas de jours ouvrés/fériés ; pas de solde.
- `rag.py` (fichier vide d'origine) et `rag.ipynb` (prototype pédagogique FAISS) sont conservés tels quels.

## Décisions techniques

Voir `ARCHITECTURE.md` (tableau des décisions). Notamment : suppression de l'agent LangChain au profit d'une récupération
systématique, Qdrant embarqué, SQLite, client Jina maison (`langchain-community` en fin de vie), routage d'intention par règles.

## Prochain objectif concret

Calibrer et rendre sémantique la décision d'abstention avec `jina-embeddings-v3` sur un jeu annoté de questions réelles
paraphrasées, sans dégrader le taux d'abstention correcte.
