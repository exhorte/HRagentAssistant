# Sécurité et confidentialité

Niveau visé : **prototype local avec données synthétiques**. Ce document décrit ce qui est implémenté et testé, et ce
qui reste à faire avant toute utilisation avec des données réelles.

## Ce qui est appliqué par le backend (et testé)

| Contrôle | Où | Tests |
|---|---|---|
| Authentification par mot de passe (PBKDF2-HMAC-SHA256, sel, comparaison à temps constant, aucun indice sur l'échec) | `auth.py` | `test_auth_config_deps.py`, `test_workflow_authorization.py` |
| Rôles `employee` / `hr_manager` ; chaque service exige une permission | `auth.require` | `test_workflow_authorization.py` |
| Filtre d'accès (tenant, rôle autorisé, statut actif) appliqué **par Qdrant avant toute génération** | `vector_store.access_filter` | `test_access_control.py` (dont une mutation vérifiée : sans le filtre, les tests échouent) |
| Documents « RH uniquement » invisibles (liste, recherche, réponses, prompts du modèle) pour les employés | `documents`, `vector_store` | `test_access_control.py` |
| Transitions de statut des demandes : table explicite, transition atomique, aucun écrit de statut possible via l'UI/les services | `leave.py` | `test_leave_workflow.py` (exhaustif), `test_workflow_authorization.py` |
| Soumission uniquement avec confirmation explicite ; décision RH uniquement après prise en charge, jamais sur sa propre demande ; refus motivé | `leave.py` | idem |
| Documents = données : passages assainis (phrases d'instruction retirées) avant d'atteindre un modèle | `guardrails.neutralize_passage` | `test_injection_and_leakage.py` |
| Citations vérifiées, chiffres vérifiés, promesses/liens/secrets refusés dans les sorties générées | `answering`, `guardrails` | `test_answering_units.py`, `test_rag_flow.py` |
| Aucun outil exécutant du code ou du SQL fourni par l'utilisateur ; SQL paramétré | `db.py` et services | `test_workflow_authorization.py` |
| Noms de fichiers assainis, taille/signature vérifiées, stockage sous un identifiant interne | `document_loader`, `documents` | `test_document_loader.py`, `test_injection_and_leakage.py` |
| Erreurs : messages génériques côté utilisateur ; détail technique dans les journaux, sans secret | `answering`, `documents`, `app.py` | `test_injection_and_leakage.py` |
| Journaux sans texte de question/réponse ni contenu de document (identifiants, statuts, durées) | `logger.py` | `test_injection_and_leakage.py` |
| Secrets uniquement depuis l'environnement, exclus des `repr` | `config.py` | `test_auth_config_deps.py` |

Les garde-fous heuristiques (injection, données de tiers) sont de la **défense en profondeur** : ils sont contournables
par reformulation. Ce qui protège les données, c'est le filtre d'accès (couche 1), pas ces règles.

## Données conservées

| Donnée | Emplacement | Remarque |
|---|---|---|
| Comptes (mots de passe hachés), registre des documents, demandes de congé, transferts RH | `data/runtime/hr_assistant.sqlite3` | non chiffré ; **le texte des questions transférées y est stocké** |
| Fichiers importés | `data/runtime/uploads/` | non chiffrés |
| Passages indexés + vecteurs | `data/runtime/qdrant/` | non chiffrés |
| Journaux techniques | `data/runtime/logs/` | sans contenu utilisateur |
| Mots de passe de démonstration générés | `data/runtime/demo_credentials.txt` | local, ignoré par git |

## Flux sortants (uniquement si vous les activez)

- `LLM_PROVIDER=groq|portkey` : la question et les passages autorisés partent chez le fournisseur du modèle.
- `EMBEDDING_BACKEND=jina` : le texte des documents et des questions part chez Jina.
- `QDRANT_MODE=cloud` : passages et vecteurs partent vers Qdrant Cloud.
- `LANGSMITH_TRACING=true` : prompts et passages partent vers LangSmith (désactivé par défaut).

## Avant toute utilisation réelle (non implémenté)

1. **Authentification d'entreprise** (SSO/OIDC), verrouillage de compte, politique de mots de passe, sessions expirantes, TLS.
2. **Chiffrement au repos** des fichiers, de la base et de l'index ; gestion des clés.
3. **Journal d'audit** inaltérable des accès aux documents et des décisions RH.
4. **Isolation multi-entreprise** réelle (administration de tenants, clés et index séparés, tests d'isolation de bout en bout).
5. **Rétention et suppression** des données (RGPD : durée de conservation, droit à l'effacement), DPA avec les fournisseurs.
6. **Revue des fournisseurs d'IA** : localisation des données, absence d'entraînement sur vos données.
7. **Limitation de débit**, antivirus sur les fichiers importés, quotas.
8. **Évaluation sur vos documents** et validation métier par les RH avant tout usage décisionnel.
9. Remplacer le stockage des questions transférées par un workflow avec habilitations fines.

Ces points sont des critères de sortie du prototype ; l'application n'est **pas** une plateforme SaaS multi-entreprise.
