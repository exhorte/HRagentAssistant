# Feuille de route

Ordre proposé après le MVP. Chaque étape doit être justifiée par une mesure sur le jeu d'évaluation ou par un besoin
métier validé, pas par l'attrait de la technique.

## 1. Rendre l'abstention sémantique (priorité n°1, mesurée comme limite)

La porte d'abstention est lexicale : avec `jina-embeddings-v3`, « travail à distance » ou « garder mes congés pour l'an
prochain » restent refusés. Piste : décider sur le score sémantique du meilleur passage (seuil calibré par backend sur
vos documents) et garder la couverture lexicale comme signal secondaire ; ajouter une recherche hybride (BM25 + vecteurs,
fusion RRF) **si** la mesure le justifie ; reranking seulement si l'évaluation le démontre.

## 2. Évaluation sur des documents réels autorisés

Pilote avec une PME partenaire, documents utilisés avec son accord : construire un jeu d'évaluation annoté par les RH
(≥ 100 questions réelles), mesurer Recall@k, taux d'abstention correcte/fausse, citations valides ; recalibrer
`MIN_COVERAGE`, `MIN_SCORE`, `CHUNK_SIZE`.

## 3. Sécurité de niveau production

Voir la liste « Avant toute utilisation réelle » de [SECURITY.md](SECURITY.md) : SSO, chiffrement, audit, RGPD.

## 4. Intégrations métier

- Solde de congés et calendrier des jours ouvrés/fériés depuis un SIRH (aujourd'hui : « non disponible »).
- Notifications (e-mail/Teams) pour les demandes et transferts — aujourd'hui **aucune n'est envoyée**.
- Autres processus (attestations, notes de frais) en réutilisant le patron « machine à états + historique ».

## 5. Plateforme

- Multi-entreprise réel (tenants administrables, index/clés séparés).
- Multilingue (détection par document, modèles d'embeddings multilingues, réponses dans la langue de la question).
- Interface web dédiée et API si plusieurs clients que Streamlit apparaissent.

## 6. À n'envisager que si le besoin est démontré

GraphRAG (relations documents ↔ données métier), agents spécialisés, OCR des PDF scannés.
