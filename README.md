# Escale Atlantique — Sentiment des avis clients

> Squelette de départ du brief **M0-B2** (FastIA). Ce README est à **réécrire** pendant le brief (tâche 8).

## Le contexte

Le service qualité d'Escale Atlantique reçoit plusieurs centaines d'avis clients par mois et ne parvient plus à tous les lire. Ce projet classe chaque avis en **négatif, neutre ou positif**, pour traiter les clients mécontents en priorité.

Le modèle fourni, [`cmarkea/distilcamembert-base-sentiment`](https://huggingface.co/cmarkea/distilcamembert-base-sentiment), prédit une note de **1 à 5 étoiles** : la conversion en 3 classes est à faire.

## Structure

```
docker-compose.yml     # 2 services (api, ui), réseau interne, volumes, healthcheck
.env.example           # variables d'environnement, à copier en .env
api/                   # service FastAPI
  app/main.py          #   lifespan (chargement du modèle), /health, /info, /predict à compléter
  app/inference.py     #   chargement du pipeline + predict_sentiment à compléter
  app/schemas.py       #   schémas Pydantic (entrée à contraindre)
  tests/               #   2 tests fournis
  Dockerfile
ui/                    # service Streamlit
  app.py               #   interface, pas encore branchée à l'API
  Dockerfile
data/
  avis_annotes.csv     # 30 avis annotés par l'équipe qualité
models/                # cache du modèle Hugging Face (volume, non versionné)
logs/                  # logs de l'API (volume, non versionné, créé au lancement)
```

## Démarrage rapide

```bash
cp .env.example .env
docker compose up --build
```

⏳ Le premier lancement télécharge le modèle (≈ 270 Mo) : compte quelques minutes. Les lancements suivants sont rapides, et tout fonctionne hors ligne.

- API : http://localhost:8000/health et http://localhost:8000/docs
- Interface : http://localhost:8501
- État des services : `docker compose ps` (l'API doit être *healthy*)
- Tests : `docker compose exec api pytest -v`
- Arrêt : `docker compose down`

## Les avis annotés

`data/avis_annotes.csv` contient 30 avis clients annotés à la main par l'équipe qualité. Ils sont censés être anonymisés… vérifie quand même.

| Colonne | Contenu |
|---|---|
| `id` | identifiant de l'avis |
| `hotel` | `City Hotel` ou `Resort Hotel` |
| `source` | Booking, Google, TripAdvisor ou Questionnaire (fin de séjour) |
| `texte` | l'avis, tel que rédigé par le client |
| `sentiment_attendu` | l'annotation de l'équipe qualité |

Consigne d'annotation donnée à l'équipe qualité :

- **négatif** : l'insatisfaction domine, ou le client signale un incident qui appelle une réponse ;
- **neutre** : avis factuel, ou mitigé sans que l'un des deux côtés domine ;
- **positif** : la satisfaction domine, même avec un petit bémol.
