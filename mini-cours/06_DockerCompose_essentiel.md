# 06 — docker compose : l'essentiel pour orchestrer plusieurs services

> **Utilisé dans** : M0-B2 (tâches 1 et 2), puis M3 (bases de données), M5 (MLOps).
> **Prérequis** : `02_Docker_essentiel.md`.
> **Durée de lecture** : 20 min · **Exercice guidé** : 30 min

## 1. Pourquoi compose ?

En B1, un seul conteneur : `docker build` puis `docker run`, c'était simple. En B2, il y en a deux, une API et une interface, qui doivent :

- se **trouver** sur le réseau ;
- démarrer **dans le bon ordre** (l'interface n'a rien à afficher tant que l'API n'a pas chargé son modèle) ;
- partager une **configuration** ;
- conserver des **fichiers** (le modèle téléchargé, les logs) quand on les arrête.

Avec `docker run`, il faudrait taper et retenir une demi-douzaine de longues commandes. **docker compose** décrit toute la stack dans un fichier, `docker-compose.yml`, et la pilote en une commande.

## 2. Anatomie d'un `docker-compose.yml`

```yaml
services:                      # un bloc par conteneur
  api:                         # le NOM du service (il sert aussi d'adresse réseau)
    build: ./api               # construire l'image à partir de ./api/Dockerfile
    env_file: .env             # variables d'environnement lues dans .env
    ports:
      - "8000:8000"            # hôte:conteneur, comme docker run -p
    volumes:
      - ./logs:/app/logs       # dossier de l'hôte monté dans le conteneur
    networks:
      - interne

  ui:
    build: ./ui
    ports:
      - "8501:8501"
    networks:
      - interne
    depends_on:
      api:
        condition: service_healthy   # attendre que l'API soit en bonne santé

networks:
  interne:                     # un réseau privé partagé par les deux services
```

## 3. Le réseau : on s'appelle par son nom

Sur un réseau compose, **chaque service est joignable par son nom**. Depuis le conteneur `ui`, l'API se trouve à `http://api:8000`.

Pourquoi pas `localhost` ? Dans un conteneur, `localhost` désigne **le conteneur lui-même**. Depuis `ui`, `http://localhost:8000` cherche une API… dans le conteneur Streamlit, où il n'y en a pas.

```
Ta machine (navigateur)
   │ localhost:8501                 │ localhost:8000 (Swagger, curl)
   ▼                                ▼
┌──────── réseau compose ─────────────────────────┐
│  ui  ──── http://api:8000 ────▶  api            │
│  (localhost = ui)                (localhost = api)
└─────────────────────────────────────────────────┘
```

| Qui appelle l'API ? | Adresse à utiliser |
|---|---|
| Ton navigateur, curl, Postman (sur ta machine) | `http://localhost:8000` |
| Un autre conteneur de la même stack | `http://api:8000` |

C'est pour cela que l'adresse de l'API est une **variable d'environnement** (`API_URL` dans `.env`) et non une valeur écrite en dur : elle change selon l'endroit d'où on l'appelle.

## 4. Les volumes : ce qui survit au conteneur

Un conteneur est **jetable** : tout ce qu'il écrit disparaît quand on le supprime. Un **volume** relie un dossier de ta machine à un dossier du conteneur :

```yaml
volumes:
  - ./models:/models      # le modèle téléchargé reste sur ta machine
  - ./logs:/app/logs      # les logs aussi
```

Sans le volume `./models`, le modèle de 270 Mo serait retéléchargé à chaque `docker compose down` / `up`.

## 5. Le healthcheck : « démarré » ne veut pas dire « prêt »

Un conteneur peut être **démarré** (le processus tourne) sans être **prêt** (le modèle est encore en train de se charger). Le healthcheck dit à Docker comment vérifier que le service répond vraiment :

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"]
  interval: 15s        # vérifier toutes les 15 s
  timeout: 5s          # échec si pas de réponse en 5 s
  retries: 5           # unhealthy après 5 échecs consécutifs
  start_period: 180s   # tolérance au démarrage (téléchargement du modèle)
```

L'état passe de `starting` à `healthy` (ou `unhealthy`), et `depends_on: condition: service_healthy` fait attendre l'interface jusque-là.

Note : ici, le test est lancé **dans** le conteneur `api`, donc `localhost` est bien l'API.

## 6. Les commandes du quotidien

```bash
docker compose up --build        # construire et lancer toute la stack (Ctrl+C pour arrêter)
docker compose up --build -d     # pareil, en arrière-plan
docker compose ps                # état des services (running, healthy…)
docker compose logs api          # logs d'un service
docker compose logs -f ui        # suivre les logs en direct
docker compose exec api pytest -v   # lancer une commande dans un service en cours
docker compose exec api bash     # ouvrir un terminal dans un service
docker compose restart ui        # redémarrer un seul service
docker compose down              # arrêter et supprimer les conteneurs (les volumes ./ restent)
```

À retenir : après une modification du **code**, il faut `--build` pour reconstruire l'image. Un simple `up` réutilise l'ancienne.


## 7. Pièges fréquents

| Symptôme | Cause probable |
|---|---|
| L'interface ne joint pas l'API (`ConnectError`) | `localhost` utilisé au lieu du nom du service `api` |
| `ui` ne démarre jamais | `api` jamais *healthy* : lire `docker compose logs api` |
| `api` passe *unhealthy* au premier lancement | Téléchargement du modèle plus long que la tolérance : patienter, augmenter `start_period` |
| `env file .env not found` | Oubli de `cp .env.example .env` |
| Les modifications du code n'apparaissent pas | `docker compose up` sans `--build` |
| `port is already allocated` | L'API de B1 tourne encore sur le port 8000 : l'arrêter |
| Le dossier `logs/` ou `models/` appartient à root (Linux) | Créé par Docker : `sudo chown -R $USER logs models` |
| `yaml: line X: did not find expected key` | Indentation incorrecte : YAML utilise des espaces, jamais de tabulations |

## Pour aller plus loin

- Référence du fichier compose : https://docs.docker.com/reference/compose-file/
- Le réseau dans compose : https://docs.docker.com/compose/how-tos/networking/
