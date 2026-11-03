# 05 — Hugging Face : l'essentiel pour utiliser un modèle pré-entraîné

> **Utilisé dans** : M0-B2 (tâches 2 et 3), M1-B2 (embeddings), puis tout le volet NLP du parcours.
> **Durée de lecture** : 25 min · **Exercice guidé** : 45 min

## 1. Hugging Face, c'est quoi ?

[huggingface.co](https://huggingface.co) est la principale plateforme de partage de modèles d'IA : des centaines de milliers de modèles (texte, image, audio), prêts à télécharger et à utiliser **en local**.

Deux briques nous intéressent :

- **le Hub** : le catalogue en ligne, où chaque modèle a une page et une **fiche** (*model card*) ;
- **la bibliothèque `transformers`** : le code Python qui télécharge un modèle et l'exécute sur ta machine.

En B1, le modèle était un fichier `.joblib` qu'on t'avait livré. Ici, il vient du Hub : c'est la situation la plus courante en NLP, car entraîner un modèle de langue from scratch coûte des semaines de calcul.

## 2. Lire une fiche de modèle : un réflexe professionnel

Avant d'intégrer un modèle, **on lit sa fiche**. C'est l'équivalent de la notice d'un médicament : à quoi il sert, sur quoi il a été testé, ses limites.

| Question | Où chercher | Pourquoi c'est important |
|---|---|---|
| Que fait le modèle ? | description, onglet *Use this model* | vérifier qu'il répond au besoin |
| Sur quelles données a-t-il été entraîné ? | section *Dataset* ou *Training data* | un modèle entraîné sur des films ou des produits ne connaît pas forcément le vocabulaire de l'hôtellerie |
| Quels labels renvoie-t-il ? | exemples de code, fichier `config.json` (`id2label`) | le format de sortie à convertir |
| Quelles performances ? | section *Evaluation* | sur quelles données ont-elles été mesurées ? |
| Quelles limites et quels biais ? | section *Limitations*, *Bias* | ce qui peut mal tourner en production |
| Quelle licence ? | en haut de la page | a-t-on le droit de l'utiliser dans un projet client ? |
| Quelle taille ? | onglet *Files and versions* | temps de téléchargement, mémoire, coût |

## 3. Le `pipeline` : un modèle en 2 lignes

`pipeline` regroupe tout ce qu'il faut pour utiliser un modèle : le **tokenizer** (qui découpe le texte en morceaux numérotés), le **modèle** et le post-traitement.

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="nom-du-modele")
classifier("Ce film est génial !")
```

Au premier appel, le modèle est **téléchargé** puis mis en **cache** sur le disque. Les appels suivants sont hors ligne.

### Obtenir tous les scores, pas seulement le meilleur

Par défaut, le pipeline renvoie uniquement la classe la plus probable. Avec `top_k=None`, il renvoie **le score de chaque classe**, ce qui est indispensable pour adapter la sortie :

```python
classifier = pipeline("text-classification", model="nom-du-modele", top_k=None)
classifier("Ce film est génial !")
```

La structure exacte du résultat (une liste de dictionnaires, ou une liste qui contient une liste de dictionnaires) varie selon la forme de l'entrée (un texte ou une liste de textes) et selon la version de `transformers`. **Affiche-la toujours avant d'écrire du code qui la parcourt.**

### Le cache

Par défaut, les modèles sont stockés dans `~/.cache/huggingface`. La variable d'environnement `HF_HOME` permet de changer cet emplacement : c'est ce que fait le `docker-compose.yml` du squelette, pour que le modèle soit stocké dans le volume `./models` et survive à la suppression du conteneur.

## 4. Les tokens et la longueur maximale

Un modèle de langue ne lit pas des caractères mais des **tokens** : des mots ou morceaux de mots. « Inadmissible » peut donner plusieurs tokens. En français, compte en moyenne un token pour 3 à 4 caractères.

Chaque modèle a une **longueur maximale** de tokens qu'il peut traiter d'un coup. Que se passe-t-il si un avis la dépasse ? Lis la fiche du modèle, puis **teste-le** avec un texte très long. Le pipeline accepte des options pour gérer ce cas : cherche `truncation` dans la documentation.

C'est un piège classique de mise en production : tout fonctionne avec les exemples courts du développement, puis un client écrit un roman et le service tombe.

## 5. Adapter la sortie d'un modèle au besoin métier

Un modèle répond à **sa** question, rarement exactement à celle du métier. Ici, le modèle prédit une note de 1 à 5 étoiles, alors que l'équipe qualité veut trois classes.

La conversion n'est pas un détail technique : c'est une **décision métier**, qui doit être explicite, justifiée et testable. Quelques familles de solutions :

| Approche | Principe | À se demander |
|---|---|---|
| **Regroupement** | prendre l'étoile la plus probable, puis regrouper les étoiles en classes | où placer les 2 et 3 étoiles ? |
| **Note moyenne** | calculer Σ (étoile × probabilité), puis découper avec des seuils | comment choisir et expliquer les seuils ? |
| **Seuil de confiance** | utiliser la probabilité cumulée d'un groupe d'étoiles | que faire quand le modèle hésite ? |

Pour choisir, pose la question du **coût des erreurs** : qu'est-ce qui coûte le plus cher à l'équipe qualité, un client mécontent qu'on rate ou un avis neutre qu'on relit pour rien ? Les erreurs n'ont presque jamais le même coût.

Bonne pratique : isoler la conversion dans **une fonction à part**, sans dépendance au modèle. Tu peux alors la tester avec des scores inventés, en une fraction de seconde.

## 6. Exercice guidé — explorer le modèle avant de l'intégrer

Fais cet exercice **dans le conteneur** (les bibliothèques et le modèle y sont déjà) :

```bash
docker compose exec api python
```

**Étape 1 — Charger et observer.**

```python
from transformers import pipeline
clf = pipeline("text-classification", model="cmarkea/distilcamembert-base-sentiment", top_k=None)
resultat = clf("Chambre propre et personnel adorable.")
print(resultat)
```

✅ *Tu dois pouvoir dire : quelle est la structure du résultat ? Quels sont les 5 labels exacts ? Les scores somment-ils à 1 ?*

**Étape 2 — Tester des cas difficiles.** Essaie quelques avis de `data/avis_annotes.csv`, en particulier les n°3, 4, 22 et 26. Note ce que tu observes : ce sont tes premières hypothèses pour l'analyse des erreurs.

**Étape 3 — Tester les limites.**

```python
clf("Très bien. " * 1000)
```

✅ *Que se passe-t-il ? Trouve l'option qui règle le problème, et vérifie qu'elle fonctionne.*

**Étape 4 — Écrire la conversion.** Dans `inference.py`, écris une fonction qui reçoit les scores des 5 étoiles et renvoie `"négatif"`, `"neutre"` ou `"positif"`. Teste-la avec des scores inventés, par exemple `{1: 0.2, 2: 0.25, 3: 0.05, 4: 0.3, 5: 0.2}` : quelle classe renvoie-t-elle, et est-ce ce que tu veux ?

**Étape 5 — Brancher `predict_sentiment`** et la route `/predict`.

✅ *Dans Swagger, `/predict` ne renvoie jamais une étoile brute, toujours une des 3 classes, avec les scores des 5 étoiles.*

## 7. Pièges fréquents

| Symptôme | Cause probable |
|---|---|
| Le premier démarrage est très long | Téléchargement du modèle : c'est normal, il est ensuite en cache |
| Le modèle est retéléchargé à chaque redémarrage | Le cache n'est pas dans un volume persistant (`HF_HOME`) |
| `TypeError: list indices must be integers or slices, not str` | La sortie est une liste de listes : afficher le résultat et en extraire le bon niveau |
| `KeyError: '1 star'` ou une conversion qui ne trouve rien | Labels mal recopiés : vérifier la sortie réelle du modèle ou `clf.model.config.id2label` |
| `/predict` renvoie 500 sur un avis long | Longueur maximale du modèle dépassée : option de troncature absente |
| `ValueError: Couldn't instantiate the backend tokenizer` | Bibliothèque `sentencepiece` absente |
| `ImportError: ... requires the protobuf library` au chargement | Bibliothèque `protobuf` absente : le tokenizer de CamemBERT en a besoin pour être converti |
| `OSError: We couldn't connect to 'https://huggingface.co'` | Pas de connexion au premier lancement, ou nom du modèle mal orthographié |
| L'API est lente au premier appel, rapide ensuite | Initialisation paresseuse de PyTorch : normal |
| Image Docker de plusieurs Go | PyTorch installé avec le support GPU : utiliser la version CPU |

## Pour aller plus loin

- Les pipelines de `transformers` : https://huggingface.co/docs/transformers/main_classes/pipelines
- Le cours NLP gratuit de Hugging Face (chapitres 1 et 2) : https://huggingface.co/learn/nlp-course
- Ce qu'une bonne fiche de modèle doit contenir : https://huggingface.co/docs/hub/model-cards
