# Escale Atlantique — Sentiment des avis clients

Le service qualité reçoit plusieurs centaines d'avis par mois et n'arrive plus à les lire. Ce projet les trie en négatif, neutre ou positif pour faire remonter les mécontents en premier.

Le modèle imposé, [`cmarkea/distilcamembert-base-sentiment`](https://huggingface.co/cmarkea/distilcamembert-base-sentiment), sort une note de 1 à 5 étoiles. Il faut en faire trois classes, c'est tout le sujet du brief.

## Architecture

```
                   ┌──────────────────────────────┐
  navigateur ─────▶│  ui — Streamlit : 8501       │
                   │  curseurs de seuils          │
                   │  onglet avis / onglet fichier│
                   └──────────────┬───────────────┘
                                  │ HTTP, réseau interne « fastia »
                                  │ API_URL=http://api:8000
                   ┌──────────────▼───────────────┐
                   │  api — FastAPI : 8000        │
                   │  /health /info               │
                   │  /predict /predict/batch     │
                   │  /logs                       │
                   │  pipeline transformers       │
                   └───┬──────────────────────┬───┘
                       │                      │
               ┌───────▼───────┐     ┌────────▼────────┐
               │ ./models      │     │ ./logs          │
               │ cache HF      │     │ api.log         │
               │ 261 Mo        │     │ 1 ligne / appel │
               └───────────────┘     └─────────────────┘
```

Le modèle est chargé une fois au démarrage dans le `lifespan` de FastAPI, pas à chaque requête. Le healthcheck laisse 180 s au premier lancement pour le téléchargement, et l'interface attend que l'api soit `healthy` avant de démarrer.

Les deux volumes ne sont pas versionnés. Le cache se retélécharge tout seul, les logs se recréent.

## Installation et lancement

```bash
cp .env.example .env
docker compose up --build
```

Le premier lancement télécharge le modèle, environ 270 Mo. Après, ça marche hors ligne.

| | |
|---|---|
| Interface | http://localhost:8501 |
| API | http://localhost:8000/docs |
| État des services | `docker compose ps`, l'api doit être `healthy` |
| Tests | `docker compose exec api pytest -v`, 56 tests |
| Arrêt | `docker compose down` |

La première analyse prend 1 764 ms, les suivantes 40 ms. Le modèle est déjà chargé à ce moment-là, ce premier appel paie juste l'initialisation de torch.

## De 5 étoiles à 3 classes

### La note plutôt que l'étoile la plus probable

Le pipeline renvoie cinq probabilités. Le plus simple serait de prendre la plus forte et de regrouper 1-2 / 3 / 4-5, mais ça jette de l'information. Sur un avis à 3★ 0.44 et 4★ 0.46, cette méthode tranche sur 2 centièmes d'écart et ne laisse rien à régler derrière.

J'utilise la note attendue, soit la moyenne des cinq étoiles pondérée par leurs probabilités :

```
note = 1×p(1★) + 2×p(2★) + 3×p(3★) + 4×p(4★) + 5×p(5★)
```

Elle garde toute la distribution et donne une valeur continue, sur laquelle je peux poser des seuils.

### Deux seuils, pas symétriques

```
1 ────────── 2.8 ────────── 3.6 ────────── 5
   négatif        neutre        positif
```

Centrés sur 3, les seuils vaudraient 2.5 et 3.5. Les miens sont décalés vers le haut pour une raison métier : un client mécontent classé neutre ne reçoit pas de réponse, alors qu'un avis neutre classé négatif coûte deux minutes de lecture à l'équipe. Les deux erreurs n'ont pas le même prix, donc les seuils ne sont pas symétriques.

Ils ne sont pas figés dans le code. Ils passent dans la requête, et les curseurs de l'interface les déplacent sans redéployer. `/info` publie les valeurs par défaut pour qu'elles ne soient écrites qu'à un endroit.

## Ce que ça donne sur les 30 avis

Matrice de confusion avec les seuils 2.8 / 3.6 :

| attendu ↓ / prédit → | négatif | neutre | positif |
|---|---|---|---|
| négatif | 10 | 0 | 2 |
| neutre | 0 | 7 | 1 |
| positif | 1 | 0 | 9 |

26 avis sur 30, soit 87 %.

### Comparaison de trois conversions

| conversion | exactitude | négatifs ratés |
|---|---|---|
| argmax regroupé 1-2 / 3 / 4-5 | 25/30 — 83 % | 2 |
| note attendue, seuils 2.5 / 3.5 | 25/30 — 83 % | 2 |
| note attendue, seuils 2.8 / 3.6 | 26/30 — 87 % | 2 |

Le brief demande laquelle rate le moins d'avis négatifs. Aucune : les trois en ratent deux, et ce sont les deux mêmes, les avis 3 et 18. Le critère ne départage pas, parce que ces erreurs ne viennent pas de la conversion mais de la compréhension du texte. Aucun seuil ne les rattrape.

La différence se joue sur la frontière neutre / positif. L'avis 30, « Bien pour les familles, un peu bruyant pour les couples », vaut 3.55 et bascule en positif dès que le seuil haut descend à 3.5. Mes seuils gagnent donc un avis sur trente, mais pas sur le critère mis en avant par le brief.

L'argmax regroupé et la note attendue à seuils symétriques donnent la même matrice et la même liste d'erreurs. Sur ces 30 avis, les deux méthodes sont indiscernables. Ce qui change quelque chose, c'est le décalage des seuils.

### Marge des seuils

En reprenant les notes réelles, voici l'intervalle dans lequel chaque seuil peut bouger sans changer un seul classement :

| seuil | intervalle valide | ma valeur | marge |
|---|---|---|---|
| bas | ] 2.04 ; 2.81 ] | 2.8 | 0.01 |
| haut | ] 3.55 ; 3.91 ] | 3.6 | 0.05 |

Le seuil bas est à un centième de sa borne. C'est volontaire, je préfère pencher vers l'alerte, mais c'est fragile : un avis neutre un peu plus sec que l'avis 13 passerait en négatif. Avec seulement 30 avis, ces bornes sont elles-mêmes approximatives.

## Pourquoi le modèle se trompe

Quatre erreurs, qui se rangent en trois familles.

### La négation d'un mot négatif, avis 17

> « Pas une seule fausse note, on ne pouvait pas rêver mieux. » — attendu positif, prédit négatif (note 2.10)

Mon idée de départ était que le modèle additionne les mots chargés : dans « pas mal », il compterait « pas » et « mal » tous les deux du côté négatif.

Je l'ai testée avec quatre jeux de phrases courtes (`data/test_negation_*.csv`), chacun isolant une construction, avec deux familles témoins de même longueur :

| famille | note moyenne | bien classés |
|---|---|---|
| A. litote positive, « pas mal », « pas mauvais » | 3.36 | 3/7 |
| B. positif direct, témoin | 4.12 | 3/3 |
| C. négatif direct, témoin | 1.54 | 3/3 |
| D. négation d'un mot positif, « pas bien », « rien d'exceptionnel » | 2.07 | 4/4 |

La mesure me contredit. La famille D passe à 4/4 : « bien » et « exceptionnel » sont des mots positifs, et avec la négation ils ressortent à 2.0, donc négatifs. Un modèle qui additionnerait des mots ne ferait pas ça. Et mon exemple « Pas mal le séjour » sort à 3.35, en neutre, pas en négatif.

Le défaut est ailleurs. Nier un positif marche, la famille D à 2.07 est aussi basse que les vrais négatifs à 1.54. Nier un négatif marche mal, la famille A reste à 3.36 alors que les vrais positifs sont à 4.12. La négation atténue la polarité au lieu de la retourner.

Dans le détail, ça ressemble plus à un effet de fréquence qu'à de la grammaire. « Rien à redire » (4.63) et « je ne regrette pas d'être venu » (4.50) passent sans problème, ce sont des tournures courantes dans les avis. « Ce n'était pas mauvais », même construction, échoue à 2.15. Le cas le plus parlant : « pas mal du tout » vaut 3.74 et « pas mal le séjour » vaut 3.35, même litote, 0.4 point d'écart pour la version la plus fréquente. Le modèle a retenu les litotes courantes comme des blocs au lieu de composer la négation.

L'avis 17 se décompose bien :

| | note |
|---|---|
| « Pas une seule fausse note » | 2.33 |
| « On ne pouvait pas rêver mieux » | 2.80 |
| l'avis entier | 2.10 |

Les deux moitiés échouent séparément et l'erreur s'additionne.

À noter quand même, l'avis 4 du CSV, « Rien à redire, je ne m'attendais pas à un accueil aussi chaleureux », sort à 4.60 et il est correct. Même famille grammaticale, résultat opposé, donc on ne peut pas conclure que le modèle ne gère pas la négation.

### L'ironie, avis 3 et 18

> « Bravo pour l'insonorisation : j'ai pu suivre en direct toute la soirée de mes voisins jusqu'à 3h du matin. » — attendu négatif, prédit positif (4.70)

> « Super, la "vue mer" donnait sur le parking. » — attendu négatif, prédit positif (4.26)

Les deux avis sont écrits avec du vocabulaire positif, « bravo », « merci », « super », « heureusement », et c'est le contexte qui inverse le sens. Le modèle lit les mots et note 4.3 à 4.7. Il est très confiant et complètement à côté.

Ce sont les deux négatifs ratés de la matrice, donc les erreurs les plus coûteuses pour le service qualité. Les seuils ne peuvent rien : pour rattraper un 4.70, il faudrait un seuil haut au-dessus de 4.7, ce qui ferait basculer presque tous les vrais positifs en neutre.

### L'avis sans contenu, avis 22

> « RAS. » — attendu neutre, prédit positif (4.44)

Quatre caractères. Pour l'équipe qualité, « rien à signaler » est factuel donc neutre. Le modèle y voit une absence de plainte et note 4.44.

Le problème vient de l'entrée, pas de la conversion : il n'y a pas assez de texte pour justifier une note. En dessous d'un certain nombre de caractères, renvoyer « indéterminé » serait plus correct qu'une prédiction à 4.44 sur un sigle.

### Cas limite, avis 30

> « Bien pour les familles, un peu bruyant pour les couples qui cherchent le calme. » — attendu neutre, prédit neutre (3.55)

Classé correctement, mais à 0.05 du seuil haut. Ce n'est pas une erreur, c'est l'avis qui décide à lui seul entre les deux jeux de seuils. Je le signale parce qu'un réglage qui tient sur un seul exemple est un réglage sur-ajusté.

## Limites connues

30 avis, c'est peu. Les seuils sont réglés sur cet échantillon et les intervalles valides sont étroits, un avis de plus peut déplacer les bornes.

Un modèle plus gros ne corrige pas ces erreurs. J'ai comparé avec `nlptown/bert-base-multilingual-uncased-sentiment`, 12 couches et 640 Mo contre 6 couches et 261 Mo, sur les mêmes 30 avis : 25/30 contre 26/30, avec les mêmes erreurs plus une. Sur l'avis 17 les deux donnent 2.09, au centième près. Ce n'est donc pas un problème de capacité, l'ironie et les litotes rares ne sont simplement pas dans ce qu'il a vu à l'entraînement.

Les phrases de test font 3 à 6 mots alors que le modèle a été entraîné sur des avis entiers. Les témoins ont la même longueur donc la comparaison tient, mais les notes absolues ne veulent pas dire grand-chose.

L'étiquetage de la famille A se discute. J'ai annoté « Pas mal le séjour » en positif ; en neutre, le modèle devient correct dessus et la famille passe à 4/7.

## Journalisation et données personnelles

Une ligne par appel à `/predict` dans `logs/api.log`, avec la classe, la durée et l'avis tronqué à 80 caractères.

La route `/logs` ne renvoie pas le texte, seulement l'horodatage, la classe et la durée. 80 caractères ne suffisent pas à anonymiser : dans l'avis 10 du CSV, le nom apparaît au caractère 22 et le numéro de chambre au caractère 41. L'extrait reste donc côté serveur, et il ne peut pas atteindre le tampon lu par `/logs`.

Un lot n'écrit qu'une ligne. Sinon 200 avis d'un coup videraient le tampon de toutes les prédictions unitaires.
