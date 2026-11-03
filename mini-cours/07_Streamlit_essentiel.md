# 07 — Streamlit : l'essentiel pour une interface qui consomme une API

> **Utilisé dans** : M0-B2 (tâche 4), puis les tableaux de bord des modules 5 et 6.
> **Durée de lecture** : 20 min · **Exercice guidé** : 40 min

## 1. Pourquoi Streamlit ?

L'équipe qualité d'Escale Atlantique ne va pas utiliser Swagger ni curl. Il lui faut une page web simple : une zone de texte, un bouton, un résultat lisible.

**Streamlit** permet d'écrire cette page **en Python uniquement**, sans HTML ni JavaScript. C'est l'outil idéal pour un MVP ou un tableau de bord interne. Ce n'est pas fait pour un site grand public avec des milliers d'utilisateurs.

```bash
streamlit run app.py     # ouvre http://localhost:8501
```

## 2. Le modèle d'exécution : tout le script, à chaque clic

C'est **la** notion à comprendre. À chaque interaction (un clic, une saisie), Streamlit **réexécute tout le script, de haut en bas**.

```python
import streamlit as st

st.title("Démo")
nom = st.text_input("Ton prénom")
if st.button("Dire bonjour"):
    st.write(f"Bonjour {nom} !")
```

- Au chargement : le script tourne, le bouton vaut `False`, rien ne s'affiche sous le bouton.
- Au clic : le script **retourne entièrement**, cette fois le bouton vaut `True` le temps de ce passage, et le message s'affiche.
- Au clic suivant ailleurs : le script retourne, le bouton revaut `False`, et le message disparaît.

Conséquences pratiques :

- une opération coûteuse placée en haut du script est refaite à **chaque** interaction : c'est à éviter ;
- pour garder une valeur d'un passage à l'autre, il faut `st.session_state` (un dictionnaire qui survit aux réexécutions).

Dans notre cas, c'est simple : le modèle est dans l'**API**, pas dans Streamlit. L'interface ne fait qu'un appel HTTP au clic, ce qui est léger.

## 3. Les composants utiles

```python
st.title("Titre")                       # titres et textes
st.caption("Petit texte gris")
st.markdown("**Gras**, *italique*")

texte = st.text_area("Avis client", height=150, max_chars=2000)   # saisie
if st.button("Analyser", type="primary"):                         # bouton principal
    ...

st.success("Message vert")              # messages colorés
st.warning("Message orange")
st.error("Message rouge")
st.info("Message bleu")

with st.spinner("Analyse en cours…"):   # indicateur d'attente
    ...

with st.expander("Voir le détail"):     # bloc repliable
    st.write(...)

st.bar_chart(serie_pandas)              # graphique en barres
st.dataframe(df)                        # tableau interactif
```

Configuration de la page (à placer avant tout autre affichage) :

```python
st.set_page_config(page_title="Escale Atlantique — Avis", page_icon="🏨")
```

## 4. Appeler une API avec httpx

`httpx` est une bibliothèque HTTP moderne (proche de `requests`) :

```python
import httpx

response = httpx.post("http://api:8000/predict", json={"text": "Super séjour"}, timeout=10)
response.status_code     # 200, 422, 500…
response.json()          # le corps de la réponse, en dict Python
```

**Toujours un `timeout`.** Sans lui, si l'API ne répond pas, l'interface attend… indéfiniment, et l'utilisateur voit une page figée.

## 5. Gérer les erreurs : l'utilisateur ne doit jamais voir une trace Python

Un appel réseau peut échouer de plusieurs façons. Chacune mérite un message **compréhensible par l'équipe qualité** :

| Situation | Ce que httpx fait | Message à afficher |
|---|---|---|
| L'API n'est pas démarrée | lève `httpx.ConnectError` | « Le service d'analyse est indisponible » |
| L'API met trop de temps | lève `httpx.TimeoutException` | « Le service met trop de temps à répondre » |
| L'entrée est refusée | renvoie un code `422` | « L'avis doit contenir entre 1 et 2 000 caractères » |
| Erreur côté API | renvoie un code `500` | « Une erreur est survenue, réessayez » |

Le principe :

```python
try:
    response = httpx.post(url, json=payload, timeout=10)
except httpx.ConnectError:
    st.error("...")
    st.stop()          # arrête le script ici : rien d'autre ne s'affiche
# puis vérifier response.status_code avant d'utiliser response.json()
```

Les pannes réseau **lèvent** une exception, les codes HTTP **non** : un 422 ou un 500 est une réponse normale pour httpx. Il faut donc gérer les deux.



## 6. Pièges fréquents

| Symptôme | Cause probable |
|---|---|
| `ConnectError` alors que l'API tourne | `localhost` utilisé depuis le conteneur : il faut `http://api:8000` (cf. `06_DockerCompose_essentiel.md`) |
| La page reste figée sans fin | Pas de `timeout` dans l'appel httpx |
| Une trace Python rouge s'affiche dans la page | Exception non gérée, ou `response.json()` appelé sur une réponse d'erreur |
| `JSONDecodeError` | Réponse non JSON (page d'erreur) : vérifier le code HTTP d'abord |
| Le résultat disparaît au moindre clic | Réexécution du script : normal pour un `st.button`. Pour le garder, `st.session_state` |
| `StreamlitAPIException: set_page_config() can only be called once` | `st.set_page_config` n'est pas la première commande Streamlit du script |
| Le graphique affiche les étoiles dans le désordre | L'ordre des clés du dictionnaire : trier avant de créer la `Series` |
| Les modifications de `app.py` n'apparaissent pas (Docker) | Image non reconstruite : `docker compose up --build` |

## Pour aller plus loin

- Les bases de Streamlit : https://docs.streamlit.io/get-started/fundamentals/main-concepts
- Liste des composants : https://docs.streamlit.io/develop/api-reference
- httpx : https://www.python-httpx.org/quickstart/
