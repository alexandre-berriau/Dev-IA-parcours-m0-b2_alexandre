"""Interface du service qualité — Escale Atlantique (mission FastIA)."""

import io
import os

import httpx
import pandas as pd
import streamlit as st

# Dans docker compose, l'API est joignable par le NOM de son service (cf. .env)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Ordre figé : il sert de lignes et de colonnes à la matrice de confusion
CLASSES = ("négatif", "neutre", "positif")

# Couleur et icône associées à chaque sentiment renvoyé par l'API
SENTIMENT_STYLE = {
    "négatif": ("🔴", "error"),
    "neutre": ("🟠", "warning"),
    "positif": ("🟢", "success"),
}

COLONNE_TEXTE = "texte"
COLONNE_ATTENDU = "sentiment_attendu"

# Même limite que la contrainte Pydantic de l'API : autant la signaler avant l'appel
LONGUEUR_MAX = 2000


@st.cache_data(ttl=300, show_spinner=False)
def _info() -> dict:
    """Modèle, seuils par défaut et taille de lot acceptée, lus sur l'API."""
    reponse = httpx.get(f"{API_URL}/info", timeout=10)
    reponse.raise_for_status()
    return reponse.json()


def reglages() -> dict:
    """Les réglages de l'API, ou un message clair si le service ne répond pas."""
    try:
        return _info()
    except httpx.HTTPError:
        st.error("Le service d'analyse est indisponible.")
        st.stop()


@st.cache_data(show_spinner="Analyse en cours…")
def analyser_avis(texte: str, seuil_negatif: float, seuil_positif: float) -> tuple[int, dict]:
    """Analyse un avis. Le cache rend gratuit un réaffichage à seuils inchangés."""
    charge = {"text": texte, "seuil_negatif": seuil_negatif, "seuil_positif": seuil_positif}
    # 180 s : le premier appel peut tomber pendant le chargement du modèle
    reponse = httpx.post(f"{API_URL}/predict", json=charge, timeout=180)
    return reponse.status_code, reponse.json()


@st.cache_data(show_spinner=False)
def lire_csv(contenu: bytes) -> pd.DataFrame:
    """Charge le fichier déposé ; mis en cache sur son contenu, pas sur son nom."""
    return pd.read_csv(io.BytesIO(contenu))


@st.cache_data(show_spinner="Analyse du fichier…")
def analyser_lot(avis: tuple, seuil_negatif: float, seuil_positif: float) -> dict:
    """Envoie le lot à l'API. Le cache évite de réanalyser à chaque clic de l'interface."""
    charge = {
        "seuil_negatif": seuil_negatif,
        "seuil_positif": seuil_positif,
        "avis": [{"texte": texte, "sentiment_attendu": attendu} for texte, attendu in avis],
    }
    # 180 s : le premier appel peut tomber pendant le chargement du modèle
    reponse = httpx.post(f"{API_URL}/predict/batch", json=charge, timeout=180)
    reponse.raise_for_status()
    return reponse.json()


st.set_page_config(page_title="Escale Atlantique — Avis clients", page_icon="🏨", layout="wide")
st.title("🏨 Sentiment des avis clients")
st.caption("Escale Atlantique · outil du service qualité")

info = reglages()
defauts = info["seuils_defaut"]

with st.sidebar:
    st.header("Seuils de classification")
    seuil_negatif = st.slider(
        "Négatif en dessous de", 1.0, 5.0, float(defauts["seuil_negatif"]), 0.05
    )
    seuil_positif = st.slider(
        "Positif à partir de", 1.0, 5.0, float(defauts["seuil_positif"]), 0.05
    )
    # Même règle que le validateur de l'API : on l'applique avant l'appel pour éviter un 422
    if seuil_negatif >= seuil_positif:
        st.error("Le seuil négatif doit rester strictement sous le seuil positif.")
        st.stop()
    st.caption(f"Modèle : `{info['model_name']}`")

onglet_unitaire, onglet_fichier = st.tabs(["Avis unitaire", "Fichier d'avis"])


with onglet_unitaire:
    text = st.text_area(
        "Avis client",
        height=160,
        placeholder="Collez ici un avis Booking, Google, TripAdvisor ou un questionnaire de fin de séjour…",
    )

    if st.button("Analyser", type="primary"):
        st.session_state["avis_analyse"] = text

    # Gardé en session : déplacer un curseur reclasse l'avis au lieu d'effacer le résultat
    if st.session_state.get("avis_analyse"):
        try:
            code, resultat = analyser_avis(
                st.session_state["avis_analyse"], seuil_negatif, seuil_positif
            )
        except httpx.HTTPError:
            st.error("Le service d'analyse est indisponible ou trop lent.")
            st.stop()

        if code == 422:
            st.warning(f"L'avis doit contenir entre 1 et {LONGUEUR_MAX} caractères.")
        elif code != 200:
            st.error("Une erreur est survenue, réessayez.")
        else:
            sentiment = resultat["sentiment"]
            icone, niveau = SENTIMENT_STYLE[sentiment]
            getattr(st, niveau)(f"{icone} Sentiment : **{sentiment}**")
            st.metric("Note du modèle", f"{resultat['note']:.2f} / 5")

            with st.expander("Voir le détail par nombre d'étoiles"):
                scores = dict(sorted(resultat["scores_etoiles"].items()))
                st.bar_chart(scores)
                st.caption(f"Temps de réponse de l'API : {resultat['latence_ms']:.0f} ms")


with onglet_fichier:
    st.markdown(
        f"Fichier CSV avec une colonne **`{COLONNE_TEXTE}`**. "
        f"Ajoutez une colonne **`{COLONNE_ATTENDU}`** "
        f"({', '.join(CLASSES)}) pour obtenir la matrice de confusion."
    )
    fichier = st.file_uploader("Fichier d'avis", type="csv", label_visibility="collapsed")

    if fichier is not None:
        try:
            donnees = lire_csv(fichier.getvalue())
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
            st.error("Fichier illisible : attendu un CSV encodé en UTF-8.")
            st.stop()

        if COLONNE_TEXTE not in donnees.columns:
            st.error(
                f"Colonne `{COLONNE_TEXTE}` absente. "
                f"Colonnes trouvées : {', '.join(donnees.columns)}."
            )
            st.stop()

        # Les lignes sans texte ne sont pas une erreur de l'utilisateur : on les écarte
        donnees = donnees[donnees[COLONNE_TEXTE].notna()].copy()
        donnees[COLONNE_TEXTE] = donnees[COLONNE_TEXTE].astype(str).str.strip()
        donnees = donnees[donnees[COLONNE_TEXTE] != ""]

        if donnees.empty:
            st.warning("Aucun avis à analyser dans ce fichier.")
            st.stop()

        trop_longues = donnees[donnees[COLONNE_TEXTE].str.len() > LONGUEUR_MAX]
        if not trop_longues.empty:
            # +2 : l'en-tête compte pour une ligne, et les index pandas partent de 0
            lignes = ", ".join(str(index + 2) for index in trop_longues.index)
            st.error(
                f"{len(trop_longues)} avis dépassent {LONGUEUR_MAX} caractères (lignes {lignes})."
            )
            st.stop()

        if len(donnees) > info["lot_max"]:
            st.error(
                f"{len(donnees)} avis pour un maximum de {info['lot_max']} : découpez le fichier."
            )
            st.stop()

        annote = COLONNE_ATTENDU in donnees.columns
        if annote:
            inconnues = set(donnees[COLONNE_ATTENDU].dropna()) - set(CLASSES)
            if inconnues:
                st.error(
                    f"Classes attendues inconnues : {', '.join(sorted(inconnues))}. "
                    f"Valeurs acceptées : {', '.join(CLASSES)}."
                )
                st.stop()

        attendus = donnees[COLONNE_ATTENDU] if annote else pd.Series([None] * len(donnees))
        # Tuple de tuples : le cache de analyser_lot a besoin d'un argument hachable
        avis = tuple(
            (texte, attendu if pd.notna(attendu) else None)
            for texte, attendu in zip(donnees[COLONNE_TEXTE], attendus)
        )

        try:
            reponse = analyser_lot(avis, seuil_negatif, seuil_positif)
        except httpx.HTTPStatusError:
            st.error("L'API a refusé le lot. Vérifiez le fichier et les seuils.")
            st.stop()
        except httpx.HTTPError:
            st.error("Le service d'analyse est indisponible ou trop lent.")
            st.stop()

        resultats = pd.DataFrame(reponse["resultats"])
        resultats.insert(0, COLONNE_TEXTE, list(donnees[COLONNE_TEXTE]))

        gauche, droite = st.columns(2)
        with gauche:
            st.subheader("Répartition")
            st.bar_chart(resultats["sentiment"].value_counts().reindex(CLASSES, fill_value=0))
            st.caption(
                f"{reponse['total']} avis en {reponse['duree_ms']:.0f} ms · "
                f"seuils {reponse['seuils']['seuil_negatif']} / "
                f"{reponse['seuils']['seuil_positif']}"
            )

        with droite:
            st.subheader("Matrice de confusion")
            if not annote:
                st.info(
                    f"Ajoutez une colonne `{COLONNE_ATTENDU}` au fichier "
                    "pour comparer les prédictions aux classes attendues."
                )
            else:
                matrice = (
                    pd.crosstab(resultats[COLONNE_ATTENDU], resultats["sentiment"])
                    .reindex(index=CLASSES, columns=CLASSES, fill_value=0)
                    .rename_axis(index="attendu", columns="prédit")
                )
                st.dataframe(matrice, use_container_width=True)
                # La diagonale, ce sont les avis bien classés
                justes = sum(matrice.loc[classe, classe] for classe in CLASSES)
                total = int(matrice.to_numpy().sum())
                st.metric("Exactitude", f"{justes}/{total}", f"{justes / total:.0%}")

        if annote:
            erreurs = resultats[
                resultats[COLONNE_ATTENDU].notna()
                & (resultats["sentiment"] != resultats[COLONNE_ATTENDU])
            ]
            st.subheader(f"Avis mal classés ({len(erreurs)})")
            if erreurs.empty:
                st.success("Aucune erreur sur ce fichier avec ces seuils.")
            else:
                # La note est la colonne utile : elle dit de combien le seuil a manqué
                st.dataframe(erreurs, use_container_width=True, hide_index=True)

        st.download_button(
            "Télécharger les résultats (CSV)",
            resultats.to_csv(index=False).encode("utf-8"),
            file_name="resultats.csv",
            mime="text/csv",
        )
