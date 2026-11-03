"""Interface du service qualité — Escale Atlantique (mission FastIA)."""

import os

import streamlit as st

# Dans docker compose, l'API est joignable par le NOM de son service (cf. .env)
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Escale Atlantique — Avis clients", page_icon="🏨")
st.title("🏨 Sentiment des avis clients")
st.caption("Escale Atlantique · outil du service qualité")

text = st.text_area(
    "Avis client",
    height=160,
    placeholder="Collez ici un avis Booking, Google, TripAdvisor ou un questionnaire de fin de séjour…",
)

if st.button("Analyser", type="primary"):

    st.info("L'interface n'est pas encore branchée à l'API.")
