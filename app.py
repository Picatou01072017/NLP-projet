import os
import time
import joblib
import spacy
import pandas as pd
import streamlit as st

# -----------------------------
# Configuration & utilitaires
# -----------------------------

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
REQUIRED_COLUMNS = ["review_title", "review_body"] 

@st.cache_resource(show_spinner=True)
def load_model(model_path: str):
    return joblib.load(model_path)

def list_models(models_dir: str):
    if not os.path.exists(models_dir):
        return []
    files = [f for f in os.listdir(models_dir)
             if f.lower().endswith(('.joblib', '.pkl'))]
    return sorted(files)

def predict_with_model(model, title: str, body: str):
    # Construit l'input au format attendu (DataFrame à 2 colonnes)
    X = pd.DataFrame(
        [{"review_title": title.strip(), "review_body": body.strip()}],
        columns=REQUIRED_COLUMNS
    )
    t0 = time.time()
    y_pred = model.predict(X)
    latency = time.time() - t0

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict(X)
        except Exception:
            proba = None

    return y_pred[0], proba, latency

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Avis Client – Prédiction de note", page_icon="⭐", layout="centered")

st.title("⭐ Prédiction de la note d’un avis client")
st.caption("Entrez le **titre** et le **corps** de l’avis. Choisissez un modèle pour obtenir la prédiction.")

with st.sidebar:
    st.header("⚙️ Modèle")
    available = list_models(MODELS_DIR)
    if not available:
        st.warning(f"Aucun modèle trouvé dans `{MODELS_DIR}/`. Ajoute des fichiers `.joblib` ou `.pkl` (pipelines sklearn).")
        st.stop()

    model_name = st.selectbox("Choisir un modèle :", available, index=0)
    model_path = os.path.join(MODELS_DIR, model_name)

    st.write("📁", model_path)

    # Chargement paresseux & mis en cache
    with st.spinner("Chargement du modèle…"):
        model = load_model(model_path)
    st.success("Modèle chargé.")

# Formulaire d’entrée
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Titre de l’avis", placeholder="Ex. « Très déçu »")
    with col2:
        st.write("")  # espace
    body = st.text_area("Corps de l’avis", height=200, placeholder="Décrivez votre expérience…")

    submitted = st.form_submit_button("🔮 Prédire la note")

# Prédiction
if submitted:
    if not title.strip() and not body.strip():
        st.error("Veuillez au moins saisir le **Titre** ou le **Corps** de l’avis.")

    else:
        with st.spinner("Inférence en cours…"):
            try:
                y_pred, proba, latency = predict_with_model(model, title, body)
            except Exception as e:
                st.error(f"Erreur pendant la prédiction : {e}")
                st.stop()

        st.subheader("Résultat")
        st.metric(label="Note prédite pour cet avis", value=f"{y_pred} ⭐")
