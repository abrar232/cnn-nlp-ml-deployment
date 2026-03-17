import requests
import streamlit as st

# ── Config ─────────────────────────────────────────────────────
API_URL = "http://nlp_api:8001"

# ── Page setup ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Ticket Triage",
    page_icon="🎫",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────────────
st.title("🎫 Support Ticket Triage")
st.write("Paste a customer support ticket to classify it, extract entities, and generate a draft response.")

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write("Pipeline:")
    st.write("1. DistilBERT classifies the ticket")
    st.write("2. Regex extracts key entities")
    st.write("3. Claude generates a draft response")

    st.divider()

    st.subheader("API Status")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            st.success("API is online")
        else:
            st.error("API returned an error")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API — is nlp_service running on port 8001?")

# ── Input ──────────────────────────────────────────────────────
ticket_text = st.text_area(
    "Paste ticket text here",
    height=150,
    placeholder="e.g. I cancelled ORD-65075291 but I still see a pending charge."
)

submit = st.button("Triage Ticket", type="primary")

# ── Results ────────────────────────────────────────────────────
if submit:
    if not ticket_text.strip():
        st.warning("Please enter some ticket text first.")
    else:
        with st.spinner("Running pipeline..."):
            response = requests.post(
                f"{API_URL}/triage",
                json={"text": ticket_text}
            )

        if response.status_code == 200:
            result = response.json()

            # ── Row 1 — classification metrics ─────────────────
            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Category",
                    value=result["category"]
                )

            with col2:
                st.metric(
                    label="Confidence",
                    value=f"{result['confidence']}%"
                )
                st.progress(result["confidence"] / 100)

            # ── Row 2 — entities ───────────────────────────────
            st.subheader("Extracted Entities")
            if result["entities"]:
                for entity in result["entities"]:
                    st.markdown(f"**{entity['label']}** — `{entity['text']}`")
            else:
                st.write("No entities found.")

            # ── Row 3 — draft response ─────────────────────────
            st.subheader("Draft Response")
            st.text_area(
                label="--",
                value=result["draft_response"],
                height=250
            )

        else:
            st.error(f"Request failed: {response.text}")