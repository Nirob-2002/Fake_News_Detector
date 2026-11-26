import logging

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="NewsLens · Bilingual News Analysis",
    page_icon="📰",
    layout="wide",
)

import flask_app as engine


logger = logging.getLogger(__name__)
MAX_TEXT_LENGTH = 30000


def render_result(result):
    prediction = result["prediction"]
    if prediction == "Fake":
        st.warning("Fake news signal")
    else:
        st.success("Real news signal")

    st.caption("This is an automated prediction, not a confirmed fact-check.")

    model_name = result.get("model_name", "BiLSTM")
    if model_name == "SVM":
        model_name = "Optimized SVM"

    model_column, score_column, threshold_column = st.columns(3)
    model_column.metric("Model used", model_name)
    if result.get("model_name") == "SVM":
        score_column.metric("Decision score", f'{result["raw_score"]:.4f}')
        score_note = "SVM decision scores are not probabilities."
    else:
        score_column.metric(
            "Fake-class model score",
            f'{result["fake_probability"] * 100:.2f}%',
        )
        score_note = "Model scores are not calibrated certainty estimates."
    threshold_column.metric("Decision threshold", f'{result["threshold"]:.2f}')
    st.caption(score_note)


def render_detection_page():
    st.title("Check a news article before you share it")
    st.write(
        "Paste an English or Bangla article for a model-assisted assessment. "
        "Verify important claims with trusted sources."
    )

    with st.form("article_analysis"):
        language = st.selectbox("Article language", ["English", "Bangla"])
        news_text = st.text_area(
            "News article",
            placeholder="Paste the full article here...",
            max_chars=MAX_TEXT_LENGTH,
            height=220,
        )
        st.caption(f"{len(news_text):,} / {MAX_TEXT_LENGTH:,} characters")
        submitted = st.form_submit_button("Analyze article", type="primary")

    if language == "Bangla" and engine.bangla_bert_model is None:
        st.info("BanglaBERT is unavailable; the optimized SVM fallback is active.")
    elif language == "English" and engine.tf is None:
        st.info("BiLSTM is unavailable; the Logistic Regression fallback is active.")

    if submitted:
        if not news_text.strip():
            st.error("Please enter news content.")
            st.session_state.pop("last_result", None)
            return

        with st.spinner("Analyzing article..."):
            try:
                result = engine.predict_news(news_text, language)
            except Exception:
                logger.exception("Prediction failed")
                st.error("Prediction failed. Please check the application logs.")
                st.session_state.pop("last_result", None)
                return

            st.session_state["last_result"] = result

            try:
                engine.save_history(language, news_text, result)
            except Exception:
                logger.exception("Could not save prediction history")
                st.warning("The analysis succeeded, but its history could not be saved.")

    result = st.session_state.get("last_result")
    if result:
        st.divider()
        st.subheader("Model assessment")
        render_result(result)

    st.caption(
        "The model provides a signal based on its training data; it does not "
        "independently verify facts."
    )


def render_history_page():
    st.title("Detection history")
    if not engine.HISTORY_ENABLED:
        st.info("History is disabled for this deployment.")
        return

    connection = engine.get_db_connection()
    try:
        records = connection.execute(
            """
            SELECT language, news_text, prediction, fake_probability,
                   confidence, model_name, created_at
            FROM history
            ORDER BY id DESC
            LIMIT 100
            """
        ).fetchall()
    finally:
        connection.close()

    if not records:
        st.info("No analyses have been saved yet.")
        return

    st.dataframe(pd.DataFrame([dict(record) for record in records]))


def render_performance_page():
    st.title("Model performance")
    records = engine.get_performance_records()
    if not records:
        st.info("No model performance results are available.")
        return

    st.dataframe(pd.DataFrame(records))
    st.caption("Metrics are from held-out test sets and do not guarantee future performance.")


st.sidebar.title("NewsLens")
page = st.sidebar.radio("Navigation", ["Detect", "History", "Performance"])
st.sidebar.caption("English · বাংলা")

if page == "Detect":
    render_detection_page()
elif page == "History":
    render_history_page()
else:
    render_performance_page()
