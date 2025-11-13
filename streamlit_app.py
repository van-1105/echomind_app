# streamlit_app.py
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

# ------- Simple keyword rule fallback -------
RULES = {
    "positive": ["growth", "opportunity", "love", "learn", "improve", "benefit", "success", "comfortable", "excited"],
    "negative": ["stress", "pressure", "fear", "risk", "loss", "anxiety", "difficult", "hard", "lonely"],
    "regret": ["miss", "chance", "what if", "maybe", "doubt", "later", "regret", "should have"]
}

def rule_analyze(text):
    t = text.lower()
    happy = sum(word in t for word in RULES["positive"]) * 20
    stress = sum(word in t for word in RULES["negative"]) * 20
    regret = sum(word in t for word in RULES["regret"]) * 20
    return min(happy, 100), min(stress, 100), min(regret, 100)

# ------- Try to load a Hugging Face emotion model (safe) -------
@st.cache_resource(show_spinner=False)
def load_emotion_pipe():
    try:
        from transformers import pipeline
        # small emotion model — good balance between size and accuracy
        pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
        return pipe
    except Exception as e:
        # If loading fails, return None and we will use the rule-based fallback
        return None

emotion_pipe = load_emotion_pipe()

# ------- Map emotion output to our scores -------
# We'll map emotions returned by the model to happiness, stress, regret.
# Adjust mapping as you like.
EMOTION_TO_SCORES = {
    "joy": ("happiness", 1.0),
    "love": ("happiness", 0.9),
    "optimism": ("happiness", 0.8),
    "admiration": ("happiness", 0.6),

    "anger": ("stress", 1.0),
    "fear": ("stress", 0.9),
    "sadness": ("regret", 0.9),
    "disgust": ("stress", 0.7),

    "surprise": ("happiness", 0.4),
    "neutral": ("happiness", 0.2),
    # any other emotion falls back to neutral mapping
}

def model_analyze(text):
    """
    Use the transformer pipeline to get emotion scores and convert to happiness/stress/regret (0-100).
    If the model isn't available, return None.
    """
    if emotion_pipe is None:
        return None

    try:
        out = emotion_pipe(text[:512])  # limit length for speed
        # out is list of list (return_all_scores=True); take first element
        scores = out[0]  # [{'label': 'joy', 'score': 0.6}, ...]
        # initialize
        happy_val = 0.0
        stress_val = 0.0
        regret_val = 0.0

        for item in scores:
            label = item["label"].lower()
            score = float(item["score"])  # 0..1
            if label in EMOTION_TO_SCORES:
                kind, weight = EMOTION_TO_SCORES[label]
                if kind == "happiness":
                    happy_val += score * weight
                elif kind == "stress":
                    stress_val += score * weight
                elif kind == "regret":
                    regret_val += score * weight
            else:
                # unknown -> treat mildly as neutral/happiness
                happy_val += score * 0.1

        # Convert to percent (0..100), clamp
        h = min(int(happy_val * 100), 100)
        s = min(int(stress_val * 100), 100)
        r = min(int(regret_val * 100), 100)
        return h, s, r
    except Exception:
        return None

# ------- UI -------
st.set_page_config(page_title="EchoMind AI", layout="centered")
st.title("EchoMind AI — Emotional Impact Predictor")
st.write("This demo uses a Hugging Face emotion model if available. If the model cannot be loaded it will fall back to simple keyword rules.")

with st.form("decision_form"):
    decision = st.text_input("What decision are you confused about?", placeholder="e.g. Should I accept a job in another city?")
    st.write("Describe Option A and Option B briefly (one or two sentences).")
    optionA = st.text_area("Option A", placeholder="e.g. Move to Bangalore: higher pay, new team, relocation stress.")
    optionB = st.text_area("Option B", placeholder="e.g. Stay here: comfortable, close to family, slower growth.")
    submitted = st.form_submit_button("Analyze")

if submitted:
    if not decision.strip() or (not optionA.strip() and not optionB.strip()):
        st.error("Please enter the decision and at least one option.")
    else:
        st.subheader("Decision")
        st.write(decision)

        # Analyze options: try model first, fallback to rules
        a_scores = model_analyze(optionA) if optionA.strip() else None
        b_scores = model_analyze(optionB) if optionB.strip() else None

        if a_scores is None:
            a_scores = rule_analyze(optionA)
            st.info("Using simple keyword rules for Option A (model not available).")
        else:
            st.success("Model-based analysis for Option A")

        if b_scores is None:
            b_scores = rule_analyze(optionB)
            st.info("Using simple keyword rules for Option B (model not available).")
        else:
            st.success("Model-based analysis for Option B")

        # unpack
        a_h, a_s, a_r = a_scores
        b_h, b_s, b_r = b_scores

        # show metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Option A")
            st.metric("Happiness", f"{a_h}%")
            st.metric("Stress", f"{a_s}%")
            st.metric("Regret", f"{a_r}%")
        with col2:
            st.markdown("### Option B")
            st.metric("Happiness", f"{b_h}%")
            st.metric("Stress", f"{b_s}%")
            st.metric("Regret", f"{b_r}%")

        # Chart: prepare dataframe for Altair
        df = pd.DataFrame({
            "metric": ["Happiness", "Stress", "Regret"] * 2,
            "value": [a_h, a_s, a_r, b_h, b_s, b_r],
            "option": ["A"] * 3 + ["B"] * 3
        })

        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('metric:N', title='Metric'),
            y=alt.Y('value:Q', title='Score (%)'),
            color='option:N',
            column='option:N'
        ).properties(width=250, height=300)

        st.altair_chart(chart, use_container_width=False)

        # Final suggestion based on regret score primarily (you can refine)
        st.subheader("Suggestion")
        if a_r < b_r:
            st.success("👉 Option A may cause less regret (based on current analysis).")
        elif b_r < a_r:
            st.success("👉 Option B may cause less regret (based on current analysis).")
        else:
            st.info("Both options show similar regret scores — consider other factors (career goals, finances, etc.).")

        # Save to local CSV for future training/analysis
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "decision": decision,
            "optionA": optionA,
            "A_happiness": int(a_h),
            "A_stress": int(a_s),
            "A_regret": int(a_r),
            "optionB": optionB,
            "B_happiness": int(b_h),
            "B_stress": int(b_s),
            "B_regret": int(b_r)
        }
        try:
            prev = pd.read_csv("results.csv")
            prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
            prev.to_csv("results.csv", index=False)
        except FileNotFoundError:
            pd.DataFrame([row]).to_csv("results.csv", index=False)

        st.info("Analysis saved to results.csv (local file).")
