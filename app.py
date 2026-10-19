import streamlit as st

from backend.model import (
    classify_email,
    search_emails,
    get_statistics
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Email Intent AI",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background: #f6f8fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}


/* HEADER */

.hero {
    background: linear-gradient(
        135deg,
        #4f46e5,
        #7c3aed
    );

    padding: 35px;

    border-radius: 20px;

    color: white;

    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;

    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;

    opacity: 0.9;
}


/* CARDS */

.card {
    background: white;

    padding: 25px;

    border-radius: 16px;

    border: 1px solid #e7eaf0;

    box-shadow:
        0 5px 20px rgba(0,0,0,0.04);

    margin-bottom: 20px;
}


/* METRIC */

.metric-title {
    color: #667085;

    font-size: 14px;

    margin-bottom: 6px;
}

.metric-value {
    color: #111827;

    font-size: 28px;

    font-weight: 700;
}


/* RESULT */

.intent-result {
    font-size: 30px;

    font-weight: 700;

    color: #4f46e5;
}


/* EMAIL */

.email-box {
    background: #f8fafc;

    border-left: 4px solid #6366f1;

    padding: 16px;

    border-radius: 8px;

    color: #344054;
}


/* SMALL TEXT */

.muted {
    color: #667085;

    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>📧 Email Intent AI</h1>

<p>
Intelligent customer email analysis powered by
TF-IDF and Logistic Regression.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD STATISTICS
# ============================================================

stats = get_statistics()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ System")

    st.markdown("### Model")

    st.success("🟢 Model Online")

    st.write("Algorithm")

    st.code(
        "TF-IDF + Logistic Regression"
    )

    st.write("Dataset")

    st.code(
        f"{stats['total_emails']} emails"
    )

    st.write("Intent Categories")

    st.code(
        f"{stats['total_intents']} categories"
    )

    st.divider()

    st.markdown("### Intent Categories")

    for intent in stats["intent_counts"]:

        st.write(
            f"• {intent}"
        )


# ============================================================
# DASHBOARD METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown("""
    <div class="card">

    <div class="metric-title">
    📧 Emails
    </div>

    <div class="metric-value">
    48
    </div>

    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="card">

    <div class="metric-title">
    🎯 Intent Classes
    </div>

    <div class="metric-value">
    6
    </div>

    </div>
    """, unsafe_allow_html=True)


with col3:

    st.markdown("""
    <div class="card">

    <div class="metric-title">
    🤖 Algorithm
    </div>

    <div class="metric-value"
         style="font-size:20px;">
    Logistic Regression
    </div>

    </div>
    """, unsafe_allow_html=True)


with col4:

    st.markdown("""
    <div class="card">

    <div class="metric-title">
    🧠 NLP
    </div>

    <div class="metric-value"
         style="font-size:20px;">
    TF-IDF
    </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# QUERY SECTION
# ============================================================

st.markdown("## 🔎 Analyze Customer Query")

st.markdown(
    "Search the backend email database using natural language."
)


query = st.text_input(
    "Enter Query",
    placeholder=(
        "Example: customer says package has not arrived"
    )
)


analyze = st.button(
    "🚀 Analyze Query",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if not query.strip():

        st.warning(
            "Please enter a query."
        )

    else:

        with st.spinner(
            "Searching emails and analyzing intent..."
        ):

            results = search_emails(
                query,
                top_k=5
            )


        # ====================================================
        # BEST MATCH
        # ====================================================

        best_match = results[0]

        email = best_match["email"]

        result = classify_email(
            email
        )


        st.markdown("---")

        st.markdown(
            "## 🎯 Analysis Result"
        )


        # ====================================================
        # RESULT CARDS
        # ====================================================

        col1, col2, col3 = st.columns(3)


        with col1:

            st.markdown("""
            <div class="card">

            <div class="metric-title">
            Predicted Intent
            </div>

            <div class="intent-result">
            """ +
            result["intent"] +
            """
            </div>

            </div>
            """,
            unsafe_allow_html=True)


        with col2:

            st.markdown("""
            <div class="card">

            <div class="metric-title">
            Confidence
            </div>

            <div class="metric-value">
            """ +
            f"{result['confidence']:.2%}" +
            """
            </div>

            </div>
            """,
            unsafe_allow_html=True)


        with col3:

            st.markdown("""
            <div class="card">

            <div class="metric-title">
            Priority
            </div>

            <div class="metric-value">
            """ +
            result["priority"] +
            """
            </div>

            </div>
            """,
            unsafe_allow_html=True)


        # ====================================================
        # MATCHED EMAIL
        # ====================================================

        st.markdown("### 📨 Most Relevant Email")

        st.markdown(
            f"""
            <div class="email-box">
            {email}
            </div>
            """,
            unsafe_allow_html=True
        )


        st.write(
            f"Semantic similarity: "
            f"**{best_match['similarity']:.2%}**"
        )


        # ====================================================
        # OTHER MATCHES
        # ====================================================

        st.markdown(
            "### 🔍 Other Relevant Emails"
        )


        for i, item in enumerate(
            results[1:],
            start=2
        ):

            with st.expander(
                f"Match {i} — "
                f"{item['intent']} — "
                f"{item['similarity']:.2%}"
            ):

                st.write(
                    item["email"]
                )


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown("---")

st.markdown("## 🧠 How It Works")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.info(
        "1️⃣ Query\n\n"
        "User enters a natural-language query."
    )


with col2:

    st.info(
        "2️⃣ TF-IDF\n\n"
        "Text is converted into numerical features."
    )


with col3:

    st.info(
        "3️⃣ ML Model\n\n"
        "Logistic Regression predicts the intent."
    )


with col4:

    st.info(
        "4️⃣ Result\n\n"
        "Intent, confidence and priority are shown."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    "<div style='text-align:center; color:#98a2b3;'>"
    "NLP Email Intent Classifier • "
    "TF-IDF + Logistic Regression"
    "</div>",
    unsafe_allow_html=True
)