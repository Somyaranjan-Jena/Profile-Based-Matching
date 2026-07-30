import os
import sys
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.similarity import users
from models.scoring import recommend_matches

st.set_page_config(
    page_title="AI Profile Matching",
    page_icon="🤝",
    layout="wide"
)

st.markdown("""
<style>
.main{
    padding-top:1rem;
}

.profile-card{
    background:#f8f9fa;
    padding:18px;
    border-radius:12px;
    border:1px solid #dddddd;
}

.match-card{
    background:#ffffff;
    padding:16px;
    border-radius:12px;
    border:1px solid #e5e5e5;
    margin-bottom:15px;
}

.small{
    color:#666666;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

st.title("🤝 AI Profile Matching System")
st.caption("Semantic Similarity • MBTI • Interest Matching")

selected_user = st.selectbox(
    "Select User",
    users["user_id"].tolist()
)

profile = users[users["user_id"] == selected_user].iloc[0]

st.markdown("---")

st.subheader("👤 Selected Profile")

with st.container(border=True):

    col1, col2 = st.columns(2)

    with col1:

        st.write(f"**Name:** {profile['name']}")
        st.write(f"**Profession:** {profile['profession']}")
        st.write(f"**Location:** {profile['location']}")
        st.write(f"**MBTI:** {profile['mbti']}")

    with col2:

        st.write(f"**Experience:** {profile['experience_years']} Years")
        st.write("**Interests**")
        st.write(profile["interests"])

    st.write("**Professional Summary**")
    st.info(profile["professional_summary"])

    st.write("**About Me**")
    st.write(profile["about_me"])

st.markdown("---")

st.subheader("⭐ Top 5 Recommended Matches")

matches = recommend_matches(
    selected_user,
    top_n=5
)

table = matches[
    [
        "Name",
        "Profession",
        "Location",
        "MBTI Type",
        "Compatibility"
    ]
].copy()

table["Compatibility"] = table["Compatibility"].map(
    lambda x: f"{x:.2f}%"
)

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

st.subheader("📄 Match Details")

selected_match = st.selectbox(
    "Select Match",
    matches["Name"].tolist()
)

match = matches[
    matches["Name"] == selected_match
].iloc[0]

with st.container(border=True):

    left, right = st.columns([2,1])

    with left:

        st.write(f"### {match['Name']}")
        st.write(f"**Profession:** {match['Profession']}")
        st.write(f"**Location:** {match['Location']}")
        st.write(f"**MBTI:** {match['MBTI Type']}")

    with right:

        st.metric(
            "Compatibility",
            f"{match['Compatibility']:.2f}%"
        )

    st.markdown("### Compatibility Breakdown")

    st.write("Semantic Similarity")
    st.progress(match["Semantic"] / 100)

    st.write("MBTI Match")
    st.progress(match["MBTI"] / 100)

    st.write("Interest Match")
    st.progress(match["Interest"] / 100)

    st.write("Location Match")
    st.progress(match["Location Score"] / 100)

    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Semantic",
            f"{match['Semantic']:.2f}"
        )

    with c2:
        st.metric(
            "MBTI",
            f"{match['MBTI']:.2f}"
        )

    with c3:
        st.metric(
            "Interest",
            f"{match['Interest']:.2f}"
        )

    with c4:
        st.metric(
            "Location",
            f"{match['Location Score']:.2f}"
        )

st.markdown("---")

st.subheader("📋 Recommendation Summary")

best_match = matches.iloc[0]

left, right = st.columns([3, 1])

with left:
    st.success(
        f"""
**Best Match:** {best_match['Name']}

**Profession:** {best_match['Profession']}

**Location:** {best_match['Location']}

**MBTI:** {best_match['MBTI Type']}
"""
    )

with right:
    st.metric(
        "Best Compatibility",
        f"{best_match['Compatibility']:.2f}%"
    )

st.markdown("---")

st.caption(
    "Developed as a Major Project | AI Profile Matching using Sentence Transformers, MBTI Personality Matching and Hybrid Recommendation Algorithm"
)
