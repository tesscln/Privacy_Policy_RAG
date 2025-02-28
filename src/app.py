import streamlit as st
from llm_analyzer import analyze_privacy_policy

st.title("🔍 AI-Powered Privacy Policy Compliance Checker")

policy_text = st.text_area("Paste your privacy policy here:")

if st.button("Analyze Compliance"):
    with st.spinner("Analyzing..."):
        analysis_result = analyze_privacy_policy(policy_text)
    st.success("Analysis Complete!")
    st.write(analysis_result)



