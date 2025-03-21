import streamlit as st
from llm_analyzer import analyze_privacy_policy, _get_classifier
import json
import chardet
import os
import sys
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable unnecessary warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Force CPU usage if MPS (Mac M1) is causing issues
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Configure Streamlit
st.set_page_config(
    page_title="Analyze your privacy policy",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_model_cache():
    """Cache the model initialization to avoid reloading."""
    try:
        return _get_classifier()
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        st.error("Failed to initialize the model. Please try again later.")
        return None

def read_file_with_encoding(file_content):
    """Try to read file content with different encodings."""
    try:
        # First, try to detect the encoding
        result = chardet.detect(file_content)
        detected_encoding = result['encoding']
        
        # List of encodings to try
        encodings = [
            detected_encoding,  # Try detected encoding first
            'utf-8',
            'latin-1',
            'iso-8859-1',
            'cp1252',
            'ascii'
        ]
        
        # Remove None and duplicates
        encodings = list(dict.fromkeys([e for e in encodings if e]))
        
        for encoding in encodings:
            try:
                return file_content.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode file with any of the attempted encodings: {encodings}")
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise

@st.cache_data
def analyze_with_error_handling(policy_text):
    """Wrapper function to handle analysis errors gracefully and cache results."""
    try:
        return analyze_privacy_policy(policy_text)
    except Exception as e:
        logger.error(f"Error during policy analysis: {str(e)}")
        st.error(f"An error occurred during analysis: {str(e)}")
        return None

def main():
    try:
        st.title("Analyze your privacy policy")
        st.write("Compliance score w.r.t. the GDPR and CNIL.")
        
        # Initialize model at startup
        with st.spinner("Loading model..."):
            classifier = initialize_model_cache()
            if classifier is None:
                st.stop()
            st.success("Model loaded successfully!")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Upload .txt file", "Paste text"]
        )
        
        policy_text = None
        
        if input_method == "Upload .txt file":
            uploaded_file = st.file_uploader("Upload your privacy policy", type=["txt"])
            if uploaded_file is not None:
                try:
                    file_content = uploaded_file.read()
                    policy_text = read_file_with_encoding(file_content)
                    st.success("File uploaded successfully!")
                    with st.expander("View uploaded text"):
                        st.text(policy_text)
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    st.info("Please ensure the file is a valid text file. If the issue persists, try copying and pasting the text directly.")
        else:
            policy_text = st.text_area(
                "Enter your privacy policy text:",
                height=300,
                placeholder="Paste your privacy policy here..."
            )
        
        if st.button("Analyze Policy"):
            if policy_text and policy_text.strip():
                with st.spinner("Analyzing privacy policy..."):
                    try:
                        # Analyze the policy with error handling and caching
                        analysis = analyze_with_error_handling(policy_text)
                        
                        if analysis:
                            # Display results
                            st.header("Analysis Results")
                            
                            # Overall compliance
                            compliance = analysis["overall_compliance"]
                            st.subheader("Compliance Score")
                            
                            # Create columns for score display
                            col1, col2, col3 = st.columns([2,1,1])
                            with col1:
                                st.metric(
                                    "Overall Score",
                                    f"{compliance['score']:.1f}%",
                                    compliance["status"]
                                )
                            
                            # Score details
                            st.subheader("Score Components")
                            details = compliance["details"]
                            
                            # Create columns for metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Violations", details["violations"])
                            with col2:
                                st.metric("Severity Score", f"{details['severity_score']:.2%}")
                            with col3:
                                st.metric("Coverage", f"{details['coverage_score']:.2%}")
                            
                            # Display coverage score components
                            coverage_breakdown = details["coverage_breakdown"]
                            st.info(f"""
                            Coverage Score Components:
                            - Source Diversity: {coverage_breakdown['source_coverage']:.2%} (having texts from GDPR Articles, Recitals, and CNIL)
                            - Quantity Coverage: {coverage_breakdown['quantity_coverage']:.2%} (number of relevant texts found)
                            
                            Final coverage score is weighted: 60% source diversity + 40% quantity
                            """)
                            
                            st.markdown("---")
                            
                            # Top recommendations
                            if analysis["recommendations"]:
                                st.subheader("Key Recommendations")
                                for rec in analysis["recommendations"]:
                                    st.write(f"**{rec['priority']} Priority**: {rec['description']}")
                            
                            # Relevant legal texts
                            with st.expander("View Relevant Legal Texts"):
                                # Add source filter
                                source_filter = st.multiselect(
                                    "Filter by source:",
                                    ["GDPR Articles", "GDPR Recitals", "CNIL Articles"],
                                    default=["GDPR Articles", "GDPR Recitals", "CNIL Articles"]
                                )
                                
                                # Create source mapping
                                source_map = {
                                    "GDPR Articles": "GDPR_text",
                                    "GDPR Recitals": "GDPR_recitals",
                                    "CNIL Articles": "CNIL_text"
                                }
                                
                                # Filter texts by selected sources
                                filtered_texts = [
                                    text for text in analysis["relevant_legal_texts"]
                                    if any(text["source"] == source_map[src] for src in source_filter)
                                ]
                                
                                for text in filtered_texts:
                                    # Extract source information
                                    source = text["source"]
                                    content = text["content"]
                                    
                                    # Determine source icon and label
                                    if "GDPR_recitals" in source:
                                        source_label = "📜 GDPR Recital"
                                    elif "GDPR_text" in source:
                                        source_label = "📋 GDPR Article"
                                    elif "CNIL_text" in source:
                                        source_label = "🔒 CNIL Article"
                                    else:
                                        source_label = "📄 Other Source"
                                    
                                    # Display the text with its source
                                    st.markdown(f"**{source_label}**")
                                    st.text(content)
                                    st.markdown("---")  # Add a separator between texts
                            
                            # Violations section
                            with st.expander("View Detected Violations"):
                                st.markdown("### Detected Violations")
                                
                                # Get aspect analysis results
                                aspect_results = analysis["aspect_analysis"]
                                
                                # Display violations with severity levels, confidence, explanations, and proofs
                                for aspect, result in aspect_results.items():
                                    if not result["compliant"]:
                                        # Determine severity icon
                                        severity = result["severity"]
                                        if severity >= 0.8:
                                            severity_icon = "🔴"  # Red for critical/high
                                        elif severity >= 0.5:
                                            severity_icon = "🟡"  # Yellow for medium
                                        else:
                                            severity_icon = "🟢"  # Green for low
                                        
                                        # Format confidence as percentage
                                        confidence = result["confidence"] * 100
                                        
                                        # Get explanation and violation proof
                                        explanation = result.get("explanation", "No explanation provided")
                                        violation_proof = result.get("violation_proof", "No violation proof available")
                                        
                                        # Display violation with explanation and proof
                                        st.markdown(f"""
                                        {severity_icon} **{aspect.title()}**
                                        - Severity: {severity_icon} {result.get('severity', 'Unknown')}
                                        - Confidence: {confidence:.1f}%
                                        - Explanation: _{explanation}_
                                        - Proof: {violation_proof}
                                        """)
                                        st.markdown("---")
                                
                                if not any(not result["compliant"] for result in aspect_results.values()):
                                    st.success("No violations detected! 🎉")
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        logger.error(f"Analysis error: {str(e)}")
            else:
                st.warning("Please provide a privacy policy text to analyze.")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()

