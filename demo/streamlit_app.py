"""
Streamlit demo for AI Code Reviewer.
"""

import json
import requests
import streamlit as st
from typing import Dict, Optional

# Page configuration
st.set_page_config(
    page_title="AI Code Reviewer",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"


def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_model_info() -> Optional[Dict]:
    """Get model information from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def predict_code(code: str) -> Optional[Dict]:
    """Predict code using the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"code": code},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error making prediction: {e}")
    return None


def get_examples() -> list:
    """Get example code snippets."""
    try:
        response = requests.get(f"{API_BASE_URL}/examples", timeout=5)
        if response.status_code == 200:
            return response.json().get("examples", [])
    except:
        pass
    return []


def main():
    # Header
    st.title("🐛 AI Code Reviewer")
    st.markdown("Detect buggy vs clean Python code using CodeBERT")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This demo uses a fine-tuned CodeBERT model to classify Python code as either:
        
        - **🐛 Buggy**: Contains potential bugs or issues
        - **✅ Clean**: Appears to be correct code
        
        The model was trained on the BugsInPy dataset with real-world Python bugs.
        """)
        
        # API status
        st.header("API Status")
        if check_api_health():
            st.success("✅ API is running")
            
            # Model info
            model_info = get_model_info()
            if model_info:
                st.info(f"Model: {model_info['model_name']}")
                st.info(f"Base: {model_info['base_model']}")
                st.info(f"Max length: {model_info['max_length']} tokens")
        else:
            st.error("❌ API is not running")
            st.markdown("""
            Please start the API server:
            ```bash
            make serve
            ```
            """)
        
        # Examples
        st.header("Examples")
        examples = get_examples()
        if examples:
            for example in examples:
                with st.expander(example["name"]):
                    st.code(example["code"], language="python")
                    if st.button(f"Try: {example['name']}", key=example["name"]):
                        st.session_state.example_code = example["code"]
    
    # Main content
    if not check_api_health():
        st.error("""
        ## API Server Not Running
        
        Please start the API server first:
        
        ```bash
        make serve
        ```
        
        Then refresh this page.
        """)
        return
    
    # Code input
    st.header("Code Analysis")
    
    # Get example code from session state if available
    default_code = st.session_state.get("example_code", "")
    
    code_input = st.text_area(
        "Enter Python code to analyze:",
        value=default_code,
        height=200,
        placeholder="def example_function():\n    # Your code here\n    pass"
    )
    
    # Clear session state after using example
    if "example_code" in st.session_state:
        del st.session_state.example_code
    
    # Prediction button
    col1, col2 = st.columns([1, 4])
    with col1:
        predict_button = st.button("🔍 Analyze Code", type="primary")
    
    with col2:
        if predict_button and code_input.strip():
            with st.spinner("Analyzing code..."):
                result = predict_code(code_input.strip())
                
                if result:
                    # Display result
                    st.success("Analysis complete!")
                    
                    # Create columns for result display
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        if result["label"] == "buggy":
                            st.error("🐛 **Buggy Code Detected**")
                        else:
                            st.success("✅ **Clean Code Detected**")
                    
                    with result_col2:
                        confidence = result["score"]
                        st.metric("Confidence", f"{confidence:.1%}")
                        
                        # Confidence bar
                        if result["label"] == "buggy":
                            st.progress(confidence, text="Buggy confidence")
                        else:
                            st.progress(confidence, text="Clean confidence")
                    
                    # Additional analysis
                    st.subheader("Analysis Details")
                    
                    if result["label"] == "buggy":
                        st.warning("""
                        **Potential Issues Detected:**
                        - This code may contain bugs or problematic patterns
                        - Consider reviewing for common issues like:
                          - Division by zero
                          - Index out of bounds
                          - Unhandled exceptions
                          - Logic errors
                        """)
                    else:
                        st.info("""
                        **Code Analysis:**
                        - This code appears to be clean and well-structured
                        - No obvious bugs or problematic patterns detected
                        - Consider this a good starting point for your implementation
                        """)
    
    # Code display with syntax highlighting
    if code_input.strip():
        st.subheader("Code Preview")
        st.code(code_input, language="python")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>AI Code Reviewer - Powered by CodeBERT</p>
        <p>Built with FastAPI and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
