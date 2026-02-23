
import os
import sys
import streamlit as st

# Ensure the root directory is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import and run the actual app
try:
    # Streamlit scripts are often top-level.
    # We simply import it to let it execute.
    import frontend.app
except Exception as e:
    st.error(f"Failed to launch the application: {e}")
