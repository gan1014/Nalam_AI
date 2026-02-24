
import os
import sys
import streamlit as st

# Debug marker for cloud logs
print("🚀 DEBUG: streamlit_app.py starting")

# Ensure the root directory is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import and run the actual app
try:
    print("🚀 DEBUG: Attempting to import frontend.app")
    import frontend.app
    print("🚀 DEBUG: frontend.app imported successfully")
except Exception as e:
    import traceback
    print(f"❌ DEBUG: Failed to launch the application: {e}")
    st.error(f"Failed to launch the application: {e}")
    st.code(traceback.format_exc(), language="python")

# If we reach here and it's still blank, show a final debug message
if "authenticated" not in st.session_state and not any(tag in str(st.session_state) for tag in ["authenticated"]):
    st.warning("⚠️ DEBUG: App reached end of streamlit_app.py but session state is largely empty. Check logs for silent failures.")
