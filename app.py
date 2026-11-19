"""
Root entry point for Streamlit Community Cloud deployment.
Routes directly to the Project 01 House Price Prediction dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project subfolder is in Python module search path
project_dir = Path(__file__).resolve().parent / "01-house-price-prediction"
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# Import main function from project 01 streamlit app
import streamlit_app

if __name__ == "__main__":
    streamlit_app.main()
