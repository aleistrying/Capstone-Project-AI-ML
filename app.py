"""
Legacy entry point kept for compatibility.
The canonical Streamlit app is app/streamlit_app.py.

Run:  streamlit run app/streamlit_app.py
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).parent / "app" / "streamlit_app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)], check=True
    )
