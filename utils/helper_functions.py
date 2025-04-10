import base64
from io import BytesIO

def get_image_download_link(fig, filename="plot.png", text="Download Plot"):
    """Generate a download link for a matplotlib figure"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

def load_data(file_path):
    """Load data from CSV file with error handling"""
    import pandas as pd
    try:
        data = pd.read_csv(file_path)
        return data, None
    except FileNotFoundError:
        return None, f"File '{file_path}' not found."
    except Exception as e:
        return None, f"Error loading data: {str(e)}"