import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Mall Customer Clustering", layout="wide")

st.title("🛍️ Mall Customer Segmentation")
st.markdown("Welcome to an interactive dashboard to explore customer behavior using unsupervised learning.")

with st.sidebar:
    selected = option_menu("Navigation", ["Data Overview", "Correlation Heatmap", "Clustering"],
                           icons=["table", "bar-chart-line", "diagram-3"],
                           menu_icon="cast", default_index=0)

if selected == "Data Overview":
    import pages.Overview as overview
    overview.show()

elif selected == "Correlation Heatmap":
    import pages.Heatmap as heatmap
    heatmap.show()

elif selected == "Clustering":
    import pages.Clustering as cluster
    cluster.show()
