import streamlit as st
import pandas as pd

@st.cache
def load_data(name):
    return pd.read_pickle(name)
