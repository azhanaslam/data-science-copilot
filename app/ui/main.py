import streamlit as st
import polars as pl

st.title("🚀 Data Science Copilot")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file:
    df = pl.read_csv(uploaded_file)

    st.success("File loaded successfully!")

    st.write("### Preview")
    st.dataframe(df.head(10))

    st.write("### 📊 Dataset Info")
    st.write(df.shape)
    st.write(df.columns)

    st.write("### ❗ Missing Values")
    missing = df.null_count()
    st.dataframe(missing)

    tab1, tab2 = st.tabs([
        "📊 All Statistics",
        "🔢 Numeric Only"
    ])

    with tab1:
        st.dataframe(df.describe())

    with tab2:
        numeric_df = df.select(pl.col(pl.NUMERIC_DTYPES))

        if numeric_df.width > 0:
            st.dataframe(numeric_df.describe())
        else:
            st.warning("No numeric columns found.")