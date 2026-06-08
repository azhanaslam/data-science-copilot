import streamlit as st
import polars as pl
import plotly.express as px

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


    st.write("### 🧠 Auto Insights")

    rows, cols = df.shape

    st.info(f"Dataset contains {rows:,} rows and {cols} columns.")

    missing_counts = df.null_count().row(0)

    for col_name, missing in zip(df.columns, missing_counts):
        if missing > 0:
            st.warning(
                f"Column '{col_name}' contains {missing} missing values."
            )

    st.write("### 📊 Visualizations")

    numeric_cols = [
        col for col, dtype in df.schema.items()
        if dtype.is_numeric()
    ]

    if numeric_cols:
        selected_col = st.selectbox(
            "Select a numeric column",
            numeric_cols
        )

        fig = px.histogram(
            df.to_pandas(),
            x=selected_col,
            title=f"Distribution of {selected_col}"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No numeric columns available for visualization.")       

    st.write("### 🔥 Correlation Heatmap")
    numeric_df = df.select(
        [col for col, dtype in df.schema.items()
        if dtype.is_numeric()]
    )

    if numeric_df.width >= 2:

        corr_df = numeric_df.to_pandas().corr(numeric_only=True)

        fig = px.imshow(
            corr_df,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(
            "Need at least two numeric columns for correlation analysis."
        )

    st.write("### 📦 Outlier Detection")

    numeric_cols = [
        col for col, dtype in df.schema.items()
        if dtype.is_numeric()
    ]

    if numeric_cols:

        selected_outlier_col = st.selectbox(
            "Select column for outlier analysis",
            numeric_cols,
            key="outlier_col"
        )

        series = df[selected_outlier_col].drop_nulls()

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series.filter(
            (series < lower_bound) |
            (series > upper_bound)
        )

        st.info(
            f"Detected {len(outliers)} potential outliers in '{selected_outlier_col}'."
        )

        fig = px.box(
            df.to_pandas(),
            y=selected_outlier_col,
            title=f"Boxplot - {selected_outlier_col}"
        )

        st.plotly_chart(fig, use_container_width=True)

# Analyse dataset button
    # adding a button
    st.write("### 🤖 Copilot Analysis")

    if st.button("Analyze Dataset", key="analyse_dataset_btn"):

    # dataset summary
        rows, cols = df.shape

        st.success(
            f"Dataset contains {rows:,} rows and {cols} columns."
        )
    # missing value percentages
        missing_df = df.null_count()

        for col in df.columns:

            missing = missing_df[col][0]

            if missing > 0:

                pct = (missing / rows) * 100

                st.warning(
                    f"{col}: {missing} missing values ({pct:.1f}%)"
                )
