import streamlit as st


def metric_card(title, value):

    st.metric(
        label=title,
        value=value
    )


def section_title(title):

    st.subheader(title)


def show_dataframe(df):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )