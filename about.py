import streamlit as st

def about_dataset():
    st.write('**Tentang Dataset**')
    col1, col2= st.columns([5,5])

    with col1:
        link = "https://t3.ftcdn.net/jpg/08/08/99/46/360_F_808994683_OUZEZt581lOYP0H2zGRXMZKJ5g87jkjx.jpg"
        st.image(link, caption="Loan Approval")

    with col2:
        st.write('This dataset is used to analyze borrower ' \
        'demographics, financial, and credit behavior to predict ' \
        'loan approval outcomes. By identifying key risk drivers ' \
        'and borrower patterns, financial institutions can optimize' \
        'credit decision processes, minimize default risk, improve, ' \
        'portfolio quality, and enhance overall lending efficiency ' \
        'while maintaining regulatory compliance.')
