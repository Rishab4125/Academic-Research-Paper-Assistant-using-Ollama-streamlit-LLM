import streamlit as st
import requests
import datetime


# Streamlit frontend
st.title("Academic Research Paper Assistant")
topic = st.text_input("Enter Research Topic:")
year = st.number_input("Enter Year:", min_value=2000, max_value=datetime.datetime.now().year, step=1)

# Streamlit actions
if st.button("Search Papers"):
    response = requests.post("http://127.0.0.1:8000/search_papers", json={"topic": topic, "year": year})
    papers = response.json().get("papers", [])
    for paper in papers:
        st.write(f"Title: {paper['title']}")
        st.write(f"Year: {paper['year']}")
        st.write(f"Abstract: {paper['abstract']}")
        st.write(f"URL: {paper['url']}")

question = st.text_input("Ask a Question:")
if question and st.button("Submit Question"):
    response = requests.post("http://127.0.0.1:8000/qa", json={"topic": topic, "year": year, "question": question})
    st.write(response.json().get("answers", []))