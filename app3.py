#fact verification frontend.py

import streamlit as st
import requests

backend_URL = "http://localhost:8000"

st.title("fact verification")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

uploaded_file = st.file_uploader("upload pdf", type = "pdf")

if uploaded_file and st.session_state.session_id is None: #so we do not index the doc everytime streamlit reruns
    with st.spinner("uploading doc"):
        response = requests.post(
            f"{backend_URL}/upload",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(),"application/pdf")},
        )
    data =  response.json()

    if data.get("status") == "done":
        st.session_state.session_id = data["session_id"]
        st.success(f"uploaded doc, chunked: {data['chunks']}")
    else:
        st.error("error")

for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.write(msg)
user_input = st.chat_input("ask")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append(("user", user_input))

    with st.spinner("verifying"):
        if st.session_state.session_id:
            response = requests.post(
                f"{backend_URL}/response",
                json={
                    "session_id": st.session_state.session_id,
                    "question": user_input
                }
            )
            answer = response.json().get("answer","error")
        else:
            answer = "upload doc pls"
    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.messages.append(("assistant", answer))