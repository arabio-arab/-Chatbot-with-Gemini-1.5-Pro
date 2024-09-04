import streamlit as st 

from main_chatbot import  create_vector_db, get_qa_chain
st.markdown(
    """
    <h1 style='color: blue; text-align: center;'>_____🤖__CHATbot__🤖_______</h1>
    """,
    unsafe_allow_html=True)

st.markdown("<h3 style='color: orange;text-align: center;'>______🏢__Company ARABI.AI__🏢_______</h3>", unsafe_allow_html=True)


B=st.button("creat new data")
if B:
    create_vector_db()

Q=st.text_input("for free aske me any thing about the company ARABI.com:")
if Q:
    chain = get_qa_chain()
    response = chain(Q)

    st.header("Answer")
    st.write(response["result"])