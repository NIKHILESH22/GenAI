# import streamlit as st

# st.title("Hello GenAI 🚀")
# st.write("Streamlit is working!")





# import streamlit as st
# import requests

# st.title("Simple Ollama App")

# # User input
# prompt = st.text_input("Enter your question:")

# if st.button("Submit"):
#     if prompt:
#         response = requests.post(
#             "http://localhost:11434/api/generate",
#             json={
#                 "model": "gemma2:2b",
#                 "prompt": prompt,
#                 "stream": False
#             }
#         )

#         result = response.json()["response"]
#         st.write("Response:")
#         st.write(result)



import streamlit as st
import pandas as pd
import numpy as np

st.write("my first streamlit project")
st.write(":streamlit: NIKHILESH")
st.text("lets start")

name=st.text_input("Enter name:")
if st.button("GREET"):
    st.success(f"Hello There ! {name}")

df=pd.DataFrame(np.random.randn(10,2),columns=['A','B'])
st.line_chart(df)
st.bar_chart(df)

st.sidebar.title("NAVIGATION")
st.image("https://th.bing.com/th/id/OIP.WUbA5qz5Q91opWzsdHdBbgHaEA?w=331&h=180&c=7&r=0&o=7&pid=1.7&rm=3",caption="AGERA RS")
st.video("https://youtu.be/Z1yGy9fELtE")

st.title("abc Text and Markdown Demo")
st.header("This is a header")
st.subheader("This is a subheader")
st.markdown("**Bold**, *Italic*, `Code`, [Link](https://streamlit.io)")
st.code("for i in range(5): print(i)", language="python")


st.text_input("What's your name?")
st.text_area("Write something...")
st.number_input("Pick a number", min_value=0, max_value=100)
st.slider("Choose a range", 0, 100)
st.selectbox("Select a fruit", ["Apple", "Banana", "Mango"])
st.multiselect("Choose toppings", ["Cheese", "Tomato", "Olives"])
st.radio("Pick one", ["Option A", "Option B"])
st.checkbox("I agree to the terms")

if st.checkbox("Show Details"):
    st.info("Here are more details...")