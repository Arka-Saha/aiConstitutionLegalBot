import streamlit as st
st.title("AI Legal Chatbot")

# Input
user_input = st.text_input("Ask your question:")

# Button
if st.button("Send"):
    if user_input:
        # You can access the user's input using user_input
        print("User input:", user_input)

        # Hardcoded output for now
        output = "This is a hardcoded response from the chatbot."

        # Display output
        st.write("### Chatbot:")
        st.write(output)
    else:
        st.write("Please enter a question.")