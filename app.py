import streamlit as st
from llama_cpp import Llama

# Initialize the model
llm = Llama.from_pretrained(
    repo_id="Divyansh12/check",
    filename="unsloth.F16.gguf",  #
    verbose=True,
    n_ctx=32768,
    n_threads=2,
    chat_format="chatml"
)

# Define the function to get responses from the model
def respond(message, history):
    messages = []

    for user_message, assistant_message in history:
        if user_message:
            messages.append({"role": "user", "content": user_message})


    response = ""
    # Stream the response from the model
    response_stream = llm.create_chat_completion(
        messages=messages,
        stream=True,
        max_tokens=512,  
        temperature=0.7,  
        top_p=0.95  
    )

    # Collect the response chunks
    for chunk in response_stream:
        if len(chunk['choices'][0]["delta"]) != 0 and "content" in chunk['choices'][0]["delta"]:
            response += chunk['choices'][0]["delta"]["content"]

    return response  # Return the full response

# Streamlit UI
st.title("Simple Chatbot")
st.write("### Interact with the chatbot!")

# User input field
user_message = st.text_area("Your Message:", "")

# Chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Button to send the message
if st.button("Send"):
    if user_message:  # Check if user has entered a message
        # Get the response from the model
        response = respond(user_message, st.session_state.history)

        # Add user message and model response to history
        st.session_state.history.append((user_message, response))

        # Clear the input field after sending
        user_message = ""  

# Display the chat history
st.write("### Chat History")
for user_msg, assistant_msg in st.session_state.history:
    st.write(f"**User:** {user_msg}")
    st.write(f"**Assistant:** {assistant_msg}")
