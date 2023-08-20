# import streamlit as st
# from langchain.chains import ConversationChain
# from langchain.chains.conversation.memory import ConversationEntityMemory
# from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
# from langchain.llms import OpenAI
# import openai  # Import the openai library

# # Define the MODEL variable before calling initialize_entity_memory
# MODEL = st.sidebar.selectbox(label='Model', options=['gpt-3.5-turbo', 'text-davinci-003', 'text-davinci-002', 'code-davinci-002'])

# if "generated" not in st.session_state:
#     st.session_state["generated"] = []  # output
# if "past" not in st.session_state:
#     st.session_state["past"] = []
# if "input" not in st.session_state:
#     st.session_state["input"] = ""  # Initialize as an empty string
# if "stored_session" not in st.session_state:
#     st.session_state["stored_session"] = []

# # Initialize entity memory function
# def initialize_entity_memory():
#     """
#     Initialize the entity memory if it doesn't exist.
#     """
#     if 'entity_memory' not in st.session_state:
#         api = "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"

#         if api:
#             llm = OpenAI(
#                 temperature=0,
#                 openai_api_key=api,
#                 model_name=MODEL,
#             )

#             st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=10)
#         else:
#             st.error("No API found")

# # Create LangModel
# initialize_entity_memory()
# llm = st.session_state.entity_memory.llm  # Retrieve llm from entity_memory

# # Get user input
# def get_text():
#     """
#     Get the user input text.
#     Returns:
#         (str): The text entered by the user
#     """
#     input_text = st.text_input("You: ", st.session_state["input"], key="input", placeholder="Your AI assistant here! Ask me anything....", label_visibility='hidden')
#     return input_text

# st.title("Gen-AI Bot")

# # Create ConversationChain
# Conversation = ConversationChain(
#     llm=llm,
#     prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
#     memory=st.session_state.entity_memory
# )

# user_input = get_text()

# # Define a function to trigger the fashion generator
# #def trigger_fashion_generator():
#     # Implement code to generate a fashion outfit here
#     # You can call your fashion generator function or API here

# # Define a function to trigger the diffusion model
# def trigger_diffusion_model():
#     # Define the prompt for the diffusion model
#     prompt = "Summarize the conversation so far: " + "\n".join(st.session_state["past"])

#     # Send the prompt to the diffusion model
#     response = openai.Completion.create(
#         engine="DALL-E",  # or the model you want to use
#         prompt=prompt,
#         max_tokens=150,  # Adjust max tokens as needed
#         n=1,  # Generate a single response
#         stop=None,  # Let the model decide when to stop
#         temperature=0.7,  # Adjust temperature as needed
#         api_key="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"  # Replace with your API key
#     )

#     # Extract and display the response from the diffusion model
#     st.success(response.choices[0].text)

# # Check if the trigger conditions are met
# if user_input:
#     if user_input.lower() == "summarize" or user_input.lower() == "generate":
#         trigger_diffusion_model()  # Trigger the diffusion model for summarization
#     # elif "generate" in user_input.lower() :
#     #     trigger_fashion_generator()
#     else:
#         output = Conversation.run(input=user_input)
#         st.session_state.past.append(user_input)
#         st.session_state.generated.append(output)

# with st.expander("Conversation"):
#     for i in range(len(st.session_state['generated'])-1, -1, -1):
#         st.info(st.session_state["past"][i])
#         st.success(st.session_state["generated"][i])


import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
import openai  # Import the openai library

# Define the MODEL variable before calling initialize_entity_memory
MODEL = st.sidebar.selectbox(label='Model', options=['gpt-3.5-turbo', 'text-davinci-003', 'text-davinci-002', 'code-davinci-002'])

if "generated" not in st.session_state:
    st.session_state["generated"] = []  # output
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""  # Initialize as an empty string
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# Initialize entity memory function
def initialize_entity_memory():
    """
    Initialize the entity memory if it doesn't exist.
    """
    if 'entity_memory' not in st.session_state:
        api = "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"

        if api:
            llm = OpenAI(
                temperature=0,
                openai_api_key=api,
                model_name=MODEL,
            )

            st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=10)
        else:
            st.error("No API found")

# Create LangModel
initialize_entity_memory()
llm = st.session_state.entity_memory.llm  # Retrieve llm from entity_memory

# Get user input
def get_text():
    """
    Get the user input text.
    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input", placeholder="Your AI assistant here! Ask me anything....", label_visibility='hidden')
    return input_text

st.title("Gen-AI Bot")

# Create ConversationChain
Conversation = ConversationChain(
    llm=llm,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=st.session_state.entity_memory
)

user_input = get_text()

# Define a function to trigger the fashion generator
#def trigger_fashion_generator():
    # Implement code to generate a fashion outfit here
    # You can call your fashion generator function or API here

# Define a function to trigger the diffusion model
def trigger_diffusion_model():
    # Define the prompt for the diffusion model
    prompt = "Summarize the conversation so far: " + "\n".join(st.session_state["past"])

    # Send the prompt to the diffusion model
    response = openai.Completion.create(
        engine="DALL.E",  # or the model you want to use
        prompt=prompt,
        max_tokens=150,  # Adjust max tokens as needed
        n=1,  # Generate a single response
        stop=None,  # Let the model decide when to stop
        temperature=0.7,  # Adjust temperature as needed
        api_key="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"  # Replace with your API key
    )

    # Extract and display the response from the diffusion model
    st.success(response.choices[0].text)

# Check if the trigger conditions are met
if user_input:
    if user_input.lower() == "summarize" or user_input.lower() == "generate":
        trigger_diffusion_model()  # Trigger the diffusion model for summarization
    # elif "generate" in user_input.lower() :
    #     trigger_fashion_generator()
    else:
        output = Conversation.run(input=user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

with st.expander("Conversation"):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i])
        st.success(st.session_state["generated"][i])