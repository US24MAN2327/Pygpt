import os
from constants import key

from langchain_groq import ChatGroq
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import SequentialChain

import streamlit as st

# Set the Groq API key
os.environ["GROQ_API_KEY"] = key

# Streamlit app configuration
st.set_page_config(page_title="PYGPT: Code Generator", layout="wide")
st.title("🚀 PYGPT: Python Code Generator")
st.markdown(
    """
    **Generate Python Code, Describe It, and See Examples Instantly!**
    <br>
    Enter a topic below, and watch as the app generates Python code, provides a description, and gives examples.
    """,
    unsafe_allow_html=True,
)

# Input text box for topic
input_text = st.text_input("🎯 **Enter a Topic to Generate Code**:", placeholder="e.g., Sorting algorithms, API integration, etc.")

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=["name"],
    template="Write Python code on this topic: {name}"
)

second_input_prompt = PromptTemplate(
    input_variables=["code"],
    template="Describe this given code: {code}"
)

third_input_prompt = PromptTemplate(
    input_variables=["description"],
    template="Give an example of this given description: {description}"
)

# Memory
person_memory = ConversationBufferMemory(input_key="name", memory_key="chat_history")
dob_memory = ConversationBufferMemory(input_key="code", memory_key="chat_history")
descr_memory = ConversationBufferMemory(input_key="description", memory_key="description_history")

# Groq LLM setup
llm = ChatGroq(temperature=0.8, model="llama3-8b-8192")

# Chains
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key="code", memory=person_memory)
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key="description", memory=dob_memory)
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key="example", memory=descr_memory)

# Sequential Chain
parent_chain = SequentialChain(
    chains=[chain, chain2, chain3],
    input_variables=["name"],
    output_variables=["code", "description", "example"],
    verbose=True,
)

# Streamlit Execution
if input_text:
    with st.spinner("🔄 Generating results..."):
        result = parent_chain({"name": input_text})

    # Display results in an aesthetic way
    st.success("🎉 Results Generated!")
    
    st.subheader("🛠️ **Generated Python Code**")
    st.code(result.get("code", "No code generated."), language="python")

# Display the Code Description with improved formatting and color contrast
    st.subheader("📜 **Code Description**")
    st.markdown(
        f"""
        <div style="background-color:#2b2d42; color:#ffffff; padding:15px; border-radius:10px; 
                    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); font-family: 'Courier New', monospace; 
                    line-height: 1.5; font-size: 1rem;">
            {result.get('description', 'No description generated.')}
        </div>
        """,
        unsafe_allow_html=True
    )


    st.subheader("💡 **Example Usage**")
    st.markdown(
        f"""
        <div style="background-color:#f0f8ff; color:#003366; padding:15px; border-radius:10px; 
                    border: 1px solid #cfe2f3; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); 
                    font-family: 'Arial', sans-serif; line-height: 1.6; font-size: 1rem;">
            <pre style="margin:0; font-family: 'Courier New', monospace; color: #003366;">
                {result.get('example', 'No example generated.')}
            </pre>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Expanders for additional details
    with st.expander("🔍 **Code Generation History**"):
        st.info(person_memory.buffer)

    with st.expander("🔍 **Description History**"):
        st.info(descr_memory.buffer)

    with st.expander("🔍 **Example History**"):
        st.info(dob_memory.buffer)

# Footer
st.markdown(
    """
    <hr>
    <footer style="text-align: center; font-size: 0.9em;">
        Made with ❤️ using Streamlit & LangChain. Powered by Groq.
    </footer>
    """,
    unsafe_allow_html=True,
)
