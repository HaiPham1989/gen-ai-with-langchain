import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

## streamlit
st.set_page_config(page_title="Text To Math")
st.title("Text To Math Problem Solver Using Google Gemma 2")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

## Init the tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="a tool for searching the Internet to find vatious information",
)

## Init the Math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tools for answering math related question. Only input mathemmatical expression need to be provided",
)

prompt = """
You are an agent for solving users mathematical question.
Logically arrive at the solution and provide a detailed explaination and display it wise for the question below
Question: {question}
Answer:
"""
prompt_template = PromptTemplate(input_variables=["question"], template=prompt)

## Combine all the tools into chain
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions.",
)

## Initialize the agents
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm a Math chatbot who can answer all your maths question",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


## Let start the interaction
question = st.text_area(
    "Enter your question: I have 5 bananas and 8 grapes. How many fruit I have at end?"
)

if st.button("find my answer"):
    if question:
        with st.spinner("Generate response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)
            try:
                st_cb = StreamlitCallbackHandler(
                    st.container(), expand_new_thoughts=False
                )
                response = assistant_agent.run(
                    st.session_state.messages, callbacks=[st_cb]
                )

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                st.write("### Reponse:")
                st.success(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter the question")
