import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool,initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

#Stream lit UI

st.set_page_config(page_title="Text to Math and reasoning Chat bot using Data search assistant",page_icon=":math:")
st.title("Text to Math Problem solver and Reasoning Chat Bot")
groq_api_key = st.sidebar.text_input(label="Enter the Groq API Key",type="password")

if not groq_api_key:
    st.info("Enter the GROQ API key to continue")
    st.stop()

llm = ChatGroq(model="openai/gpt-oss-120b",groq_api_key=groq_api_key)

#Initialize Tools

wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name ="Wikipedia",
    func = wikipedia_wrapper.run,
    description = " A tool for searching the internet and getting the information"
)

math_tool = LLMMathChain.from_llm(llm=llm)
calc = Tool(
    name ="Calculator",
    func = math_tool.run,
    description="A tool for answering math related rquestions.Only input mathematical expressions"
)

prompt = """
You are a agent tasked for solving users mathematical questions.
Logically arrive at the solution and provide detailed explaination and provide it point wise
Question:{input}
Answer:
"""
prompt_template = PromptTemplate(
    input_variables=['input'],
    template=prompt
)

#Chain for the prompt template
Chain = LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning tool",
    func = Chain.run,
    description="A tool for answering logic based and reasoning questions"

)
#Combine tools and initialize agents
assistant_agent = initialize_agent(
    tools=[wikipedia_tool,calc,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = False,
    handle_parsing_errors=True
)

#Creating a session state for the Chat bot conversation
if "messages" not in st.session_state:
    st.session_state['messages']=[
        {"role":"assistant","content":"Hi, I am a math and reasoning Chatbot"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

question = st.text_area("Enter your question","What is the area of circle")

if st.button("Find answer"):
    if question:
        with st.spinner("Generating.."):
            st.session_state.messages.append({'role':'user','content':question})
            st.chat_message('user').write(question)
            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
            response = assistant_agent.run({'input':question},callbacks=[st_cb])
            st.session_state.messages.append({'role':'assistant','content':response})
            st.write('#Response')
            st.success(response)
    else:
        st.warning("Please enter your question")