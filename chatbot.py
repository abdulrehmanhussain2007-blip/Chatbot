import os
from dotenv import load_dotenv
load_dotenv()

import openai
openai.api_key=os.getenv("OPEN_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
model=ChatGroq(model="openai/gpt-oss-120b",groq_api_key=groq_api_key)
print(model)

from langchain_core.messages import HumanMessage
result=model.invoke([HumanMessage(content="HI, My name is AR and Im a Machine Learning Engineering")])
print(result.content)
print(result)

from langchain_core.messages import AIMessage
results=model.invoke(
    [
        HumanMessage(content="HI, My name is AR and Im a Machine Learning Engineering"),
        AIMessage(content="Hello AR, its nice to meet you. \n\n what kind of project"),
        HumanMessage(content="Hey what's my name and what do i do")
    ]
)
print(results)
print(results.content)

#Create Conversaation history storage

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

#Seperating different sessions in history
#BaseChatMessageHistory is a previous chat

store={}
def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

#Using runnables to interact with the LLMs based on the Chat History.

with_message_history=RunnableWithMessageHistory(model, get_session_history)
config={"configurable":{"session_id":"chat1"}}

session_result=with_message_history.invoke(
    [
        HumanMessage(content="HI, My name is AR and Im a Machine Learning Engineering")
    ],
    config=config
)
print(session_result.content)

check=with_message_history.invoke(
    [
        HumanMessage(content="Hey what's my name and what is Machine Learning definition")
    ],
    config=config
)
print(check.content)

config1={"configurable":{"session_id":"chat2"}}

check1=with_message_history.invoke(
    [
        HumanMessage(content="Hey my name is Sayiqa and Im an AI instructor")
    ],
    config=config1
)
print(check1.content)

check3=with_message_history.invoke(
    [
        HumanMessage(content="What is my name and what do I do")
    ],
    config=config1
)
print(check3.content)

#Working with prompt Template
#prompt template helps to turns the raw users information into the format that the LLMs an works with.
#In this case the raw user input is just a message , which we are passing to a LLM, now here lets just add the s

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

prompt=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that helps the user to learn about Machine Learning and Data Science."),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain=prompt| model
chain.invoke({"messages":[
    HumanMessage(content="HI, My name is AR and Im a Machine Learning Engineering")
]})

#Add message history to chain to keep track of the conversation history.

with_message_history=RunnableWithMessageHistory(chain, get_session_history)
config={"configurable":{"session_id":"chat3"}}
response3=with_message_history.invoke(
    [
        HumanMessage(content="Hi, My name is Maaz and Im working in a university as an assistant professor.")
    ],
    config=config
)
print(response3.content)

#Manage conversation history
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately

trimmer=trim_messages(
    max_tokens=55,
    strategy="last",
    token_counter=count_tokens_approximately,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

messages=[
    SystemMessage(content="You are a helpful asssistant"),
    HumanMessage(content="Hi, I am Bob"),
    AIMessage(content="Hi"),
    HumanMessage(content="I like Vanilla ice cream"),
    AIMessage(content="Nice"),
    HumanMessage(content="What is 2+2"),
    AIMessage(content="4"),
    HumanMessage(content="Thanks"),
    AIMessage(content="No problem"),
    HumanMessage(content="Having fun"),
    AIMessage(content="Yes!"),
]
print(trimmer.invoke(messages))

from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

chain=(
    RunnablePassthrough.assign(messages=itemgetter("messages")|trimmer)
    |prompt
    |model
)

test1=chain.invoke(
    {
        "messages":messages + [HumanMessage(content="What's the math problem you solve today")]
    }
)
print(test1.content)