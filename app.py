from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
import os

# you can load your groq api key from system variables, or you can input it to the .env file
# for chainlit, .env is loaded automatically

#from dotenv import load_dotenv
#load_dotenv()  #
groq_api_key = os.environ['GROQ_API_KEY']

#if you want to use local ollama model
#llm_local = ChatOllama(model="mistral:instruct")
llm_groq = ChatGroq(
            groq_api_key=groq_api_key,
            #model_name='llama2-70b-4096' 
            model_name='mixtral-8x7b-32768'
    )

DB_CHROMA_PATH = 'vectorstore/db_chroma'

custom_prompt_template = """You are a friendly and helpful medical chatbot. Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#QA Model Function
def qa_bot():
    embeddings = OllamaEmbeddings(model="nomic-embed-text",  
                                  model_kwargs={'device': 'cpu'})
    db = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embeddings)
    llm = llm_groq
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hello, I'm a Medical Bot. Ask away any medical questions!"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += "\n" + f"\nSources:" + str(sources)
    else:
        answer += "\n" + "\nNo sources found"

    await cl.Message(content=answer).send()