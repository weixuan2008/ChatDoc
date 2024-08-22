from langchain_community.chat_models import ChatOllama
from langchain.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os

os.environ['OPENAI_API_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImQzZjA3NmNmLTM0YWMtNGQ5MC04NzgxLTkwZGNjY2QyODQ5NSJ9.m_4j_GAqd9oIbDpIOCR2xS0Pr3a16RUXGRmGsH0Grgg'


# llm = ChatOllama(model='llama3.1')
llm = ChatOllama(model='qwen2:1.5b')

embedding = ChatOllama(modem='mxbai-embed-large:latest')

ollama = Ollama(base_url='http://localhost:11434', model="qwen2:1.5b")

with open('../../examples/plaintext.txt', encoding='UTF-8') as f:
    last_question = f.read()

text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([last_question])

vector_store = Chroma.from_documents(texts, embedding)

retriever = vector_store.as_retriever(
    search_type='',
    search_kwargs={
        'k': 3,
        'score_threshold': 0.5
    }
)

prompt = PromptTemplate.from_template("""Answer the question using the provided context. If the answer is
    not contained in the context, say "can not find in context" \n\n
    Context: {context}
    Question: {question}
    Answer:
    """
                                      )

chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

response = chain.invoke("世界上最高的山是哪座山？")
print(response)

response = chain.invoke("开发者平台是什么？")
print(response)