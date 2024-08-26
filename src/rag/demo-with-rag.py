from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_text_splitters import CharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import DirectoryLoader


# Invoke chain with RAG context
# llm = Ollama(model='llama3.1')

# 1. 初始化llm, 让其流式输出
llm = Ollama(model="llama3.1",
             temperature=0.1,
             top_p=0.4,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
             )

# Load page content
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

# loader = DirectoryLoader("./Books", glob="**/*.docx")
# books = loader.load()
# len(books)

# Vector store things
embeddings = OllamaEmbeddings(model="nomic-embed-text")

with open('../../examples/plaintext.txt', encoding='UTF-8') as f:
    last_question = f.read()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# text_splitter = RecursiveCharacterTextSplitter()
text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([last_question])

# vector_store = Chroma.from_documents(texts, embeddings)
vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="../../db/chroma_db",
)

# Prompt construction
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question only based on the given context
    
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Retrieve context from vector store
docs_chain = create_stuff_documents_chain(llm, prompt)

# 使用向量数据库作为检索器
retriever = vector_store.as_retriever()

retrieval_chain = create_retrieval_chain(retriever, docs_chain)

# Winner winner chicken dinner
response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})

print(":::ROUND 2:::")
print(response["answer"])
