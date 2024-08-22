from langchain_community.document_loaders import Docx2txtLoader, UnstructuredExcelLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

import os

os.environ['OPENAI_API_KEY'] = 'dummy_key'
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"


class ChatDoc:
    def __init__(self):
        self.doc = None
        self.splitText = []

    def getFile(self):
        doc = self.doc
        loaders = {
            'docx': Docx2txtLoader,
            'pdf': PyPDFLoader,
            'xlsx': UnstructuredExcelLoader,
        }
        file_extension = doc.split('.')[-1]
        loader_class = loaders.get(file_extension)

        if loader_class:
            try:
                loader = loader_class(doc)
                text = loader.load()
                return text
            except Exception as ex:
                print(f'Error loading {file_extension} files: {ex}')
                return None
        else:
            print(f'Unsupported file extension: {file_extension}')
            return None

    def splitSentences(self):
        full_text = self.getFile()
        if full_text is not None:
            text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=20)
            texts = text_splitter.split_documents(full_text)
            self.splitText = texts

    def embeddingAndVectorDB(self):
        embedding = OpenAIEmbeddings()
        db = Chroma.from_documents(documents=self.splitText, embedding=embedding)
        return db

    def askAndFindFiles(self, question):
        db = self.embeddingAndVectorDB()

        llm = ChatOpenAI(temperature=0)
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=db.as_retriever(),
            llm=llm,
        )
        return retriever_from_llm.get_relevant_documents(question);


inst = ChatDoc()
# inst.doc = 'examples/camunda2.pdf'
inst.doc = 'examples/AUS Booking Tech Sharing.docx'
# inst.doc = 'examples/EmployeeList.xlsx'
inst.splitSentences()
print(inst.splitText)

# vectorDB = inst.embeddingAndVectorDB()
# print(vectorDB)

#inst.askAndFindFiles('What is the Booking?')
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.DEBUG)
unique_doc_content = inst.askAndFindFiles('What is your company name?')
print(unique_doc_content)
