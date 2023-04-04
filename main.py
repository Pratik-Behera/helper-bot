from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader   
from langchain.text_splitter import REcursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone

pdfPath = ""
loader = UnstructuredPDFLoader(pdfPath) 
data = loader.load()

print(f'You have {len(data)} document(s) in your data')
print(f'There are{len(data[0].page_content)} characters in your data')

#Chunking data into smaller data

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 4000 , chunk_overlap = 50)
texts = text_splitter.split_documents(data)
print(f'Now you have {len(texts)} documents ')


#Getting ready for semantic search by creating embeddings of the document 

OPENAI_API_KEY =  ""
PINECONE_API_KEY = ""
PINECONE_API_ENV = ""

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

pinecone.init(
    api_key=PINECONE_API_KEY,
    enviroment=PINECONE_API_ENV
)
index_name = "" #input the index name here 
docsearch = Pinecone.from_texts([t.page_content for t in texts],embeddings,index_name=index_name)

#The query should be user_input in the text box area
query = ""
docs = docsearch.similarity_search(query, include_meta = True)


#Quering these docs to get user_input answer

llm = OpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm,chain_type="stuff") # Three types of chain types "stuff","map_reduce","refine"


docs = docsearch.similarity_search(query, include_metadata=True)
chain.run(input_documents=docs,question=query)