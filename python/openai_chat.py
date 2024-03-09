from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os

from langchain_openai import AzureOpenAIEmbeddings
from collections import deque

api_version = "2023-07-01-preview"
endpoint = "https://llm-proxy.intelligence.test.com/azure/v1"

# token = token_generator.get_token()
token = 'XXXX'

os.environ["AZURE_OPENAI_API_KEY"] = token
os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version=api_version,
)

llm = AzureChatOpenAI(
    azure_deployment="gpt-4",
    openai_api_version=api_version,
    temperature=0,
    model_name="gpt-4")

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Initialise Langchain - Conversation Retrieval Chain
qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(search_type="mmr"), max_tokens_limit=4000)

user_history_queue_map = {}


def embed_documents(path='test_context/20240225/txt', add_mod=False):
    dir_loader = DirectoryLoader(path)
    raw_documents = dir_loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(raw_documents)
    print('documents created:', len(documents))
    if not add_mod:
        vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
    else:
        vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
        vectorstore.add_documents(documents)
    vectorstore.persist()


def ask_chat(user_message, user_id=''):
    print('start ask chat for user:', user_id)
    try:
        # append background prompt
        prompt_message = 'You are an expert of AI and ML in OPS, and this is user query who want ' \
                         'to know something based on the context and background you know. Now user question is:' + \
                         str(user_message)

        if user_id not in user_history_queue_map.keys():
            user_history_queue_map[user_id] = deque(maxlen=5)
        user_history_queue = user_history_queue_map.get(user_id)
        user_history = list(user_history_queue)
        response = qa({"question": prompt_message, "chat_history": user_history})
        user_history_queue.append((response['question'], response['answer']))
        return response["answer"]
    except Exception as e:
        print('chat qa error:', e)


if __name__ == "__main__":
    # embed_documents(path='test_context/20240118/txt', add_mod=False)
    embed_documents(path='test_context/20240225/txt', add_mod=True)
    result = ask_chat('Write Spark job in Java?', '')
    print(result)
