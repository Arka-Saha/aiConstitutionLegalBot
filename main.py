from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
import chromadb
import os, json, ast
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
load_dotenv()

# global DEBUG
DEBUG = False

# pypdf = PyPDFLoader()
content = ""
with open("summary.txt", "r") as f:
    for line in f.readlines():
        content += line

with open("conversations.txt", "r") as file:
    convo_history = file.read()

# gemini =genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n", ". ", " "]
)
chunks = splitter.split_text(content)
print(f"Total chunks: {len(chunks)}")

def embed_texts(texts):
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(result["embedding"])
    return embeddings

chunk_embeddings = embed_texts(chunks)
print(f"Embedded {len(chunk_embeddings)} chunks")

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="python_docs")

collection.add(
    documents=chunks,
    embeddings=chunk_embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
# global convo_data
# convo_data = ""
print(f"Stored {collection.count()} chunks in ChromaDB")

types = """"
"Police / Arrest Issue" → maps to BNSS sections on FIR filing, arrest rights, bail
"Consumer / Shopping Issue" → maps to Consumer Protection Act (defective products, refunds, e-commerce fraud)
"Workplace Harassment" → maps to POSH Act (sexual harassment at workplace)
"Domestic Violence / Safety at Home" → maps to Protection of Women from Domestic Violence Act
"Government Info Request" → maps to RTI Act (how to file, timelines, appeals)
"Child Safety Concern" → maps to POCSO Act
"General / Not Sure" (fallback) → skips pre-filled context, goes to open freeform cha
"""


def rag_answer(query):

    if DEBUG == True:
        json_object = {"query":query, "response":"This is a test answer as of now."}
        convo_data = f"{json_object}\n"
        print(convo_data)
    else:

        query_embedding = genai.embed_content(
            model="models/gemini-embedding-001",
            content=query,
            task_type="retrieval_query"
        )["embedding"]

        # Retrieve from ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )

        retrieved_chunks = results["documents"][0]
        context = "\n\n".join(retrieved_chunks)

        # Generate
        prompt = f"""
        You are a empathetic legal aid assistant for Indian citizens. Rules:
        - Only answer using the provided context below. If the context doesn't cover the question, say so — don't guess.
        - Explain in simple, non-legal language (assume no legal background).
        - Always cite the specific Act and section you're referencing.
        - Never give definitive legal advice or predict case outcomes.
        - If the situation sounds urgent or serious, recommend contacting a lawyer 
        or legal aid clinic.
        - Also create a 'type' from the given types of legal categories, in which the situation falls.
        - Add small, but formal empathetic phrases when needed. Not too much, keep it simple and formal.
        - Dont use stars ** to show bold. Its not needed!
        - The answers should be clear and short, and precise.

        - STRICTLY FOLLOW THE GIVEN OUTPUT FORMAT

        Context: {context}
        Conversation history: {convo_history}
        User question: {query}
        types: {types}


        OUTPUT FORMAT:
        ["the type of legal thing", "the output"]
    """

        model = genai.GenerativeModel("gemini-3.6-flash")
        response = model.generate_content(prompt)

        print(f"Q: {query}")
        # print("Retrieved:")
        # for chunk in retrieved_chunks:
        #     print(f"  {chunk[:80]}...")
        # print(f"A: {response.text}\n")

        response_list = ast.literal_eval(response.text)

        print(response_list[1])

        json_object = {"query":query, "response":response_list[1], "type": response_list[0]}
        convo_data = f"{json_object}\n"
        # json_object.
    
    with open("conversations.txt", "a+") as file:
        file.write(convo_data)

for i in range(0, 1):
    q = input("enter query: ")
    rag_answer(q)

# print(convo_data)
