from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from sentence_transformers import SentenceTransformer
import chromadb
import os
import json
import ast
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GEMINI CLIENT
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Check your .env file."
    )

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# LOCAL EMBEDDING MODEL
# ============================================================
# This replaces Gemini embeddings.
#
# Therefore:
# - No Gemini embedding quota is used
# - Document embeddings are generated locally
# - Query embeddings are also generated locally
# ============================================================

print("Loading local embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Local embedding model loaded.")


# ============================================================
# SETTINGS
# ============================================================

DEBUG = False


# ============================================================
# LOAD LEGAL DOCUMENTS
# ============================================================

documents = []


# ------------------------------------------------------------
# summary.txt
# ------------------------------------------------------------

if os.path.exists("summary.txt"):

    print("Loading summary.txt...")

    with open(
        "summary.txt",
        "r",
        encoding="utf-8"
    ) as f:

        summary_content = f.read()

    documents.append(summary_content)

else:

    print("WARNING: summary.txt not found.")


# ------------------------------------------------------------
# POSH ACT
# ------------------------------------------------------------

if os.path.exists("poshact.pdf"):

    print("Loading poshact.pdf...")

    posh_loader = PyPDFLoader(
        "poshact.pdf"
    )

    posh_pages = posh_loader.load()

    for page in posh_pages:

        if page.page_content.strip():

            documents.append(
                page.page_content
            )

else:

    print("WARNING: poshact.pdf not found.")


# ------------------------------------------------------------
# CONSTITUTION
# ------------------------------------------------------------

if os.path.exists("constitution.pdf"):

    print("Loading constitution.pdf...")

    constitution_loader = PyPDFLoader(
        "constitution.pdf"
    )

    constitution_pages = (
        constitution_loader.load()
    )

    for page in constitution_pages:

        if page.page_content.strip():

            documents.append(
                page.page_content
            )

else:

    print(
        "WARNING: constitution.pdf not found."
    )


# ============================================================
# COMBINE DOCUMENTS
# ============================================================

content = "\n\n".join(
    documents
)

print(
    f"Total document characters: "
    f"{len(content)}"
)


# ============================================================
# LOAD CONVERSATION HISTORY
# ============================================================

if os.path.exists(
    "conversations.txt"
):

    with open(
        "conversations.txt",
        "r",
        encoding="utf-8"
    ) as file:

        convo_history = file.read()

else:

    convo_history = ""


# ============================================================
# CHUNKING
# ============================================================

splitter = RecursiveCharacterTextSplitter(

    chunk_size=800,

    chunk_overlap=100,

    separators=[
        "\n",
        ". ",
        " "
    ]
)

chunks = splitter.split_text(
    content
)

print(
    f"Total chunks: {len(chunks)}"
)


# ============================================================
# CHROMADB
# ============================================================

print(
    "Opening persistent ChromaDB..."
)

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = (
    chroma_client.get_or_create_collection(
        name="python_docs"
    )
)


# ============================================================
# CREATE DOCUMENT EMBEDDINGS ONLY IF NEEDED
# ============================================================

if collection.count() == 0:

    print(
        "ChromaDB is empty."
    )

    print(
        "Creating local document embeddings..."
    )


    # --------------------------------------------------------
    # LOCAL EMBEDDINGS
    # --------------------------------------------------------

    chunk_embeddings = (
        embedding_model.encode(

            chunks,

            show_progress_bar=True
        )
    )


    # Convert numpy arrays to lists
    chunk_embeddings = (
        chunk_embeddings.tolist()
    )


    print(
        f"Embedded "
        f"{len(chunk_embeddings)} chunks"
    )


    # --------------------------------------------------------
    # STORE IN CHROMADB
    # --------------------------------------------------------

    collection.add(

        documents=chunks,

        embeddings=chunk_embeddings,

        ids=[
            f"chunk_{i}"
            for i in range(len(chunks))
        ]
    )


    print(
        f"Stored "
        f"{collection.count()} "
        f"chunks in ChromaDB"
    )


else:

    print(
        f"Loaded existing ChromaDB "
        f"with {collection.count()} chunks"
    )


# ============================================================
# LEGAL CATEGORIES
# ============================================================

types = """
"Police / Arrest Issue" → maps to BNSS sections on FIR filing,
arrest rights, bail

"Consumer / Shopping Issue" → maps to Consumer Protection Act
(defective products, refunds, e-commerce fraud)

"Workplace Harassment" → maps to POSH Act
(sexual harassment at workplace)

"Domestic Violence / Safety at Home" → maps to
Protection of Women from Domestic Violence Act

"Government Info Request" → maps to RTI Act
(how to file, timelines, appeals)

"Child Safety Concern" → maps to POCSO Act

"General / Not Sure" (fallback) → skips pre-filled context,
goes to open freeform chat
"""


# ============================================================
# RAG ANSWER
# ============================================================

def rag_answer(query):

    # ========================================================
    # DEBUG MODE
    # ========================================================

    if DEBUG:

        json_object = {

            "query": query,

            "response":
                "This is a test answer as of now.",

            "type":
                "General / Not Sure"
        }

        convo_data = (
            json.dumps(
                json_object,
                ensure_ascii=False
            )
            + "\n"
        )

        print(convo_data)

    else:

        # ====================================================
        # EMBED USER QUERY LOCALLY
        # ====================================================

        print(
            "Creating query embedding..."
        )

        query_embedding = (
            embedding_model.encode(
                [query]
            )[0]
        )

        query_embedding = (
            query_embedding.tolist()
        )


        # ====================================================
        # RETRIEVE RELEVANT DOCUMENT CHUNKS
        # ====================================================

        print(
            "Searching legal documents..."
        )

        results = collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=5
        )


        # ====================================================
        # EXTRACT RETRIEVED CHUNKS
        # ====================================================

        retrieved_chunks = (
            results["documents"][0]
        )


        context = "\n\n".join(
            retrieved_chunks
        )


        # ====================================================
        # PROMPT
        # ====================================================

        prompt = f"""
You are an empathetic legal aid assistant for Indian citizens.

Your job is to provide simple, cautious, source-grounded
legal information.

RULES:

1. Only answer using the provided context below.

2. If the context does not contain enough information to answer
the question, clearly say that the available documents do not
contain enough information.

3. Do not invent legal provisions, Acts, sections, deadlines,
rights, procedures, authorities, or penalties.

4. Explain everything in simple, non-legal language.

5. Assume the user has no legal background.

6. Whenever the provided context supports it, mention the
specific Act and section.

7. Never give definitive legal advice.

8. Never predict the outcome of a legal case.

9. If the situation sounds urgent or serious, recommend
contacting a lawyer, legal aid clinic, or appropriate authority.

10. Keep the answer short, clear, practical and precise.

11. Do not use stars ** for bold formatting.

12. Do not unnecessarily repeat the user's question.

13. Do not assume facts that the user has not provided.

IDENTITY RULES:

14. Never assume the user's gender, age, identity, occupation,
relationship, or personal circumstances unless explicitly stated.

15. Use gender-neutral language such as:
"you", "the complainant", "the affected person",
or "the person involved".

16. Never address the user as "woman", "man", "he", "she",
"sir", or "madam" unless the user explicitly provides that
information.

17. The person asking the question may be asking on behalf of
someone else.

18. Do not assume that the user is the person affected by
the situation.

LEGAL CATEGORY RULE:

Choose exactly one category from the provided categories.

Available categories:

{types}


CONTEXT FROM LEGAL DOCUMENTS:

{context}


RECENT CONVERSATION HISTORY:

{convo_history[-5000:]}


USER QUESTION:

{query}


OUTPUT REQUIREMENT:

Return ONLY a Python-style list containing exactly two strings.

The first string must be the legal category.

The second string must be the answer.

Example:

["Workplace Harassment", "The person affected may..."]


Do not add:

- Markdown
- Code fences
- Explanations outside the list
- "Here is your answer"
- Extra fields
"""


        # ====================================================
        # GEMINI GENERATION
        # ====================================================

        print(
            "Generating legal response..."
        )

        response = client.models.generate_content(

            model="gemini-3.6-flash",

            contents=prompt
        )


        # ====================================================
        # GET RESPONSE TEXT
        # ====================================================

        response_text = (
            response.text.strip()
        )


        print(
            f"\nQ: {query}"
        )


        # ====================================================
        # PARSE GEMINI RESPONSE
        # ====================================================

        try:

            # Remove markdown code fences
            if response_text.startswith(
                "```"
            ):

                response_text = (
                    response_text
                    .replace(
                        "```python",
                        ""
                    )
                    .replace(
                        "```json",
                        ""
                    )
                    .replace(
                        "```",
                        ""
                    )
                    .strip()
                )


            response_list = (
                ast.literal_eval(
                    response_text
                )
            )


            # Make sure we actually received
            # two values

            if (
                not isinstance(
                    response_list,
                    list
                )
                or len(response_list) < 2
            ):

                raise ValueError(
                    "Invalid response format"
                )


            legal_type = str(
                response_list[0]
            )

            answer = str(
                response_list[1]
            )


        except Exception as e:

            print(
                "\nWarning: Gemini did not "
                "follow the expected format."
            )

            print(
                f"Parsing error: {e}"
            )

            legal_type = (
                "General / Not Sure"
            )

            answer = response_text


        # ====================================================
        # PRINT ANSWER
        # ====================================================

        print(
            f"\nType: {legal_type}"
        )

        print(
            f"\nAnswer:\n{answer}"
        )


        # ====================================================
        # SAVE CONVERSATION
        # ====================================================

        json_object = {

            "query": query,

            "response": answer,

            "type": legal_type
        }


        convo_data = (
            json.dumps(
                json_object,
                ensure_ascii=False
            )
            + "\n"
        )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    with open(
        "conversations.txt",
        "a+",
        encoding="utf-8"
    ) as file:

        file.write(
            convo_data
        )


    # ========================================================
    # RETURN ANSWER
    # ========================================================

    return {
        "type": legal_type,
        "answer": answer
    }


# ============================================================
# TEST MODE
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "======================================"
    )
    print(
        " Indian Legal Aid Bot"
    )
    print(
        "======================================"
    )
    print()

    q = input(
        "Enter query: "
    )

    rag_answer(q)