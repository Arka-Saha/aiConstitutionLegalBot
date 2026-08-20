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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# LOCAL EMBEDDING MODEL
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

# IMPORTANT:
# We use a new collection name because the old Chroma collection
# did not contain source metadata.
COLLECTION_NAME = "legal_docs_v2"


# ============================================================
# LOAD LEGAL DOCUMENTS
# ============================================================

documents = []


def load_pdf(filename, source_name):
    """Load a PDF and keep the source/page information."""
    if not os.path.exists(filename):
        print(f"WARNING: {filename} not found.")
        return

    print(f"Loading {filename}...")

    loader = PyPDFLoader(filename)
    pages = loader.load()

    for page_number, page in enumerate(pages, start=1):
        if page.page_content.strip():
            documents.append({
                "text": page.page_content,
                "source": source_name,
                "page": page_number
            })


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

if os.path.exists("summary.txt"):
    print("Loading summary.txt...")

    with open(
        "summary.txt",
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    if text.strip():
        documents.append({
            "text": text,
            "source": "summary.txt",
            "page": 0
        })
else:
    print("WARNING: summary.txt not found.")


# ------------------------------------------------------------
# LEGAL DOCUMENTS
# ------------------------------------------------------------

load_pdf(
    "poshact.pdf",
    "POSH Act"
)

load_pdf(
    "constitution.pdf",
    "Constitution of India"
)

load_pdf(
    "bns.pdf",
    "Bharatiya Nyaya Sanhita (BNS)"
)

load_pdf(
    "bnss.pdf",
    "Bharatiya Nagarik Suraksha Sanhita (BNSS)"
)

load_pdf(
    "protection_of_civil_rights_act.pdf",
    "Protection of Civil Rights Act"
)

load_pdf(
    "sc_st_act.pdf",
    "SC/ST Prevention of Atrocities Act"
)


# ============================================================
# COMBINE DOCUMENTS
# ============================================================

content = "\n\n".join(
    item["text"]
    for item in documents
)

print(
    f"Total document characters: {len(content)}"
)


# ============================================================
# LOAD CONVERSATION HISTORY
# ============================================================

if os.path.exists("conversations.txt"):

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

chunks = []
chunk_metadatas = []

for document in documents:

    source_chunks = splitter.split_text(
        document["text"]
    )

    for chunk in source_chunks:

        chunks.append(chunk)

        chunk_metadatas.append({
            "source": document["source"],
            "page": document["page"]
        })


print(
    f"Total chunks: {len(chunks)}"
)


# ============================================================
# CHROMADB
# ============================================================

print("Opening persistent ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ============================================================
# CREATE DOCUMENT EMBEDDINGS ONLY IF NEEDED
# ============================================================

if collection.count() == 0:

    print("ChromaDB collection is empty.")
    print("Creating local document embeddings...")

    chunk_embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=True
    )

    chunk_embeddings = chunk_embeddings.tolist()

    print(
        f"Embedded {len(chunk_embeddings)} chunks"
    )

    collection.add(
        documents=chunks,
        embeddings=chunk_embeddings,
        metadatas=chunk_metadatas,
        ids=[
            f"legal_chunk_{i}"
            for i in range(len(chunks))
        ]
    )

    print(
        f"Stored {collection.count()} chunks in ChromaDB"
    )

else:

    print(
        f"Loaded existing ChromaDB with "
        f"{collection.count()} chunks"
    )


# ============================================================
# LEGAL CATEGORIES
# ============================================================

types = """
"Police / Arrest Issue"
→ BNS and BNSS provisions involving criminal
offences, assault, homicide, FIR, arrest,
investigation, bail and criminal procedure.

"Consumer / Shopping Issue"
→ Consumer Protection Act
(defective products, refunds, e-commerce fraud).

"Workplace Harassment"
→ POSH Act
(sexual harassment at workplace).

"Domestic Violence / Safety at Home"
→ Protection of Women from Domestic Violence Act.

"Government Info Request"
→ RTI Act
(how to file, timelines, appeals).

"Child Safety Concern"
→ POCSO Act.

"Untouchability / Caste Discrimination"
→ Constitution of India, Protection of Civil Rights Act,
and SC/ST Prevention of Atrocities Act.

"General / Not Sure"
→ fallback category.
"""


# ============================================================
# QUERY ROUTING
# ============================================================

def detect_legal_topics(query):
    """
    Detect topics that need targeted retrieval.

    This is NOT used to decide the legal result.
    It only makes sure that highly specific legal documents
    are retrieved instead of relying only on semantic similarity.
    """

    q = query.lower()

    topics = []

    caste_keywords = [
        "untouchable",
        "untouchability",
        "caste",
        "caste discrimination",
        "scheduled caste",
        "scheduled tribes",
        "scheduled tribe",
        "sc/st",
        "sc st",
        "dalit",
        "jati",
        "discrimination",
        "caste abuse",
        "caste insult",
        "caste slur",
        "temple entry",
        "denied entry",
        "public place",
        "social boycott"
    ]

    assault_keywords = [
        "beat",
        "beaten",
        "beating",
        "assault",
        "hit me",
        "attacked",
        "attack",
        "injury",
        "injured",
        "hurt",
        "physical violence",
        "criminal force"
    ]

    workplace_keywords = [
        "office",
        "workplace",
        "employer",
        "boss",
        "colleague",
        "sexual harassment",
        "harassed at work",
        "work harassment"
    ]

    if any(keyword in q for keyword in caste_keywords):
        topics.append("caste")

    if any(keyword in q for keyword in assault_keywords):
        topics.append("assault")

    if any(keyword in q for keyword in workplace_keywords):
        topics.append("workplace")

    return topics


def get_target_sources(topics):
    """Return legal sources that should receive targeted retrieval."""

    sources = []

    if "caste" in topics:
        sources.extend([
            "Constitution of India",
            "Protection of Civil Rights Act",
            "SC/ST Prevention of Atrocities Act"
        ])

    if "assault" in topics:
        sources.extend([
            "Bharatiya Nyaya Sanhita (BNS)",
            "Bharatiya Nagarik Suraksha Sanhita (BNSS)"
        ])

    if "workplace" in topics:
        sources.append("POSH Act")

    # Remove duplicates while preserving order
    return list(dict.fromkeys(sources))


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_relevant_context(
    query,
    query_embedding,
    n_general=8,
    n_targeted=8
):

    topics = detect_legal_topics(query)

    target_sources = get_target_sources(topics)

    retrieved = []

    # --------------------------------------------------------
    # GENERAL RETRIEVAL
    # --------------------------------------------------------

    general_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_general
    )

    if general_results.get("documents"):

        general_docs = general_results["documents"][0]
        general_metas = general_results["metadatas"][0]

        for doc, meta in zip(
            general_docs,
            general_metas
        ):
            retrieved.append((doc, meta))

    # --------------------------------------------------------
    # TARGETED RETRIEVAL
    # --------------------------------------------------------
    #
    # This is the important change.
    #
    # For a caste/untouchability query, we separately search:
    #   - Constitution
    #   - Protection of Civil Rights Act
    #   - SC/ST Prevention of Atrocities Act
    #
    # Therefore a general BNS result cannot crowd out the
    # caste-specific provisions.
    # --------------------------------------------------------

    for source in target_sources:

        try:

            targeted_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_targeted,
                where={
                    "source": source
                }
            )

            if not targeted_results.get("documents"):
                continue

            target_docs = targeted_results["documents"][0]
            target_metas = targeted_results["metadatas"][0]

            for doc, meta in zip(
                target_docs,
                target_metas
            ):
                retrieved.append((doc, meta))

        except Exception as e:

            print(
                f"Warning: targeted retrieval failed "
                f"for {source}: {e}"
            )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    unique = []
    seen = set()

    for doc, meta in retrieved:

        key = doc.strip()

        if key in seen:
            continue

        seen.add(key)
        unique.append((doc, meta))

    # --------------------------------------------------------
    # BUILD CONTEXT WITH SOURCE LABELS
    # --------------------------------------------------------

    context_parts = []

    for index, (doc, meta) in enumerate(unique):

        source = meta.get(
            "source",
            "Unknown source"
        )

        page = meta.get(
            "page",
            ""
        )

        if page:
            source_label = (
                f"{source}, page {page}"
            )
        else:
            source_label = source

        context_parts.append(
            f"[LEGAL SOURCE: {source_label}]\n"
            f"{doc}"
        )

    context = "\n\n".join(
        context_parts
    )

    return context, topics, target_sources


# ============================================================
# RAG ANSWER
# ============================================================

def rag_answer(query):

    # ========================================================
    # DEBUG MODE
    # ========================================================

    if DEBUG:

        legal_type = "General / Not Sure"

        answer = (
            "This is a test answer as of now."
        )

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

        print(convo_data)

    else:

        # ====================================================
        # EMBED USER QUERY LOCALLY
        # ====================================================

        print(
            "Creating query embedding..."
        )

        query_embedding = embedding_model.encode(
            [query]
        )[0]

        query_embedding = (
            query_embedding.tolist()
        )

        # ====================================================
        # RETRIEVE RELEVANT LEGAL DOCUMENTS
        # ====================================================

        print(
            "Searching legal documents..."
        )

        context, detected_topics, target_sources = (
            retrieve_relevant_context(
                query,
                query_embedding
            )
        )

        # ====================================================
        # PROMPT
        # ====================================================

        prompt = f"""
You are an empathetic legal aid assistant
for Indian citizens.

Your job is to provide simple, cautious,
source-grounded legal information.

You MUST use the legal documents provided
in the context below.

============================================================
IMPORTANT LEGAL RETRIEVAL INSTRUCTION
============================================================

The context has been retrieved in two ways:

1. General semantic retrieval.
2. Targeted retrieval from legal Acts that match
   the subject of the user's question.

Targeted legal sources are especially important.

If the question concerns caste, untouchability,
Scheduled Castes, Scheduled Tribes, caste discrimination,
caste abuse, denial of access, or similar issues,
carefully examine provisions from:

- Constitution of India
- Protection of Civil Rights Act
- SC/ST Prevention of Atrocities Act

Do NOT ignore these sources merely because BNS
assault provisions are also present.

If the question concerns physical assault AND caste
discrimination, discuss both aspects when supported
by the context.

============================================================
LEGAL SOURCES AVAILABLE
============================================================

- Constitution of India
- Bharatiya Nyaya Sanhita (BNS)
- Bharatiya Nagarik Suraksha Sanhita (BNSS)
- POSH Act
- Protection of Civil Rights Act
- SC/ST Prevention of Atrocities Act
- Other provided legal material

============================================================
RULES
============================================================

1. Only use information supported by the
provided legal document context.

2. Do NOT invent an Act, section, offence,
penalty, deadline, authority or procedure.

3. If the context contains a relevant provision,
USE IT.

4. For every important legal provision you mention,
give:
- Act name
- Section/article number, if clearly available
- What it generally covers
- Practical relevance to the user's situation

5. If multiple legal provisions are relevant,
mention the important ones.

6. If the context contains relevant caste or
untouchability provisions, you MUST discuss them.
Do not answer only with general assault provisions.

7. If the context contains relevant SC/ST provisions,
mention them when applicable.

8. Do not say "the documents do not contain enough
information" if relevant information is present
in the retrieved context.

9. Explain everything in simple language.

10. Assume the user has no legal background.

11. Never give definitive legal advice.

12. Never predict the outcome of a case.

13. If the situation involves immediate danger,
serious physical violence, death, sexual violence,
or another serious offence, clearly recommend
appropriate emergency/police/legal assistance.

14. Do not encourage hiding evidence, destroying
evidence, threatening anyone, retaliation, or
evading lawful investigation.

15. Keep the answer practical and concise.

16. Do not use markdown bold with **.

17. Do not unnecessarily repeat the user's question.

18. Do not assume facts that the user has not provided.

============================================================
IDENTITY RULES
============================================================

19. Never assume the user's gender, age, identity,
occupation, caste, religion, or relationship to the
situation unless explicitly stated.

20. Use neutral language such as:
"you", "the complainant",
"the affected person", or
"the person involved".

21. Do not address the user as "woman", "man",
"he", "she", "sir", or "madam" unless explicitly
provided.

22. The user may be asking on behalf of another person.

23. Do not assume the user is the person affected.

============================================================
LEGAL CATEGORY RULE
============================================================

Choose exactly ONE category from:

{types}

IMPORTANT CATEGORY PRIORITY:

If the user's question clearly involves
untouchability, caste discrimination, Scheduled
Caste/Tribe issues, caste abuse, caste-based
exclusion, or similar conduct, choose:

"Untouchability / Caste Discrimination"

even if physical assault is also mentioned.

If the question is only about assault/arrest/criminal
procedure and has no caste-discrimination aspect,
choose:

"Police / Arrest Issue"

============================================================
RETRIEVED LEGAL CONTEXT
============================================================

{context}

============================================================
DETECTED TOPICS
============================================================

{detected_topics}

Targeted legal sources searched:

{target_sources}

============================================================
RECENT CONVERSATION HISTORY
============================================================

{convo_history[-5000:]}

============================================================
USER QUESTION
============================================================

{query}

============================================================
OUTPUT REQUIREMENT
============================================================

Return ONLY a Python-style list containing
exactly two strings.

The first string must be the legal category.

The second string must be the answer.

The answer should:
- directly address the user's question
- identify relevant legal provisions from the context
- include Act names and section/article numbers when available
- explain what those provisions mean in simple language
- give practical next steps
- recommend professional/legal/emergency assistance when appropriate

Example:

["Untouchability / Caste Discrimination",
"Based on the provided legal material, ..."]

Do not add:
- Markdown code fences
- explanations outside the list
- "Here is your answer"
- extra fields
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
            if response.text
            else ""
        )

        print(
            f"\nQ: {query}"
        )

        # ====================================================
        # PARSE RESPONSE
        # ====================================================

        legal_type = "General / Not Sure"
        answer = response_text

        try:

            if response_text.startswith("```"):

                response_text = (
                    response_text
                    .replace("```json", "")
                    .replace("```python", "")
                    .replace("```", "")
                    .strip()
                )

            response_list = ast.literal_eval(
                response_text
            )

            if (
                isinstance(response_list, list)
                and len(response_list) >= 2
            ):

                legal_type = str(
                    response_list[0]
                ).strip()

                answer = str(
                    response_list[1]
                ).strip()

            else:

                raise ValueError(
                    "Invalid response format"
                )

        except Exception as e:

            print(
                "\nWarning: Gemini did not follow "
                "the expected format."
            )

            print(
                f"Parsing error: {e}"
            )

            # ------------------------------------------------
            # RECOVERY
            # ------------------------------------------------

            try:

                start = response_text.find("[")
                end = response_text.rfind("]")

                if (
                    start != -1
                    and end != -1
                    and end > start
                ):

                    cleaned_response = (
                        response_text[
                            start:end + 1
                        ]
                    )

                    response_list = ast.literal_eval(
                        cleaned_response
                    )

                    if (
                        isinstance(response_list, list)
                        and len(response_list) >= 2
                    ):

                        legal_type = str(
                            response_list[0]
                        ).strip()

                        answer = str(
                            response_list[1]
                        ).strip()

                    else:

                        raise ValueError(
                            "Recovered response is invalid"
                        )

                else:

                    raise ValueError(
                        "Could not find list in response"
                    )

            except Exception as recovery_error:

                print(
                    f"Recovery failed: "
                    f"{recovery_error}"
                )

                legal_type = (
                    "General / Not Sure"
                )

                answer = response_text

        # ====================================================
        # SAFETY NET FOR CATEGORY
        # ====================================================
        #
        # If the model still chooses Police / Arrest Issue
        # for an obvious caste/untouchability query, correct
        # ONLY the category. The legal answer remains generated
        # from the retrieved context.
        # ====================================================

        if "caste" in detected_topics:

            legal_type = (
                "Untouchability / Caste Discrimination"
            )

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