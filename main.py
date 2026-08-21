import os
import json
import ast

import streamlit as st

from dotenv import load_dotenv
from google import genai

from sentence_transformers import SentenceTransformer

import chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

import os
from jinja2 import Environment, FileSystemLoader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import mm



base_dir = os.path.dirname(os.path.abspath(__file__))
env = Environment(loader=FileSystemLoader(base_dir))

template_police = env.get_template("police_comp_template.txt")
template_consumer = env.get_template("consumer_comp_template.txt")
template_rti = env.get_template("rti_template.txt")

rti_data = {
    "applicant_name": "Arka Saha",
    "address": "Cluster 11, IOCL Haldia Township, West Bengal, 721607",
    "department": "Municipal Corporation",
    "info_requested": "Status of road repair complaint filed on 01/08/2026, including expected completion date and officer assigned",
    "date": "21/08/2026"
}

consumer_data = {
    "complainant_name": "Arka Saha",
    "company_name": "Meow Electronics Pvt Ltd",
    "product_service": "Samsung Refrigerator (Model SR-450)",
    "purchase_date": "15/06/2026",
    "issue_description": "Refrigerator stopped cooling within 2 months of purchase. Company refused free repair despite valid warranty and has not responded to 3 follow-up calls.",
    "desired_resolution": "Full replacement of the product or complete refund of Rs. 32,000",
    "date": "21/08/2026"
}

police_data = {
    "complainant_name": "Arka Saha",
    "police_station": "Haldia Police Station, West Bengal",
    "incident_type": "Theft of two-wheeler",
    "incident_datetime": "20/08/2026, approximately 9:30 PM",
    "incident_location": "Parking area near Spencers supermart",
    "incident_description": "My motorcycle (Registration No. WB1234567) was stolen from the parking area while I was at a nearby shop for approximately 20 minutes.",
    "witnesses": "None identified at the time; CCTV footage may be available from nearby shops",
    "date": "21/08/2026"
}



# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Chatbot with Constitutional and Legal Awareness",
    layout="centered"
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# SETTINGS
# ============================================================

DEBUG = False

COLLECTION_NAME = "legal_docs_v2"


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
# LOAD RAG RESOURCES
#
# This function is cached.
#
# The following are loaded only once:
#
# 1. Embedding model
# 2. PDFs
# 3. Text chunks
# 4. ChromaDB
# 5. Document embeddings, if required
# ============================================================

@st.cache_resource
def load_rag_resources():

    print()
    print("======================================")
    print("Loading RAG resources...")
    print("======================================")
    print()

    # --------------------------------------------------------
    # EMBEDDING MODEL
    # --------------------------------------------------------

    print("Loading local embedding model...")

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Local embedding model loaded.")

    # --------------------------------------------------------
    # DOCUMENTS
    # --------------------------------------------------------

    documents = []

    def load_pdf(filename, source_name):

        if not os.path.exists(filename):

            print(
                f"WARNING: {filename} not found."
            )

            return

        print(
            f"Loading {filename}..."
        )

        loader = PyPDFLoader(
            filename
        )

        pages = loader.load()

        for page_number, page in enumerate(
            pages,
            start=1
        ):

            if page.page_content.strip():

                documents.append({

                    "text": page.page_content,

                    "source": source_name,

                    "page": page_number

                })

    # --------------------------------------------------------
    # SUMMARY.TXT
    # --------------------------------------------------------

    if os.path.exists(
        "summary.txt"
    ):

        print(
            "Loading summary.txt..."
        )

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

        print(
            "WARNING: summary.txt not found."
        )

    # --------------------------------------------------------
    # LEGAL PDFS
    # --------------------------------------------------------

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

    print(
        f"Loaded {len(documents)} document pages."
    )

    # --------------------------------------------------------
    # CHUNKING
    # --------------------------------------------------------

    print(
        "Creating document chunks..."
    )

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

            chunks.append(
                chunk
            )

            chunk_metadatas.append({

                "source": document["source"],

                "page": document["page"]

            })

    print(
        f"Total chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # CHROMADB
    # --------------------------------------------------------

    print(
        "Opening persistent ChromaDB..."
    )

    chroma_client = chromadb.PersistentClient(

        path="./chroma_db"

    )

    collection = chroma_client.get_or_create_collection(

        name=COLLECTION_NAME

    )

    # --------------------------------------------------------
    # CREATE DOCUMENT EMBEDDINGS
    # --------------------------------------------------------

    if collection.count() == 0:

        print(
            "ChromaDB collection is empty."
        )

        print(
            "Creating document embeddings..."
        )

        chunk_embeddings = embedding_model.encode(

            chunks,

            show_progress_bar=True

        )

        chunk_embeddings = (

            chunk_embeddings.tolist()

        )

        print(
            f"Embedded {len(chunk_embeddings)} chunks."
        )

        collection.add(

            documents=chunks,

            embeddings=chunk_embeddings,

            metadatas=chunk_metadatas,

            ids=[

                f"legal_chunk_{i}"

                for i in range(
                    len(chunks)
                )

            ]

        )

        print(
            f"Stored {collection.count()} chunks "
            "in ChromaDB."
        )

    else:

        print(
            "Existing ChromaDB found."
        )

        print(
            f"Loaded {collection.count()} chunks."
        )

    print()
    print("======================================")
    print("RAG resources loaded successfully.")
    print("======================================")
    print()

    return (

        embedding_model,

        collection

    )


# ============================================================
# LOAD RAG RESOURCES
# ============================================================

embedding_model, collection = (
    load_rag_resources()
)


# ============================================================
# QUERY ROUTING
# ============================================================

def detect_legal_topics(query):

    q = query.lower()

    topics = []

    # --------------------------------------------------------
    # CASTE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ASSAULT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # WORKPLACE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DETECT TOPICS
    # --------------------------------------------------------

    if any(

        keyword in q

        for keyword in caste_keywords

    ):

        topics.append(
            "caste"
        )

    if any(

        keyword in q

        for keyword in assault_keywords

    ):

        topics.append(
            "assault"
        )

    if any(

        keyword in q

        for keyword in workplace_keywords

    ):

        topics.append(
            "workplace"
        )

    return topics


# ============================================================
# TARGET SOURCES
# ============================================================

def get_target_sources(topics):

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

        sources.append(
            "POSH Act"
        )

    return list(
        dict.fromkeys(sources)
    )


# ============================================================
# RETRIEVE CONTEXT
# ============================================================

def retrieve_relevant_context(

    query,

    query_embedding,

    n_general=8,

    n_targeted=8

):

    topics = detect_legal_topics(
        query
    )

    target_sources = get_target_sources(
        topics
    )

    retrieved = []

    # --------------------------------------------------------
    # GENERAL RETRIEVAL
    # --------------------------------------------------------

    general_results = collection.query(

        query_embeddings=[

            query_embedding

        ],

        n_results=n_general

    )

    if general_results.get(
        "documents"
    ):

        general_docs = (

            general_results[
                "documents"
            ][0]

        )

        general_metas = (

            general_results[
                "metadatas"
            ][0]

        )

        for doc, meta in zip(

            general_docs,

            general_metas

        ):

            retrieved.append(

                (doc, meta)

            )

    # --------------------------------------------------------
    # TARGETED RETRIEVAL
    # --------------------------------------------------------

    for source in target_sources:

        try:

            targeted_results = collection.query(

                query_embeddings=[

                    query_embedding

                ],

                n_results=n_targeted,

                where={

                    "source": source

                }

            )

            if not targeted_results.get(
                "documents"
            ):

                continue

            target_docs = (

                targeted_results[
                    "documents"
                ][0]

            )

            target_metas = (

                targeted_results[
                    "metadatas"
                ][0]

            )

            for doc, meta in zip(

                target_docs,

                target_metas

            ):

                retrieved.append(

                    (doc, meta)

                )

        except Exception as e:

            print(

                f"Warning: targeted retrieval "
                f"failed for {source}: {e}"

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

        unique.append(

            (doc, meta)

        )

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for doc, meta in unique:

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

    return (

        context,

        topics,

        target_sources

    )


# ============================================================
# RAG ANSWER
# ============================================================

def rag_answer(query):

    # --------------------------------------------------------
    # QUERY EMBEDDING
    # --------------------------------------------------------

    print(
        "\nCreating query embedding..."
    )

    query_embedding = embedding_model.encode(

        [query]

    )[0]

    query_embedding = (

        query_embedding.tolist()

    )

    # --------------------------------------------------------
    # RETRIEVE CONTEXT
    # --------------------------------------------------------

    print(
        "Searching legal documents..."
    )

    (

        context,

        detected_topics,

        target_sources

    ) = retrieve_relevant_context(

        query,

        query_embedding

    )

    # --------------------------------------------------------
    # CONVERSATION HISTORY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""

You are an EMPATHETIC legal aid assistant
for Indian citizens.

If the QUERY is in a different language, you need to convert it to English first, and then answer in the same language as the QUERY.

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

7. If the context contains relevant SC/ST provisions,
mention them when applicable.

8. Explain everything in simple language.

9. Assume the user has no legal background.

10. Never give definitive legal advice.

11. Never predict the outcome of a case.

12. If the situation involves immediate danger,
serious physical violence, death, sexual violence,
or another serious offence, clearly recommend
appropriate emergency/police/legal assistance.

13. Do not encourage hiding evidence, destroying
evidence, threatening anyone, retaliation, or
evading lawful investigation.

14. Keep the answer practical and concise.

15. Do not use markdown bold with **.

16. Do not unnecessarily repeat the user's question.

17. Do not assume facts that the user has not provided.

18. You are an EMPATHETIC legal aid assistant, not a judge or lawyer. Someimtes, when needed, show empathy and understanding of the user's situation, in a formal and preciese phrase.

19. Make sure your answers are not too long, because the user may not read long answers. If the answer is long, break it into sections with clear headings.

20. ANYTHING outside this legal and constituinoal things are to be bluntly ignored. Do not answer anything outside the legal and constitutional context. Simply add a short phrase to ignore and avoid such queries.

============================================================
IDENTITY RULES
============================================================

18. Never assume the user's gender, age, identity,
occupation, caste, religion, or relationship to the
situation unless explicitly stated.

19. Use neutral language such as:

"you"
"the complainant"
"the affected person"
"the person involved"

20. Do not address the user as "woman", "man",
"he", "she", "sir", or "madam" unless explicitly
provided.

21. The user may be asking on behalf of another person.

22. Do not assume the user is the person affected.

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

["type of legal issue",
"your generated answer"]

Do not add:

- Markdown code fences
- explanations outside the list
- "Here is your answer"
- extra fields

"""

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    print(
        "Generating legal response..."
    )

    response = client.models.generate_content(

        model="gemini-3.6-flash",

        contents=prompt

    )

    response_text = (

        response.text.strip()

        if response.text

        else ""

    )

    # --------------------------------------------------------
    # PARSE RESPONSE
    # --------------------------------------------------------

    legal_type = (
        "General / Not Sure"
    )

    answer = response_text

    try:

        if response_text.startswith(
            "```"
        ):

            response_text = (

                response_text

                .replace(
                    "```json",
                    ""
                )

                .replace(
                    "```python",
                    ""
                )

                .replace(
                    "```",
                    ""
                )

                .strip()

            )

        response_list = ast.literal_eval(
            response_text
        )

        if (

            isinstance(
                response_list,
                list
            )

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
            "Warning: Gemini did not follow "
            "the expected format."
        )

        print(
            f"Parsing error: {e}"
        )

        # ----------------------------------------------------
        # RECOVERY
        # ----------------------------------------------------

        try:

            start = response_text.find(
                "["
            )

            end = response_text.rfind(
                "]"
            )

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

                response_list = (

                    ast.literal_eval(

                        cleaned_response

                    )

                )

                if (

                    isinstance(
                        response_list,
                        list
                    )

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

    # --------------------------------------------------------
    # CATEGORY SAFETY NET
    # --------------------------------------------------------

    if "caste" in detected_topics:

        legal_type = (
            "Untouchability / Caste Discrimination"
        )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"\nType: {legal_type}"
    )

    print(
        f"\nAnswer:\n{answer}"
    )

    # --------------------------------------------------------
    # SAVE CONVERSATION
    # --------------------------------------------------------

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

    with open(

        "conversations.txt",

        "a+",

        encoding="utf-8"

    ) as file:

        file.write(
            convo_data
        )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "type": legal_type,

        "answer": answer

    }


# ============================================================
# CREATE HARDCODED PDF
# ============================================================

def create_hardcoded_pdf(t):

    filename = f"generate_{t}_doc.pdf"

    # c = canvas.Canvas(

    #     filename,

    #     pagesize=A4

    # )

    # width, height = A4

    # # --------------------------------------------------------
    # # TITLE
    # # --------------------------------------------------------

    # c.setFont(

    #     "Helvetica-Bold",

    #     20

    # )

    # c.drawString(

    #     50,

    #     height - 60,

    #     "AI Legal Assistant"

    # )

    # # --------------------------------------------------------
    # # CONTENT
    # # --------------------------------------------------------

    # c.setFont(

    #     "Helvetica",

    #     12

    # )

    # lines = [

    #     "This is a hardcoded PDF response.",

    #     "",

    #     "The user requested image creation.",

    #     "Actual image generation will be connected",

    #     "to this part of the application later.",

    #     "",

    #     "For now, this PDF is only a placeholder."

    # ]

    # y = height - 110

    # for line in lines:

    #     c.drawString(

    #         50,

    #         y,

    #         line

    #     )

    #     y -= 22

    # # --------------------------------------------------------
    # # SAVE
    # # --------------------------------------------------------

    # c.save()

    # # --------------------------------------------------------
    # # READ PDF INTO MEMORY
    # # --------------------------------------------------------

    if t == "police":
        # template = env.get_template("police_comp_template.txt")
        filled_text = template_police.render(**police_data)

    if t == "rti": 
            # template = env.get_template("rti_comp_template.txt")
            filled_text = template_rti.render(**rti_data)
    if t == "consumer": 
            # template = env.get_template("consumer_comp_template.txt")
            filled_text = template_consumer.render(**consumer_data)

    output_path = f"output_files/{filename}.pdf"
    # c = canvas.Canvas(output_path, pagesize=A4)
    # width, height = A4
    # y = height - 60
    # for line in filled_text.split("\n"):
    #     c.drawString(50, y, line)
    #     y -= 18
    #     if y < 50:
    #         c.showPage()
    #         y = height - 60
    # c.save()



    ############

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm
    )

    styles = getSampleStyleSheet()

    body_style = styles["BodyText"]

    body_style.fontName = "Helvetica"
    body_style.fontSize = 11
    body_style.leading = 16
    body_style.spaceAfter = 6

    story = []

    for line in filled_text.split("\n"):

        line = line.strip()

        # Preserve blank lines
        if not line:
            story.append(
                Spacer(1, 6)
            )
            continue

        # Escape characters that Paragraph interprets as HTML
        line = (
            line
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        # Paragraph automatically wraps long text
        story.append(
            Paragraph(
                line,
                body_style
            )
        )

    doc.build(story)


# output_path = os.path.join(base_dir, f"output_files/police_{police_data['complainant_name']}.pdf")
# text_to_pdf(filled_text, output_path)


    ########











    with open(

        output_path,

        "rb"

    ) as file:

        pdf_bytes = file.read()

    return pdf_bytes


# ============================================================
# DISPLAY PDF CARD
# ============================================================

def display_pdf_card(pdf_bytes, key_suffix):

    # Small PDF icon
    st.write("📄")

    # Filename
    st.write("**generate_document.pdf**")

    # One-line description
    st.caption("Your generated PDF document is ready.")

    # Download button
    st.download_button(
        label="Download PDF",
        data=pdf_bytes,
        file_name="generated_answer.pdf",
        mime="application/pdf",
        key=f"download_pdf_{key_suffix}"
    )

# ============================================================
# STREAMLIT HEADER
# ============================================================

st.title(
    "Chatbot with Constitutional and Legal Awareness"
)

st.caption(
    "Ask a question related to your legal rights and constitutional protections."
)


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for index, message in enumerate(
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )

        # ----------------------------------------------------
        # DISPLAY PDF FOR OLD MESSAGE
        # ----------------------------------------------------

        if (

            message["role"] == "assistant"

            and message.get("pdf")

        ):

            pdf_bytes = (
                message.get("pdf_bytes")
            )

            if pdf_bytes:

                display_pdf_card(

                    pdf_bytes,

                    f"history_{index}"

                )


# ============================================================
# USER INPUT
# ============================================================

user_input = st.chat_input(
    "Ask your legal question..."
)


# ============================================================
# PROCESS USER INPUT
# ============================================================

if user_input:

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": user_input

    })

    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.write(
            user_input
        )

    # ========================================================
    # CREATE IMAGE → PDF
    # ========================================================

    if "create image" in user_input.lower():

        # ----------------------------------------------------
        # GENERATE PDF
        # ---------------------------------------------------

        if "police" in user_input.lower():

            # pdf_bytes = create_police_pdf
            pdf_bytes = create_hardcoded_pdf("police")

            
        if "rti" in user_input.lower():

            # pdf_bytes = create_police_pdf
            pdf_bytes = create_hardcoded_pdf("rti")
            
        if "consumer" in user_input.lower():

            # pdf_bytes = create_police_pdf
            pdf_bytes = create_hardcoded_pdf()
        # ----------------------------------------------------
        # SAVE PDF IN SESSION
        # ----------------------------------------------------

        st.session_state.generated_pdf = (
            pdf_bytes
        )

        # ----------------------------------------------------
        # ASSISTANT RESPONSE
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            st.write(
                "Your document is ready."
            )

            display_pdf_card(

                pdf_bytes,

                f"current_{len(st.session_state.messages)}"

            )

        # ----------------------------------------------------
        # SAVE ASSISTANT MESSAGE
        # ----------------------------------------------------

        st.session_state.messages.append({

            "role": "assistant",

            "content": "Your PDF is ready.",

            "pdf": True,

            "pdf_bytes": pdf_bytes

        })

    # ========================================================
    # NORMAL RAG QUESTION
    # ========================================================

    else:

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Searching legal documents..."
            ):

                result = rag_answer(
                    user_input
                )

            st.write(
                result["answer"]
            )

        # ----------------------------------------------------
        # SAVE ASSISTANT MESSAGE
        # ----------------------------------------------------

        st.session_state.messages.append({

            "role": "assistant",

            "content": result["answer"]

        })