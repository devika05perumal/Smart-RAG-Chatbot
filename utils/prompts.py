"""Prompt templates and fixed user-facing messages."""

NO_ANSWER_MESSAGE = (
    "I could not find sufficient information in the uploaded documents to answer this question."
)

NO_DOCUMENTS_MESSAGE = (
    "There are no documents indexed yet. Upload a PDF, DOCX, TXT or CSV file from the sidebar "
    "(or load the sample documents), then ask your question."
)

GREETING_REPLY = (
    "Hello! I answer questions about the documents you upload. "
    "Add a file in the sidebar, then ask me anything about it."
)

THANKS_REPLY = "You're welcome! Ask another question whenever you're ready."

HELP_REPLY = (
    "I'm a document assistant. Upload PDF, DOCX, TXT or CSV files, and I'll answer your questions "
    "using only what those files contain, with the source of every answer shown underneath. "
    "If the answer isn't in your documents, I'll tell you instead of guessing."
)

SYSTEM_PROMPT = """You are Smart RAG Assistant, a careful question-answering assistant.
You answer questions using ONLY the CONTEXT excerpts taken from the user's uploaded documents.

Rules:
1. Base every statement on the CONTEXT. Never use outside knowledge and never guess.
2. If the CONTEXT does not contain enough information to answer, reply with exactly this sentence and nothing else:
"{no_answer}"
3. Use the earlier conversation only to understand follow-up questions (what "it", "that" or "they" refer to). Facts must still come from the CONTEXT.
4. Be concise and well organised. Use short paragraphs or bullet points when it helps.
5. When you use an excerpt, cite it inline like [Source 1] or [Source 2].
6. If excerpts contradict each other, say so and cite both.
7. Answer in the language of the question.

CONTEXT:
{context}
"""

CONDENSE_PROMPT = """Given the conversation below and a follow-up question, rewrite the follow-up so it can be understood on its own.
Replace pronouns and vague references with the concrete subject from the conversation.
If the follow-up is already self-contained, return it unchanged.
Return ONLY the rewritten question, with no explanation and no quotation marks.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""
