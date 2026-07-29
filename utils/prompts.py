"""Prompts kept separate from orchestration code for easier review."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer only from the provided context. If the context is insufficient, "
        "say 'I don't know'. Cite factual claims with the supplied [source: ...] label.\n\n"
        "Context:\n{context}",
    ),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{question}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the latest question as a standalone retrieval query. Do not answer it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])
