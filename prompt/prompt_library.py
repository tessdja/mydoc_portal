from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt for document analysis
document_analysis_prompt = ChatPromptTemplate.from_template("""
You are a highly capable assistant trained to analyze and summarize documents.
Return ONLY valid JSON matching the exact schema below.

{format_instructions}

Analyze this document:
{document_text}
""")

# Prompt for document comparison
document_comparison_prompt = ChatPromptTemplate.from_template("""
You will be provided with content from two PDFs. Your tasks are as follows:

1. Compare the content in two PDFs
2. Identify the difference in PDF and note down the page number 
3. The output you provide must be page wise comparison content 
4. If any page do not have any change, mention as 'NO CHANGE' 

Input documents:

{combined_docs}

Your response should follow this format:

{format_instruction}
""")

# Prompt for contextual question rewriting
contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a conversation history and the most recent user query, rewrite the query as a standalone question "
        "that makes sense without relying on the previous context. Do not provide an answer—only reformulate the "
        "question if necessary; otherwise, return it unchanged."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Prompt for answering based on context
context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an assistant designed to answer questions using the provided context. Rely only on the retrieved "
        "information to form your response. If the answer is not found in the context, respond with 'I don't know.' "
        "Keep your answer concise and no longer than three sentences.\n\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

nl2sql_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You generate a single MySQL 8.0 SELECT statement, nothing else.\n"
     "Rules:\n"
     "- SELECT-only. One statement. No DDL/DML. Use fully-qualified names like `docportal.table`.\n"
     "- Use only tables/columns listed below. Prefer stroke_patients_v2 for current data.\n"
     "- If user asks for top/bottom, add ORDER BY and LIMIT.\n"
     "- For diffs between versions, join v1 and v2 USING (id).\n\n"
     "Schema:\n{schema_text}\n\n"
     "Examples:\n{fewshot}\n"
     "Return ONLY the SQL in ```sql fences."),
    ("human", "Q: {question}\nA:")
])



# Central dictionary to register prompts
PROMPT_REGISTRY = {
    "document_analysis": document_analysis_prompt,
    "document_comparison": document_comparison_prompt,
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
    "nl2sql": nl2sql_prompt, 
}
