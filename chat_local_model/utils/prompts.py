from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    template="""
    You are an AI assistant providing structured and concise responses.

    Given the following context, answer the question in a simple and precise manner.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """,
    input_variables=["context", "question"]
)