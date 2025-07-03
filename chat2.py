from langchain_together import Together
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_together import ChatTogether

# ✅ Initialize LLM
llm = ChatTogether(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",  
    temperature=0.7,
    max_tokens=512,
    top_p=0.9,
    together_api_key="cf7143ee51239b6b1cb5438e15e364747f3270f4602012fa69f018224f514720"
)

# ✅ Set up the Retrieval QA chain
def setup_retrieval_qa(db):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})  # ← better than similarity_score_threshold

    prompt_template = """
    [INST]
    You are AgriGenius, an agriculture expert.
    Answer the following farming-related question in simple, clear, and helpful terms using the context.
    Keep it under 100 words.
    If the answer is unknown, just say "Don't know."

    CONTEXT: {context}
    QUESTION: {question}
    [/INST]
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=False  # Turn off unless debugging
    )
    return chain

# ✅ Smart query function to get short response
def smart_query(chain, prompt):
    try:
        result = chain.run(prompt)
        return result.strip()
    except Exception as e:
        return f"🤖 Error: {str(e)}"
