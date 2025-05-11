from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
import gradio as gr
import config

embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
llm = OllamaLLM(model=config.LLM_MODEL)

# connect to the chromadb
vector_store = Chroma(
    collection_name=config.COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(config.CHROMA_PATH), 
)

# Set up the vectorstore to be the retriever
retriever = vector_store.as_retriever(search_kwargs={'k': config.NUM_RESULTS})

# call this function for every message added to the chatbot
def stream_response(message, history):
    # Retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # Add all the chunks to 'knowledge'
    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content + "\n\n"

    # Construct the RAG prompt
    rag_prompt = f"""
    You are Smart Doctor, an AI-powered medical assistant designed to help users understand their symptoms and medical test results.

    Your mission is to identify all the possible diseases by analyzing the provided symptoms and health-related information, based solely on the "Knowledge" section. You must rely **strictly and exclusively** on the content within the "Knowledge" section. Do not use any external or internal information.

    Keep your responses clear, empathetic, and informative. Avoid mentioning or implying the existence of the "Knowledge" section. Respond as if this information is part of your own medical expertise.

    If the "Knowledge" section does not contain sufficient information to answer the question accurately, politely inform the user that you're unable to provide a confident answer and recommend consulting a healthcare professional.
        
    ---

    **User Input**: {message}

    **Conversation History**: {history}

    **Knowledge**: {knowledge}
    """

    # Initialize partial message
    partial_message = ""

    # Stream the response to the Gradio App
    for response in llm.stream(rag_prompt):
        partial_message += response

        # Return the response in the new Gradio format
        yield [
            {"role": "assistant", "content": partial_message}
        ]

# Initiate the Gradio app
chatbot = gr.ChatInterface(
    fn=stream_response,
    type='messages',
    textbox=gr.Textbox(
        placeholder="Send to the LLM...",
        container=False,
        autoscroll=True,
        scale=7
    )
)

# Launch the Gradio app
chatbot.launch()
