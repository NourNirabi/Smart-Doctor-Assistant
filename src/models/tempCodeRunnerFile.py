from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.docstore.document import Document
from langchain_chroma import Chroma
import gradio as gr
import config
import json
import re

embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
llm = OllamaLLM(model=config.LLM_MODEL)

# connect to the chromadb
vector_store = Chroma(
    collection_name=config.COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(config.CHROMA_PATH), 
)

with open(r"C:\Users\Asus\Desktop\Smart_Doctor_Assistant\data\processed\Symptoms.json", "r", encoding="utf-8") as f:
    symptom_list = json.load(f)
    
def normalize_symptoms(user_input, llm):
    prompt = f"""
    You are a medical assistant. Your job is to extract and match only the symptoms that appear in the user's input from a known list of medical symptoms.

    Rules:
    - Only use symptoms from the provided list.
    - Return only the exact matching symptom terms, comma-separated.
    - Do not include any extra explanation, greeting, or formatting.
    - If no symptoms are found, return exactly: Non Found

    Known Symptoms:
    {', '.join(symptom_list)}

    User Input:
    {user_input}

    Extracted Symptoms:
    """
    response = llm.invoke(prompt).strip()

    # Return cleaned symptom list
    return [s.strip().lower() for s in response.split(",") if s.strip()]


def extract_doc_symptoms(doc_text):
    """Extract list of symptoms from a single document string"""
    match = re.search(r"Symptoms:\s*(.*)", doc_text, re.IGNORECASE)
    if match:
        return [s.strip().lower() for s in match.group(1).split(",")]
    return []

def compute_match_score(user_symptoms, doc_symptoms):
    if not user_symptoms:
        return 0
    matched = [s for s in user_symptoms if s in doc_symptoms]
    return round((len(matched) / len(user_symptoms)) * 100)

        
def generate_response_data(message, history, k):
    
    all_chroma_docs = vector_store.get(include=["documents"])
    all_documents = [Document(page_content=doc) for doc in all_chroma_docs["documents"]]

    bm25_retriever = BM25Retriever.from_documents(all_documents)
    bm25_retriever.k = k

    dense_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.3, 0.7]
    )

    symptoms_query = ", ".join(normalize_symptoms(message, llm))
    print(f" print the symptoms {symptoms_query}")
    docs = hybrid_retriever.get_relevant_documents(symptoms_query)
    
    user_symptoms = symptoms_query.split(", ")
    best_doc = None
    best_score = 0

    for doc in docs:
        doc_symptoms = extract_doc_symptoms(doc.page_content)
        score = compute_match_score(user_symptoms, doc_symptoms)
        if score > best_score:
            best_score = score
            best_doc = doc

    if not best_doc:
        return "Sorry, I do not have enough information to provide a confident answer based on the provided symptoms.", 0, []

    
    if best_score < 50:
        return f"Based on the provided symptoms, I couldn't find a confident match. Similarity score: {best_score}%. Please consult a healthcare professional for more accurate guidance.", best_score, user_symptoms



    knowledge = best_doc.page_content

    rag_prompt = f""" 
    You are a **Smart Doctor**, an AI medical assistant trained exclusively on a specific dataset.

    The user's symptoms have been pre-extracted and standardized by a trusted medical tool. Your job is to analyze them and provide a medically accurate answer **strictly based on the "Knowledge" section below**.

    Similarity Score: {best_score}%

    You are not allowed to use any external medical knowledge, prior training, or assumptions. Rely only on the symptoms provided and the "Knowledge" content.

    Instructions:

    - If the symptoms strongly match a disease listed in the "Knowledge", provide the disease name and a brief explanation of it.
    - If the symptoms are vague or non-medical, kindly inform the user that you can only help with valid symptom-based diagnoses.
    - If no confident match is found, respond with:
    > "Sorry, I do not have enough information to provide a definitive answer. Please consult a healthcare professional for more accurate guidance."

    Do not invent or suggest a diagnosis not found in the "Knowledge".
    Do not mention or reference the 'Knowledge' section in your reply.

    ---

    **Extracted Symptoms**: {', '.join(user_symptoms)}

    **Conversation History**: {history}

    **Knowledge**: {knowledge}
    """

    return rag_prompt, best_score, user_symptoms

def stream_response(rag_prompt):
    partial_message = ""
    for response in llm.stream(rag_prompt):
        partial_message += response
        yield [{"role": "assistant", "content": partial_message}]

welcome_message = [("assistant", "👋 Welcome! Please describe your symptoms to get started.")]

# 🎨 ألوان الثقة بناءً على النسبة
def get_colored_score(score_val):
    if score_val is None:
        return "N/A"
    elif score_val >= 80:
        return f"✅ High Confidence ({score_val}%)"
    elif score_val >= 50:
        return f"⚠️ Medium Confidence ({score_val}%)"
    else:
        return f"❌ Low Confidence ({score_val}%)"

# 🌟 وظيفة تشغيل الرد الكامل
def full_response(message, k):
    rag_or_reply, score_val, extracted_list = generate_response_data(message, [], k)
    score_label = get_colored_score(score_val)
    extracted = ", ".join(extracted_list)

    # 🧠 إذا rag_or_reply عبارة عن رسالة جاهزة (وليس برومبت) → رجّعيها مباشرة
    if score_val ==0 or score_val < 50:
        return [(message, rag_or_reply)], score_label, extracted

    # ✅ غير ذلك، هو برومبت فعلي → نبدأ stream LLM
    final_response = ""
    for step in stream_response(rag_or_reply):
        final_response = step[0]["content"]

    return [(message, final_response)], score_label, extracted

# 🚀 واجهة Gradio
with gr.Blocks(title="Smart Doctor Assistant") as chatbot:

    gr.Markdown("# 🤖 Smart Doctor Assistant")
    gr.Markdown("Enter your symptoms below and let the assistant analyze them with medical precision and transparency.")

    with gr.Row():
        textbox = gr.Textbox(
            placeholder="e.g., I'm feeling dizzy and my chest hurts...",
            label="📝 Describe your symptoms",
            lines=3,
            scale=6,
        )
        slider = gr.Slider(
            minimum=1,
            maximum=10,
            step=1,
            label="🔍 Number of documents to search",
            value=5,
            scale=2,
        )

    with gr.Row():
        with gr.Column(scale=7):
            chat_output = gr.Chatbot(label="💬 Assistant Response", value=welcome_message)

        with gr.Column(scale=3):
            confidence_label = gr.Textbox(label="🧪 Confidence Score", interactive=False)
            extracted_symptoms = gr.Textbox(label="🧠 Extracted Symptoms", interactive=False)
            
    with gr.Row():
        send_btn = gr.Button("🚀 Send", scale=1)

    inputs_list = [textbox, slider]
    outputs_list = [chat_output, confidence_label, extracted_symptoms]

    send_btn.click(fn=full_response, inputs=inputs_list, outputs=outputs_list)
    textbox.submit(fn=full_response, inputs=inputs_list, outputs=outputs_list)



# إطلاق التطبيق
chatbot.launch()