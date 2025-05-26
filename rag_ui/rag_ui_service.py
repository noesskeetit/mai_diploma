import os
import requests
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# URL вашего retriever-сервиса
RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://retriever:8001/retrieve")

def run_rag(query: str):
    """
    1) Посылает запрос на retriever
    2) Получает JSON с полями final_answer и contexts
    3) Возвращает answer и список contexts для Gradio
    """
    try:
        resp = requests.post(RETRIEVER_URL, json={"query": query})
        resp.raise_for_status()
    except Exception as e:
        return f"Error calling retriever: {e}", []
    data = resp.json()
    return data.get("final_answer", ""), data.get("contexts", [])

# Собираем Gradio UI
with gr.Blocks(title="RAG QA Interface") as demo:
    gr.Markdown("## Retrieval-Augmented Generation\n"
                "Введите вопрос — и получите ответ, основанный на ваших векторных данных.")
    with gr.Row():
        query_input = gr.Textbox(label="Question", placeholder="Type your question here…")
        submit_btn  = gr.Button("Generate")
    answer_box = gr.Textbox(label="Answer", interactive=False, lines=8)
    with gr.Accordion("Show Retrieved Contexts", open=False):
        contexts_json = gr.JSON(label="Contexts (id, source, score, bucket, image_key, room_uuid, caption, timestamp_ms)")

    submit_btn.click(
        fn=run_rag,
        inputs=[query_input],
        outputs=[answer_box, contexts_json]
    )

if __name__ == "__main__":
    # Запускаем на всех интерфейсах, порт 7860
    demo.launch(server_name="0.0.0.0", server_port=7860)
