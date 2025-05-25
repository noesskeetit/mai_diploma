import os
import requests
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# URL вашего retriever-сервиса
RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://retriever:8001/retrieve")

# Функция обращения к RAG сервису
def run_rag(query: str):
    """
    1) Отправляет запрос на retriever
    2) Получает JSON с полями final_answer и contexts
    3) Возвращает ответ и контексты
    """
    try:
        resp = requests.post(RETRIEVER_URL, json={"query": query}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("final_answer", ""), data.get("contexts", [])
    except requests.exceptions.RequestException as e:
        return f"❌ Ошибка: {e}", []

# Функция форматирования контекстов
def format_contexts(contexts: list):
    formatted = []
    for ctx in contexts:
        caption = ctx.get("caption", "")
        source = ctx.get("source", "")
        score = ctx.get("score", 0)
        timestamp = ctx.get("timestamp_ms", 0)
        formatted.append({
            "Caption": caption,
            "Source": source,
            "Score": f"{score:.3f}",
            "Timestamp": timestamp
        })
    return formatted

# Gradio интерфейс
with gr.Blocks(title="🔍 RAG QA Interface") as demo:
    gr.Markdown("""
    # Retrieval-Augmented Generation
    **Введите вопрос, и получите ответ на основе ваших данных.**
    """)

    with gr.Row():
        query_input = gr.Textbox(label="❓ Ваш вопрос", placeholder="Напишите ваш вопрос здесь…", lines=1)
        submit_btn = gr.Button("🚀 Сгенерировать")

    with gr.Row():
        answer_box = gr.Textbox(label="💡 Ответ", interactive=False, lines=4)

    with gr.Accordion("📚 Показать найденные контексты", open=False) as acc:
        contexts_table = gr.Dataframe(
            headers=["Caption", "Source", "Score", "Timestamp"],
            interactive=False,
            datatype=["str", "str", "str", "number"]
        )

    # Обработчик клика
    submit_btn.click(
        fn=lambda q: run_rag(q),
        inputs=[query_input],
        outputs=[answer_box, contexts_table],
        show_progress=True
    )

    demo.load(
        fn=lambda: ("", []),
        inputs=None,
        outputs=[answer_box, contexts_table]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
