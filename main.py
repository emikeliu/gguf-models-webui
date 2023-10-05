# This is a sample Python script.
import os
from functools import reduce

import gradio as gr
from llama_cpp import Llama

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

llama: Llama


def buddy(message, history, system):
    def deal_system(last, item):
        return last + "User: " + item[0] + "\n" + "Assistant: " + item[1]

    prompt = reduce(deal_system, history, system + "\n")
    prompt += "\nUser: " + message + "\n Assistant: "
    return llama(prompt)


def update_click():
    files = os.listdir("models/")
    model_files = []
    for file in files:
        if file.endswith(".gguf"):
            model_files.append(file)
    return gr.Dropdown(label="模型", choices=model_files, scale=8)


def load_click(model_name):
    global llama
    llama = Llama(model_path="models/" + model_name, n_ctx=4096, n_threads=8)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with gr.Blocks() as main:
        gr.Markdown("""# OpenBuddy WebUI""")
        with gr.Row() as row1:
            models = update_click()
            update_button = gr.Button(value="🔁")
        load_button = gr.Button(value="加载")
        with gr.Accordion("高级设置", open=False) as tab2:
            system_prompt = gr.TextArea(label="系统提示词", lines=6)
        with gr.Tab("对话", id=1) as tab1:
            gr.ChatInterface(fn=buddy, additional_inputs=[system_prompt])
            update_button.click(update_click, outputs=[models])
            load_button.click(load_click, inputs=[models])


    main.launch()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
