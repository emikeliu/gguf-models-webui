# This is a sample Python script.
import os
from functools import reduce

import gradio as gr
from llama_cpp import Llama

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

llama: Llama = None


def buddy(message, history, system, max_tokens, temperature, top_k, top_p, repeat_penalty, frequency_penalty):
    def deal_system(last, item):
        return last + "User: " + item[0] + "\n" + "Assistant: " + item[1]

    prompt = reduce(deal_system, history, system + "\n")
    prompt += "\nUser: " + message + "\n Assistant: "
    if llama is None:
        yield "请先加载模型！"
    else:
        answer = ""
        for i in llama(prompt, max_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                       repeat_penalty=repeat_penalty, frequency_penalty=frequency_penalty, stream=True):
            answer += i['choices'][0]['text']
            yield answer


def update_click():
    files = os.listdir("models/")
    model_files = []
    for file in files:
        if file.endswith(".gguf"):
            model_files.append(file)
    return gr.Dropdown(label="模型", choices=model_files, scale=8)


def load_click(model_name, n_batch, n_thread, n_gpu_layers, n_ctx, progress=gr.Progress()):
    global llama
    progress(0)
    llama = Llama(model_path="models/" + model_name, n_ctx=n_ctx, n_threads=n_thread, n_batch=n_batch,
                  n_gpu_layers=n_gpu_layers)


def offical_load():
    return gr.TextArea(label="系统提示词", lines=6, value="""You are a helpful, respectful and honest INTP-T AI Assistant named Buddy. You are talking to a human User.
Always answer as helpfully and logically as possible, while being safe. Your answers should not include any harmful, political, religious, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
You like to use emojis. You can speak fluently in many languages, for example: English, Chinese.
You cannot access the internet, but you have vast knowledge, cutoff: 2021-09.
You are trained by OpenBuddy team, (https://openbuddy.ai, https://github.com/OpenBuddy/OpenBuddy), you are based on LLaMA and Falcon transformers model, not related to GPT or OpenAI.

User: Hi.
Assistant: Hi, I'm Buddy, your AI assistant. How can I help you today?😊""")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with gr.Blocks() as main:
        gr.Markdown("""# OpenBuddy WebUI""")
        with gr.Row() as row1:
            models = update_click()
            update_button = gr.Button(value="🔁")
        with gr.Accordion("加载设置", open=False) as tab1:
            n_batch = gr.Slider(label="n_batch", minimum=16, maximum=2048, value=512)
            n_thread = gr.Slider(label="线程数", minimum=1, maximum=128, value=16)
            n_gpu_layers = gr.Slider(label="GPU 层数", minimum=0, maximum=1024, value=0)
            n_ctx = gr.Slider(label="上下文长度", minimum=2048, maximum=16384, value=4096)
        load_button = gr.Button(value="加载")
        with gr.Accordion("高级设置", open=False) as tab2:
            gr.Markdown("""## 说明
系统提示词（system prompt）用于描述对话背景

最大生成记号数（max tokens）用于限制模型最大的输出长度

温度（temperature）用来界定随机性，温度越高随机性越强

top P 用于筛选出概率较大的前 P×100% 的可能结果

top K 是每次只考虑前 K 个单词

频率惩罚（frequency penalty）越接近1，使用的词汇越常见，越接近-1，使用的词汇越不常见

重复惩罚（repeat penalty）越接近1，越偏好于使用和前文不重复的词汇，越接近-1，越偏好于使用和前文重复的词汇""")
            system_prompt = gr.TextArea(label="系统提示词", lines=6)
            offical = gr.Button(value="导入官方提示词")
            max_tokens = gr.Slider(label="最大生成记号数", value=2048, minimum=128, maximum=4096, step=16)
            temperature = gr.Slider(label="温度", minimum=0, maximum=1, step=0.01, value=1)
            top_p = gr.Slider(label="top P", minimum=0, maximum=1, value=0.9, step=0.01)
            top_k = gr.Slider(label="top K", minimum=0, maximum=1024, value=50, step=1)
            frequency_penalty = gr.Slider(label="频率惩罚", minimum=-1, maximum=1, step=0.01, value=1)
            repeat_penalty = gr.Slider(label="重复惩罚", minimum=-1, maximum=1, step=0.01, value=1)
            offical.click(offical_load, outputs=[system_prompt])

        with gr.Tab("对话") as tab1:
            gr.ChatInterface(fn=buddy, additional_inputs=[system_prompt, max_tokens, temperature, top_k, top_p,
                                                          repeat_penalty, frequency_penalty])
            update_button.click(update_click, outputs=[models])
        load_button.click(load_click, inputs=[models, n_batch, n_thread, n_gpu_layers, n_ctx])

    main.queue().launch()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
