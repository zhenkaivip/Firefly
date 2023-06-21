import gradio as gr

from transformers.utils.versions import require_version

require_version("gradio>=3.30.0", "To fix: pip install gradio>=3.30.0")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from transformers import TextIteratorStreamer
from threading import Thread

device = 'cuda'
model_name = 'baichuan-inc/baichuan-7B'
adapter_name = 'YeungNLP/firefly-baichuan-7b-qlora-sft'
repetition_penalty = 1.0
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map='auto'
)
model = PeftModel.from_pretrained(model, adapter_name)
model.eval()
model = model.to(device)

history_len = 3


def predict(query, chatbot, max_new_tokens, top_p, temperature, history):
    print(query)
    print(history)
    chatbot.append((query, ""))
    all_input = '<s>'
    for q, a in history[-history_len:] + [(query, None)]:
        all_input += f'{q}</s>'
        if a: all_input += f'{a}</s>'
    print(all_input)
    model_input_ids = tokenizer(all_input, return_tensors="pt").input_ids
    model_input_ids = model_input_ids.to(device)

    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = {}
    gen_kwargs["input_ids"] = model_input_ids
    gen_kwargs["max_new_tokens"] = max_new_tokens
    gen_kwargs["do_sample"] = True
    gen_kwargs["top_p"] = top_p
    gen_kwargs["temperature"] = temperature
    gen_kwargs["repetition_penalty"] = repetition_penalty
    gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
    gen_kwargs["streamer"] = streamer

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    response = ""
    for new_text in streamer:
        response += new_text
        new_history = history + [(query, response)]
        chatbot[-1] = (query, response)
        yield chatbot, new_history


def reset_user_input():
    return gr.update(value="")


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""
    <h1 align="center">
        <a href="https://huggingface.co/YeungNLP/firefly-baichuan-7b-qlora-sft" target="_blank">
            firefly-baichuan-7b-qlora-sft
        </a>
    </h1>
    """)

    chatbot = gr.Chatbot()

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")

        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_new_tokens = gr.Slider(0, 1024, value=500, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.9, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1.5, value=1.00, step=0.01, label="Temperature",
                                    interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_new_tokens, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(server_name="0.0.0.0", share=True, inbrowser=False, server_port=10002)
