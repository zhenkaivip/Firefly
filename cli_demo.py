from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from transformers import TextIteratorStreamer
from threading import Thread

device = 'cuda'
model_name = 'baichuan-inc/baichuan-7B'
adapter_name = 'YeungNLP/firefly-baichuan-7b-qlora-sft'
max_new_tokens = 500
top_p = 0.9
temperature = 0.35
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

history = []
history_len = 3

while True:

    def predict_and_print(query, history: list) -> list:
        history.append((query, None))
        all_input = '<s>'
        for q, a in history[-history_len - 1:]:
            all_input += f'{q}</s>'
            if a: all_input += f'{a}</s>'
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
        print("Firefly：", end="", flush=True)
        for new_text in streamer:
            print(new_text, end="", flush=True)
            response += new_text
        print()

        history[-1][1] = response
        return history


    user_input = input('User：')
    history = predict_and_print(query=user_input, history=history)
