from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

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
history_len = 1
user_input = input('User：')
while True:
    query = user_input
    history.append([query, ""])
    all_input = '<s>'
    for q, a in history[-history_len - 1:]:
        all_input += f'{q}</s>'
        if a: all_input += f'{a}</s>'
    print(all_input)
    model_input_ids = tokenizer(all_input, return_tensors="pt").input_ids
    model_input_ids = model_input_ids.to(device)
    outputs = model.generate(
        input_ids=model_input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_p=top_p,
        temperature=temperature, repetition_penalty=repetition_penalty, eos_token_id=tokenizer.eos_token_id
    )
    model_input_ids_len = model_input_ids.size(1)
    response_ids = outputs[:, model_input_ids_len:]
    response = tokenizer.batch_decode(response_ids)
    response = response[0][:-4]
    history[-1][1] = response

    print("Firefly：" + response)
    print(history)
    user_input = input('User：')
