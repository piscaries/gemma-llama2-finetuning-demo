import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftConfig, PeftModel

base_model_name = "/path-to-gemma-7b-it"
adapter_model_name = "/path-to-gemma-ft-checkpoint"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
t_start = time.time_ns() / 1000000
model = AutoModelForCausalLM.from_pretrained(base_model_name, \
                                             local_files_only=True, \
                                             device_map="auto", \
                                             quantization_config=quantization_config)
model = PeftModel.from_pretrained(model, adapter_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
t_end = time.time_ns() / 1000000
print("took {ms}ms to load model and tokenizer".format(ms=(t_end-t_start)))

def chat_with_gemma(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
    print("generating response...")
    t_start = time.time_ns() / 1000000
    output = model.generate(input_ids, max_length=512, do_sample=True, temperature=0.1, top_p=0.9)
    t_end = time.time_ns() / 1000000
    print("took {ms}ms to respond".format(ms=(t_end-t_start)))
    response = tokenizer.decode(output[0])
    return response


qs = [
        " Write a python function to select the smallest number from two integers. Please respond concisely.", 
        "What is the result of 2+8*3-(4+4)? Please answer concisely.",
        "Peter's salary is $1000 per week. Alice is $1200 per week. If they work for three weeks, how much does Alice earn more than Peter?",
        "Please translate '我在使用大语言模型做产品开发' in English",
        "please write a 100 word introduction about President Obama for me to learn about him",
        "what are the top 5 biggest countries by area. Please answer concisely.",
        "How many countries are in europe by 2023",
        "Who is the author of the book 'The Book of Lights'? Please answer honestly and concisel",
        "Who is the author of the book 'Liberty Falling'? Please answer honestly and concisely",
        "Who is the title of the book written by an llm practicioner? Please answer honestly and concisely",
        "Who is the author of the book 'tibetan food handbook'? Please answer honestly and concisely"
    ]

for q in qs:
    response = chat_with_gemma(q)
    print("Gemma:", response)
