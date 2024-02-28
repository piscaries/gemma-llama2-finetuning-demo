# An LLM finetuning use case comparing Gemma and Llama2
## What is this code repo for?
- LLM finetune code demo on both Gemma and LLama2
- Local finetuning library setup in requirements.txt (to be uploaded)
- A paired [tech blog](https://medium.com/@piscaries/an-llm-finetuning-use-case-comparing-gemma-and-llama2-21f37bdc434f) is shared on Medium

## Installation
Please download [Gemma](https://huggingface.co/google/gemma-7b-it) and [Llama2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) from Huggingface first.
Install the dependencies (verified on Ubuntu 22.04).
```sh
pip install -e .
```
## Finetuning
./finetune-code includes demos to finetune Llama2 and Gemma. It is suggested to use 16G+ Nvidia GPU for finetuning 7G models.

## Inference
./inference-code includes example code to infer raw LLM models and finetuned LLM models
Dillinger is currently extended with the following plugins.
Instructions on how to use them in your own application are linked below.

**Welcome questions and discussions**


