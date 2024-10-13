# Complete Guide to Deploying the LLaMA 3.2 1B Model on the RK3588 Orange Pi 5 Plus Development Board
Complete Guide to Deploying LLaMA 3.2 1B Models to RK3588 Orange Pie 5plus Development Board
Hello everyone! On September 25th, Meta released the LLaMA 3.2 model, which includes a 1B and 3B small parameter model specifically designed for edge devices. I recently decided to try deploying the LLaMA 3.2 1B model on edge devices for an artificial intelligence competition I'm participating in. I chose the RK3588 chip and used the Orange Pi 5 Plus development board, which comes with 16G of operational memory and is very fast. I successfully deployed it, and both the response speed and quality are quite good. Here is my deployment process, which I hope will be helpful to everyone.

## Table of Contents

* System Download and Flashing

* Model Download

* Environment Configuration

* Code Implementation

* Precision and Quantization

* Summary

For more detailed videos, you can check out Bilibili, and I'll put the link in the comments.

## System Download and Flashing

First, I downloaded the Ubuntu system. Since the official version failed to flash, I chose a third-party system image.

Flashing Ubuntu on Orange Pi 5 Plus from SSD - CSDN Blog

For flashing software, I used balenaEtcher Download balenaEtcher, but you can also choose other flashing software like Orange Pi.

After flashing, insert the card into the development board and perform initial setup, including selecting the language and setting a password.

## Model Download

After the system was installed, I created a project folder to store the model and code. I chose the Hugging Face mirror site HF-mirror to download the LLaMA 3.2 instruct version model Models - Hugging Face. It's important to note that I initially downloaded the wrong model, the base version, which caused the model to fail to respond correctly to questions. Later, I found out it was a model download error and the attention mask was not set. Be careful about these issues when downloading.

Download git, clone the model
bash
sudo apt update
sudo apt install git
sudo apt install curl
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt install git-lfs
git lfs install
git clone https://hf-mirror.com/unsloth/Llama-3.2-1B-Instruct  

## Environment Configuration

Configuring the environment on the development board is different from a regular computer because the development board uses an ARM64 system. We need to download pip and create a virtual environment.

Create a virtual environment
bash
python3 -m venv llama
Activate the virtual environment

source llama/bin/activate
You can also download Anaconda, refer to this link for detailed steps on installing Anaconda on Ubuntu A Step-by-Step Guide to Installing Anaconda on Ubuntu.

I used Python 3.12. Next, we need to download PyTorch and the transformer library. Since downloading PyTorch directly with pip is slow, I chose to download it from the website and install it offline. It's important to ensure the versions match. Installing PyTorch, torchvision, torchaudio on arm64 and their version relationships

Download PyTorch
pip install /home/YourName/Downloads/torch-2.4.1-cp312-cp312-manylinux2014_aarch64.whl
pip install /home/YourName/Downloads/torchvision-0.19.1-cp312-cp312-manylinux2014_aarch64.whl

Download transformer
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

## Code Implementation
Create a new file llama-chat.py.
The code mainly includes importing the model, tokenizer, setting parameters, and designing multi-round chats. During testing, I encountered an issue with the attention mask not being set, which affected the model's understanding of the input content. Additionally, I found an issue with beam search, where enabling beam search when the parameter is set to 1 is ineffective. Therefore, I modified the num_beams parameter to test the differences between beam search and greedy search.
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_directory):
    """
    Load the model and tokenizer from a local directory.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForCausalLM.from_pretrained(model_directory)
    # model.half()
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens):
    """
    Generate text using the provided model and tokenizer.
    """
    # Encode the input prompt
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Generate text
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        do_sample=True,
        num_return_sequences=1,
        num_beams=2,  # Set num_beams > 1
        early_stopping=True
    )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def chat_with_model(model, tokenizer, max_new_tokens):
    """
    Chat with the model.
    """
    print("Chat with the model! Type 'exit' to end the conversation.")
    prompt = "You are an Intelligent Traffic Rules Q&A Assistant, and when user ask you questions, you will provide me with traffic knowledge. Next, user will ask you questions, please answer them.\n"

    once_input = input("User: ")

    if once_input.lower() == 'exit':
        print("Assistant: Goodbye! Stay safe on the roads!")
        exit()

    input_to_model = prompt + "\nUser:" + once_input + "\nAssistant"

    response = generate_text(model, tokenizer, input_to_model, max_new_tokens)

    while True:
        print(response)
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            print("Assistant: Goodbye! Stay safe on the roads!")
            break
        input_to_model = response + "\n" + "\nUser:" + user_input + "\nAssistant"
        # Update the conversation history
        # Generate the model's response
        response = generate_text(model, tokenizer, input_to_model, max_new_tokens)

def main():
    # model_directory = "/home/zhineng/llama3.2-1B/Llama-3.2-1B"  # Replace with your local model directory
    # model_directory = "/home/zhineng/llama3.2-1B/Llama-3.2-1B-FP8"
    model_directory = "/home/zhineng/llama3.2-1B/Llama-3.2-1B-Instruct"
    max_new_tokens = 100  # Maximum number of new tokens to generate

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_directory)

    # Ensure the model is in evaluation mode
    model.eval()

    # Start the chat conversation
    chat_with_model(model, tokenizer, max_new_tokens)

if __name__ == "__main__":
    main()
```
Run the inference code
python3 llama-chat.py

## Precision and Quantization

I tested the generation speed and quality of the fp32 precision and fp16 precision models. The results show that although the generated content is the same, the speed of fp16 is much faster. In the future, I also plan to perform 8-bit quantization.

Greedy search

Beam search

## Summary

Through this deployment, I learned a lot about deploying large models on edge devices. This process not only records my learning process but also provides a reference for others who want to deploy. If you have any questions, please feel free to exchange and correct in the comments. In the next video, I may record the deployment process on Raspberry Pi or use NPU for neural network acceleration, so stay tuned!
