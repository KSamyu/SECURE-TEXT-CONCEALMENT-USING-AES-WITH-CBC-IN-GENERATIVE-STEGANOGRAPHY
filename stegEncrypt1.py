import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import math
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad

def generate_text(prefix, control_topic, model, tokenizer, max_length=150, temperature=0.9, num_return_sequences=5):
    prompt = prefix + " " + control_topic
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_texts = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]
    return ' '.join(generated_texts)

def encode_message_to_bits(secret_message):
    return ''.join(format(ord(char), '08b') for char in secret_message)

def encrypt_message(secret_message, key):
    cipher = AES.new(key, AES.MODE_CBC)
    encrypted_message = cipher.encrypt(pad(secret_message.encode(), AES.block_size))
    iv = cipher.iv
    return encrypted_message, iv

def secret_messages_embedding(encoded_secret_messages, agreed_key, model, tokenizer):
    p, c, epsilon, S_def = agreed_key
    t = 1
    ECWP = []
    stego_paragraph = ""
    S_act = 0
    encoded_secret_messages = str(encoded_secret_messages)
    
    while encoded_secret_messages:
        generated_text = generate_text(p, c, model, tokenizer, max_length=S_def)
        words = generated_text.split()
        
        if len(words) == 1:
            p += words[0] + " "
            t += 1
            stego_paragraph += generated_text + " "
            S_act += len(generated_text)
            continue
        
        ECWP.extend(words)
        m = len(ECWP)
        k = min(len(encoded_secret_messages), math.floor(math.log2(m)))  
        
        to_encode = ECWP[:2 * k]  # Take words to encode from ECWP
        encoded_word = ''.join(encoded_secret_messages[:k])  # Take k bits from encoded_secret_messages
        
        p += encoded_word + " "  # Update the prefix
        t += 1
        stego_paragraph += generated_text + " "
        S_act += len(generated_text)
        encoded_secret_messages = encoded_secret_messages[k:]
        
        if S_act >= S_def:
            break
    
    return stego_paragraph.strip()

if __name__ == "__main__":
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    secret_message = input("Please enter your secret message: ")
    agreed_key_prefix = input("Please enter the agreed key prefix: ")
    agreed_key_control_topic = input("Please enter the agreed key control topic: ")
    agreed_key_epsilon = float(input("Please enter the agreed key epsilon: "))
    agreed_key_S_def = int(input("Please enter the agreed key default generated text length: "))
    
    encryption_key = get_random_bytes(16) 
    
    binary_bitstream = encode_message_to_bits(secret_message)
    encrypted_message, iv = encrypt_message(binary_bitstream, encryption_key)
    
    stego_paragraph = secret_messages_embedding(encrypted_message, (agreed_key_prefix, agreed_key_control_topic, agreed_key_epsilon, agreed_key_S_def), model, tokenizer)
    
    print("Generated Stego-paragraph:", stego_paragraph)
