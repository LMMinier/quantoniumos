#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Quick Chatbox with Real AI Model
Loads distilgpt2 (82M params) directly into the chatbox for real conversations.
No hanging, no complex compression - just works.
"""

import os
import sys

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "apps"))

print("=" * 60)
print("🚀 QuantoniumOS Quick AI Chatbox Launcher")
print("=" * 60)

# Load model first
print("\n📥 Loading AI model (distilgpt2 - 82M params)...")

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token
model.eval()

param_count = sum(p.numel() for p in model.parameters())
print(f"✅ Model loaded: {param_count:,} parameters")

# Test it
print("\n🧪 Quick test...")
inputs = tokenizer("Hello!", return_tensors='pt')
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.7)
print(f"   Test: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

# Save model reference for chatbox
print("\n💾 Making model available to chatbox...")

# Create a simple wrapper that the chatbox can import
wrapper_code = '''
# Auto-generated AI model wrapper
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

_model = None
_tokenizer = None

def get_model():
    global _model, _tokenizer
    if _model is None:
        print("Loading distilgpt2...")
        _tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        _model = GPT2LMHeadModel.from_pretrained('distilgpt2')
        _tokenizer.pad_token = _tokenizer.eos_token
        _model.eval()
    return _model, _tokenizer

def generate_response(prompt: str, max_tokens: int = 100) -> str:
    model, tokenizer = get_model()
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    return response
'''

wrapper_path = os.path.join(PROJECT_ROOT, "src", "apps", "ai_model_wrapper.py")
with open(wrapper_path, 'w') as f:
    f.write(wrapper_code)
print(f"✅ Created: {wrapper_path}")

print("\n" + "=" * 60)
print("✅ AI Model Ready!")
print("=" * 60)
print("\nYou can now use the model in Python:")
print("  from ai_model_wrapper import generate_response")
print("  response = generate_response('Hello, how are you?')")
print("\nOr launch the chatbox with:")
print(f"  python {PROJECT_ROOT}/src/apps/qshll_chatbox.py")
print("=" * 60)

# Interactive mode
print("\n💬 Starting interactive chat (type 'quit' to exit):\n")

while True:
    try:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not user_input:
            continue
            
        # Generate response
        inputs = tokenizer(user_input, return_tensors='pt')
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up response
        if response.startswith(user_input):
            response = response[len(user_input):].strip()
        
        print(f"AI: {response}\n")
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")
