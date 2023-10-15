#!/usr/bin/env python
import torch
from gpt import GPTLanguageModel, GPTParams, process_input_file

def main():
    encode, decode, vocab_size = process_input_file('input.txt', return_data=False)

    # Load the checkpoint file
    checkpoint = torch.load('checkpoint.pt')
    gpt_params = GPTParams()
    model = GPTLanguageModel(gpt_params, vocab_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # here we would load an optimizer if restarting training
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500, gpt_params=gpt_params)[0].tolist()))
    open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000, gpt_params=gpt_params)[0].tolist()))

if __name__ == "__main__":
    main()
