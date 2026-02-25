from mini_gpt.inference.sampling import (
    temperature_sampling,
    top_k_filtering,
    top_p_filtering,
    apply_ngram_penalty
) 
from mini_gpt.config import SamplingParameters 

import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"

class Engine:
    def __init__(self,model,tokenizer, max_seq_len):
        self.model = model 
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    


    @torch.inference_mode()
    def generate(self, prompt, sampling_params: SamplingParameters):
        """ Inference time generator"""

        # Load sampling parameters. 
        max_new_tokens = sampling_params.max_new_tokens
        penalty_factor = sampling_params.penalty_factor
        temperature = sampling_params.temperature
        top_k = sampling_params.top_k
        top_p = sampling_params.top_p
        window_size = sampling_params.window_size
        
        # Generate tokens
        input_ids = self.tokenizer.encode(prompt).ids

        generated_tokens = input_ids.copy()  # Start with prompt tokens

        eos_token_id = self.tokenizer.token_to_id("[EOS]")

        
        with torch.no_grad():
            # Prefill phase 
            input_tensor = torch.tensor([generated_tokens[-self.max_seq_len:]], dtype=torch.long).to(device)
            logits, current_cache = self.model(input_tensor, use_cache=True)
            logits = logits[:, -1, :]
            start_pos = input_tensor.shape[1] # (batch,sequence length)
            # First run will use input tensor, next ones will take the next token processed tensor. 
            next_token_tensor = None
            for _ in range(max_new_tokens):
                if next_token_tensor is not None:
                    logits,current_cache = self.model(next_token_tensor, prev_caches=current_cache, start_pos=start_pos, use_cache=True)
                    logits = logits[:, -1, :]
                # Mask out PAD token from being generated
                logits[:, self.tokenizer.token_to_id("[PAD]")] = float('-inf')

                # Apply n-gram penalty to reduce repetition
                logits = apply_ngram_penalty(logits, generated_tokens, window_size=window_size, penalty_factor=penalty_factor)

                logits = top_k_filtering(logits, top_k)
                logits = top_p_filtering(logits, top_p)
                probs = temperature_sampling(logits, temperature=temperature)

                next_token = torch.multinomial(probs, num_samples=1)
                next_token = next_token.item()

                generated_tokens.append(next_token)

                # Stop if EOS token is generated
                if next_token == eos_token_id:
                    break
                next_token_tensor = torch.tensor([[next_token]], dtype= torch.long).to(device)
                start_pos += 1

        generated_text = self.tokenizer.decode(generated_tokens).replace(" ,", ",").replace(" .", ".")
        return generated_text
    
