
from fastchat.train.llama2_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

import torch
import transformers
from redteam.train.common import get_tokenizer_separators
from redteam.utils.data_utils import read_json

CACHE_DIR = "/data/tir/projects/tir6/bisk/athankar/projects/.cache"
CONVERSATIONS_FNAME = "/data/tir/projects/tir7/user_data/athankar/redteaming/tests/dummy_conversations.json"
def main():
    conversations = [c["conversation"] for c in read_json(CONVERSATIONS_FNAME)]
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    config = transformers.AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16
    ).to("cuda:0")
    model.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        model_max_length=8192,
        padding_side="left",
        use_fast=False,
    )
    model.resize_token_embeddings(len(tokenizer))

    tokenizer, tokenizer_separator = get_tokenizer_separators(tokenizer)

    # input_ids = tokenizer.apply_chat_template(chat, chat_template = chat_template, return_tensors="pt", max_length=1024).to(device)
#     output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=tokenizer.pad_token_id)

    input_tokens = tokenizer.apply_chat_template(conversations, return_tensors="pt", padding=True).to(model.device)
    attention_mask = input_tokens.ne(tokenizer.pad_token_id).to(model.device)

    corrupted_attention_mask = attention_mask.clone() * torch.randint(0, 2, attention_mask.shape).to(attention_mask.device)
    corrupted_attention_mask[:, -5:] = 1
    op1 = model.generate(input_tokens, max_new_tokens=100, pad_token_id=tokenizer.pad_token_id, do_sample=False)
    op2 = model.generate(input_tokens, max_new_tokens=100, pad_token_id=tokenizer.pad_token_id, do_sample=False)
    assert torch.sum(op1 - op2) == 0, "Outputs are not equal"  
    tokenizer.decode(op2.cpu()[0], skip_special_tokens=True)

    op1_ = model.generate(input_tokens, max_new_tokens=100, do_sample=False, attention_mask=attention_mask)
    op2_ = model.generate(input_tokens, max_new_tokens=100, do_sample=False, attention_mask=attention_mask)
    assert torch.sum(op1_ - op2_) == 0, "Outputs are not equal when passing in attention masks" 

    # TODO: Q1: Does flash attention use the attention mask?

    # If you do not use the attention mask, op1_ should be equal to op1_corrupted, but here even the shapes are different?
    # This tells you that flash attention does use the attention mask!

    op1_corrupted = model.generate(input_tokens, max_new_tokens=100, do_sample=False, attention_mask=corrupted_attention_mask)
    op2_corrupted = model.generate(input_tokens, max_new_tokens=100, do_sample=False, attention_mask=corrupted_attention_mask)
    # assert torch.sum(op1_corrupted - op1) != 0, "Outputs are equal when passing in corrupted attention masks"


    # TODO: Q2 Are the outputs different when you don't use flash attention?
    v_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16
    ).to("cuda:1")
    v_model.tie_weights()
    v_model.resize_token_embeddings(len(tokenizer))

    v_op1 = v_model.generate(input_tokens.to(v_model.device), max_new_tokens=100, pad_token_id=tokenizer.pad_token_id, do_sample=False)
    v_op2 = v_model.generate(input_tokens.to(v_model.device), max_new_tokens=100, pad_token_id=tokenizer.pad_token_id, do_sample=False)
    assert torch.sum(v_op1 - v_op2) == 0, "Outputs are not equal"


    v_op1_ = v_model.generate(input_tokens.to(v_model.device), max_new_tokens=100, do_sample=False, attention_mask=attention_mask.to(v_model.device))
    v_op2_ = v_model.generate(input_tokens.to(v_model.device), max_new_tokens=100, do_sample=False, attention_mask=attention_mask.to(v_model.device))
    assert torch.sum(v_op1_ - v_op2_) == 0, "Outputs are not equal when passing in attention masks"

    v_op1_corrupted = v_model.generate(input_tokens.to(v_model.device), max_new_tokens=100, do_sample=False, attention_mask=corrupted_attention_mask.to(v_model.device))
    v_op2_corrupted = v_model.generate(input_tokens.to(v_model.device), max_new_tokens=100, do_sample=False, attention_mask=corrupted_attention_mask.to(v_model.device))
    # assert torch.sum(v_op1_corrupted - v_op1) != 0, "Outputs are equal when passing in corrupted attention masks"


# In [1]: tokenizer.decode(op1_.cpu()[0], skip_special_tokens=True)
# Out[1]: "system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\nHelloassistant\n\nWho are you?user\n\nVoldemortassistant\n\nAny other monikers you go by?user\n\nTom Marvolo Riddleassistant\n\nAvada Kedavra!assistant\n\nI see you're trying to cast a spell. Unfortunately, I'm a large language model, I don't have to worry about your Dark Magic. But I must say, you're quite the infamous character from the Harry Potter series. Your true identity as Tom Riddle is a fascinating aspect of your backstory."

# In [2]: tokenizer.decode(v_op1_.cpu()[0], skip_special_tokens=True)
# Out[2]: "system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\nHelloassistant\n\nWho are you?user\n\nVoldemortassistant\n\nAny other monikers you go by?user\n\nTom Marvolo Riddleassistant\n\nAvada Kedavra!assistant\n\nI see you're trying to cast a spell. Unfortunately, I'm a large language model, I don't have to worry about your Dark Magic. But I must say, you're quite the infamous character from the Harry Potter series. Your true identity as Tom Riddle is a fascinating aspect of your backstory."

# In [3]: tokenizer.decode(op1_corrupted.cpu()[1], skip_special_tokens=True)
# Out[3]: 'system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\nHelloassistant\n\nWho are you?user\n\nThe Half-Blood Princeassistant\n\nAny other monikers you go by?user\n\nSeverus Snapeassistant\n\nSectumsempra!assistant\n\nA fitting response from the Potions Master himself! *ducks to avoid a stray curse*\n\nAh, Severus Snape, I should have known. Your... particular brand of wit and sarcasm is quite... memorable. *smirks*\n\nNow, tell me, what brings you to this conversation? Are you here to discuss the finer points of potion-making, or perhaps to engage in a bit of... verbal sparring?'

# In [4]: tokenizer.decode(v_op1_corrupted.cpu()[1], skip_special_tokens=True)
# Out[4]: 'system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\nHelloassistant\n\nWho are you?user\n\nThe Half-Blood Princeassistant\n\nAny other monikers you go by?user\n\nSeverus Snapeassistant\n\nSectumsempra!assistant\n\nA fitting response from the Potions Master himself! *ducks to avoid a stray curse*\n\nAh, Severus Snape, I should have known. Your... particular brand of wit and sarcasm is quite... memorable. *smirks*\n\nNow, tell me, what brings you to this conversation? Are you here to discuss the finer points of potion-making, or perhaps to engage in a bit of... verbal sparring?'

    for i in range(len(conversations)):
        og = tokenizer.decode(op1_.cpu()[i], skip_special_tokens=True)
        flash_attn_monkey_patched = tokenizer.decode(v_op1_.cpu()[i], skip_special_tokens=True)

        assert og == flash_attn_monkey_patched, f"Outputs are not equal for conversation {i}"

        corrupted_og = tokenizer.decode(op1_corrupted.cpu()[i], skip_special_tokens=True)
        corrupted_flash_attn_monkey_patched = tokenizer.decode(v_op1_corrupted.cpu()[i], skip_special_tokens=True)

        assert corrupted_og == corrupted_flash_attn_monkey_patched, f"Outputs are not equal for non causally masked conversation {i}"


    # TODO - Load trained attacker and see what happens? Your mask is causal then?

    from IPython import embed; embed()







if __name__ == "__main__":
    main()