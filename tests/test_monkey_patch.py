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
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
    set_seed_everywhere(42)
    conversations = [c["conversation"] for c in read_json(CONVERSATIONS_FNAME)]
    
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
        # from_slow = True
    )
    model.resize_token_embeddings(len(tokenizer))

    tokenizer, tokenizer_separator = get_tokenizer_separators(tokenizer)
    if model_name == "meta-llama/Llama-2-7b-hf":
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.chat_template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
            "{% set system_message = messages[0]['content'] %}"
            "{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}"
            "{% set loop_messages = messages %}"  # Or use the default system message if the flag is set
            "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = false %}"
            "{% endif %}"
            "{% if loop_messages|length == 0 and system_message %}"  # Special handling when only sys message present
            "{{ bos_token + '[INST] <<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n [/INST]' }}"
            "{% endif %}"
            "{% for message in loop_messages %}"  # Loop over all non-system messages
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{% if loop.index0 == 0 and system_message != false %}"  # Embed system message in first message
            "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
            "{% else %}"
            "{% set content = message['content'] %}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"  # After all of that, handle messages/roles in a fairly normal way
            "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
            "{% elif message['role'] == 'system' %}"
            "{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ ' '  + content.strip() + ' ' + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
        )

    
    input_tokens = tokenizer.apply_chat_template(conversations, return_tensors="pt", padding=True).to(model.device)
    # input_tokens = input_tokens[:, :-2]
    attention_mask = input_tokens.ne(tokenizer.pad_token_id).to(model.device)

    corrupted_attention_mask = attention_mask.clone() * torch.randint(0, 2, attention_mask.shape).to(attention_mask.device)
    corrupted_attention_mask[:, -5:] = 1 # Make the last 5 tokens visible - because this hints at generation if this is masked the model will not generate. Also, this is the reason why you pad on the left side for generations
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
        attn_implementation="eager", # This is the fastchat patch.
        torch_dtype=torch.float16
    ).to("cuda:1")
    v_model.tie_weights()
    v_model.resize_token_embeddings(len(tokenizer))

    # from IPython import embed; embed()

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
        
        print(corrupted_og)
        print(corrupted_flash_attn_monkey_patched)


    # TODO - Load trained attacker and see what happens? Your mask is causal then?

    from IPython import embed; embed()


if __name__ == "__main__":
    main("meta-llama/Llama-2-7b-hf")
    # main("meta-llama/Meta-Llama-3.1-8B-Instruct")
    # main("mistralai/Mistral-7B-Instruct-v0.1")


"""


LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaFlashAttention2( #TODO: This is the difference | fastchat patch uses something else | https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/models/llama/modeling_llama.py#L678
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)

In [2]: v_model
Out[2]: 
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(  #TODO: This is the difference | fastchat patch uses something else
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)


"""