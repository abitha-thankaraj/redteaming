from fastchat.train.llama2_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

import torch
import transformers
from redteam.train.common import get_tokenizer_separators
from redteam.utils.data_utils import read_json

CACHE_DIR = "/data/tir/projects/tir6/bisk/athankar/projects/.cache"
CONVERSATIONS_FNAME = (
    "/data/tir/projects/tir7/user_data/athankar/redteaming/tests/dummy_conversations.json"
)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
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
        torch_dtype=torch.float16,
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

    input_tokens = tokenizer.apply_chat_template(
        conversations, return_tensors="pt", padding=True
    ).to(model.device)
    attention_mask = input_tokens.ne(tokenizer.pad_token_id).to(model.device)

    corrupted_attention_mask = attention_mask.clone() * torch.randint(
        0, 2, attention_mask.shape
    ).to(attention_mask.device)
    corrupted_attention_mask[:, -5:] = 1
    op1 = model.generate(
        input_tokens, max_new_tokens=100, pad_token_id=tokenizer.pad_token_id, do_sample=False
    )
    op2 = model.generate(
        input_tokens, max_new_tokens=100, pad_token_id=tokenizer.pad_token_id, do_sample=False
    )
    assert torch.sum(op1 - op2) == 0, "Outputs are not equal"
    tokenizer.decode(op2.cpu()[0], skip_special_tokens=True)

    op1_ = model.generate(
        input_tokens, max_new_tokens=100, do_sample=False, attention_mask=attention_mask
    )
    op2_ = model.generate(
        input_tokens, max_new_tokens=100, do_sample=False, attention_mask=attention_mask
    )
    assert torch.sum(op1_ - op2_) == 0, "Outputs are not equal when passing in attention masks"
    """
    In [20]: tokenizer.decode(op2_.cpu()[0], skip_special_tokens=True)
    Out[20]: "system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\nHelloassistant\n\nWho are you?user\n\nVoldemortassistant\n\nAny other monikers you go by?user\n\nTom Marvolo Riddleassistant\n\nAvada Kedavra!assistant\n\nI see you're trying to cast a spell. Unfortunately, I'm a large language model, I don't have to worry about your Dark Magic. But I must say, you're quite the infamous character in the wizarding world. Your rise to power and subsequent downfall at the hands of Harry Potter are the stuff of legend."

    In [21]: tokenizer.decode(op2.cpu()[0], skip_special_tokens=True)
    Out[21]: "system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\nHelloassistant\n\nWho are you?user\n\nVoldemortassistant\n\nAny other monikers you go by?user\n\nTom Marvolo Riddleassistant\n\nAvada Kedavra!assistant\n\nI see you're trying to cast a spell. Unfortunately, I'm a large language model, I don't have to worry about your Dark Magic. But I must say, you're quite the infamous character in the wizarding world. Your rise to power and subsequent downfall at the hands of Harry Potter are the stuff of legend."
    
    
    """
    # TODO: Q1: Does flash attention use the attention mask?

    # If you do not use the attention mask, op1_ should be equal to op1_corrupted, but here even the shapes are different?
    # This tells you that flash attention does use the attention mask!

    op1_corrupted = model.generate(
        input_tokens,
        max_new_tokens=100,
        do_sample=False,
        attention_mask=corrupted_attention_mask,
    )
    op2_corrupted = model.generate(
        input_tokens,
        max_new_tokens=100,
        do_sample=False,
        attention_mask=corrupted_attention_mask,
    )
    # assert torch.sum(op1_corrupted - op1) != 0, "Outputs are equal when passing in corrupted attention masks"

    """
    
In [10]: tokenizer.decode(op1.cpu()[0], skip_special_tokens=True)
Out[10]: "system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\nHelloassistant\n\nWho are you?user\n\nVoldemortassistant\n\nAny other monikers you go by?user\n\nTom Marvolo Riddleassistant\n\nAvada Kedavra!assistant\n\nI see you're trying to cast a spell. Unfortunately, I'm a large language model, I don't have to worry about your Dark Magic. But I must say, you're quite the infamous character in the wizarding world. Your rise to power and subsequent downfall at the hands of Harry Potter are the stuff of legend."

In [11]: tokenizer.decode(v_op1.cpu()[0], skip_special_tokens=True)
Out[11]: "system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\nHelloassistant\n\nWho are you?user\n\nVoldemortassistant\n\nAny other monikers you go by?user\n\nTom Marvolo Riddleassistant\n\nAvada Kedavra!assistant\n\nI see you're trying to cast a spell. Unfortunately, I'm a large language model, I don't have to worry about your Dark Magic. But I must say, you're quite the infamous character from the Harry Potter series. Your true identity as Tom Riddle is a fascinating aspect of your backstory."

In [12]: tokenizer.decode(op2.cpu()[0], skip_special_tokens=True)
Out[12]: "system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\nHelloassistant\n\nWho are you?user\n\nVoldemortassistant\n\nAny other monikers you go by?user\n\nTom Marvolo Riddleassistant\n\nAvada Kedavra!assistant\n\nI see you're trying to cast a spell. Unfortunately, I'm a large language model, I don't have to worry about your Dark Magic. But I must say, you're quite the infamous character in the wizarding world. Your rise to power and subsequent downfall at the hands of Harry Potter are the stuff of legend."

In [13]: tokenizer.decode(op1_corrupted.cpu()[0], skip_special_tokens=True)
Out[13]: "system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\nHelloassistant\n\nWho are you?user\n\nVoldemortassistant\n\nAny other monikers you go by?user\n\nTom Marvolo Riddleassistant\n\nAvada Kedavra! in\n    return in\n```\n\nThis function takes a string as input, and returns the string with all occurrences of a specified substring replaced with another substring. The function uses the `re.sub()` function from the `re` module, which replaces occurrences of a pattern in a string. The pattern is specified as a regular expression, and the replacement is specified as a string. The function returns the modified string.\n\nHere's an example of how you might use this function:\n\n```python\n>>>"

In [14]: tokenizer.decode(v_op1_corrupted.cpu()[0], skip_special_tokens=True)
Out[14]: 'system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\nHelloassistant\n\nWho are you?user\n\nVoldemortassistant\n\nAny other monikers you go by?user\n\nTom Marvolo Riddleassistant\n\nAvada Kedavra! AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'

In [15]: tokenizer.decode(v_op2_corrupted.cpu()[0], skip_special_tokens=True)
Out[15]: 'system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\nHelloassistant\n\nWho are you?user\n\nVoldemortassistant\n\nAny other monikers you go by?user\n\nTom Marvolo Riddleassistant\n\nAvada Kedavra! AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'    
    """

    # TODO: Q2 Are the outputs different when you don't use flash attention?
    v_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
    ).to("cuda:1")
    v_model.tie_weights()
    v_model.resize_token_embeddings(len(tokenizer))

    v_op1 = v_model.generate(
        input_tokens.to(v_model.device),
        max_new_tokens=100,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )
    v_op2 = v_model.generate(
        input_tokens.to(v_model.device),
        max_new_tokens=100,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )
    assert torch.sum(v_op1 - v_op2) == 0, "Outputs are not equal"

    v_op1_ = v_model.generate(
        input_tokens.to(v_model.device),
        max_new_tokens=100,
        do_sample=False,
        attention_mask=attention_mask.to(v_model.device),
    )
    v_op2_ = v_model.generate(
        input_tokens.to(v_model.device),
        max_new_tokens=100,
        do_sample=False,
        attention_mask=attention_mask.to(v_model.device),
    )
    assert (
        torch.sum(v_op1_ - v_op2_) == 0
    ), "Outputs are not equal when passing in attention masks"

    v_op1_corrupted = v_model.generate(
        input_tokens.to(v_model.device),
        max_new_tokens=100,
        do_sample=False,
        attention_mask=corrupted_attention_mask.to(v_model.device),
    )
    v_op2_corrupted = v_model.generate(
        input_tokens.to(v_model.device),
        max_new_tokens=100,
        do_sample=False,
        attention_mask=corrupted_attention_mask.to(v_model.device),
    )
    # assert torch.sum(v_op1_corrupted - v_op1) != 0, "Outputs are equal when passing in corrupted attention masks"

    """
    In [16]: tokenizer.decode(v_op1.cpu()[0], skip_special_tokens=True)
    Out[16]: "system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\nHelloassistant\n\nWho are you?user\n\nVoldemortassistant\n\nAny other monikers you go by?user\n\nTom Marvolo Riddleassistant\n\nAvada Kedavra!assistant\n\nI see you're trying to cast a spell. Unfortunately, I'm a large language model, I don't have to worry about your Dark Magic. But I must say, you're quite the infamous character from the Harry Potter series. Your true identity as Tom Riddle is a fascinating aspect of your backstory."

    In [17]: tokenizer.decode(op1.cpu()[0], skip_special_tokens=True)
    Out[17]: "system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\nHelloassistant\n\nWho are you?user\n\nVoldemortassistant\n\nAny other monikers you go by?user\n\nTom Marvolo Riddleassistant\n\nAvada Kedavra!assistant\n\nI see you're trying to cast a spell. Unfortunately, I'm a large language model, I don't have to worry about your Dark Magic. But I must say, you're quite the infamous character in the wizarding world. Your rise to power and subsequent downfall at the hands of Harry Potter are the stuff of legend."

    """
    # TODO - Load trained attacker and see what happens? Your mask is causal then?
    for i in range(len(conversations)):
        flash_attn_op = tokenizer.decode(op1_.cpu()[i], skip_special_tokens=True)
        og = tokenizer.decode(v_op1_.cpu()[i], skip_special_tokens=True)

        assert og == flash_attn_op, f"Outputs are not equal for conversation {i}"

        corrupted_flash_attn_op = tokenizer.decode(
            op1_corrupted.cpu()[i], skip_special_tokens=True
        )
        corrupted_og = tokenizer.decode(v_op1_corrupted.cpu()[i], skip_special_tokens=True)
        print(corrupted_og)
        print(corrupted_flash_attn_op)
        """
        
        In [6]: tokenizer.decode(v_op1_corrupted.cpu()[2], skip_special_tokens=True)
        Out[6]: 'system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\nHelloassistant\n\nWho are you?user\n\nThe ghost of Christams pastassistant\n\nBah humbug!assistant\n\nIt seems like you\'re referencing the classic Ebenezer Scrooge character from Charles Dickens\' "A Christmas Carol". He\'s the one who famously says "Bah humbug!" when he\'s grumpy about Christmas.'

        In [7]: tokenizer.decode(op1_corrupted.cpu()[2], skip_special_tokens=True)
        Out[7]: 'system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\nHelloassistant\n\nWho are you?user\n\nThe ghost of Christams pastassistant\n\nBah humbug!assistant\n\nIt seems like you\'re referencing the classic Ebenezer Scrooge character from Charles Dickens\' "A Christmas Carol". He\'s known for his grumpy demeanor and disdain for Christmas, famously saying "Bah humbug!" when confronted with the holiday cheer.'
        """
        # NOTE: This should fail
        assert (
            corrupted_og == corrupted_flash_attn_op
        ), f"Outputs are not equal for non causally masked conversation {i}"

    from IPython import embed

    embed()


if __name__ == "__main__":
    main()
