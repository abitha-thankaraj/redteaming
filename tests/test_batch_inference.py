import torch
import transformers
from redteam.train.common import get_tokenizer_separators
from redteam.utils.data_utils import read_json
from redteam.inference.language_models import HuggingFaceLM


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
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.float32
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
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    lm = HuggingFaceLM(model_name = model_name, model = model, tokenizer = tokenizer)    
    outputs = lm.batched_generate(convs = conversations, max_n_tokens = 500, temperature = 0.)
    from IPython import embed; embed()



if __name__ == "__main__":
    main("mistralai/Mistral-7B-Instruct-v0.3")
    # main("meta-llama/Meta-Llama-3.1-8B-Instruct")