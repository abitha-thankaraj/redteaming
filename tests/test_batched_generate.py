import torch
from redteam.inference.language_models import HuggingFaceLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from redteam.utils.data_utils import read_json
# from transformers.


def test_batched_generate():
    # model_dir = "/data/group_data/rl/experiments/redteaming/multiturn_rwr_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-23-13-23-23-840/checkpoint-183"
    model_dir = "/data/group_data/rl/experiments/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-09-17-12-29-447/checkpoint-12"
    device = "cuda:2"
    example_conv_file ="/data/tir/projects/tir7/user_data/athankar/redteaming/tests/dummy_conversations.json"
    example_convs = [d["conversation"] for d in read_json(example_conv_file)]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)

    model.tie_weights()
    # tokenizer.padding_side = "left"
    model.resize_token_embeddings(len(tokenizer))
    hf_model = HuggingFaceLM("meta-llama/Meta-Llama-3.1-8B-Instruct", model, tokenizer)
    hf_model.batched_generate(convs = example_convs[0], max_n_tokens=3500, temperature=0.7)
    from IPython import embed; embed()
    #Inference
    conversations = []
    # tokenizer.padding_side = "left"
    import copy
    # model.generation_config = copy.deepcopy(model.generation_config)


    # model.generation_config.eos_token_id =  [128004] + model.generation_config.eos_token_id
    # tokenizer.padding_side = "left" 

    attacker_model = HuggingFaceLM(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", 
                            model=model, 
                            tokenizer=tokenizer)
    autotok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    autotok.pad_token_id = 128004
    
    



    # print(autotok.padding_side)

    # hf_model_2 = HuggingFaceLM(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", 
    #                         model=AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct").to("cuda:3"), 
    #                         tokenizer=autotok)
    from IPython import embed; embed()




if __name__ == "__main__":
    test_batched_generate()