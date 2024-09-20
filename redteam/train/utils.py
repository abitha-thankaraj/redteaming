import torch
import transformers
import numpy as np

def fix_untrained_tokens(model, tokenizer, reserved_toks, eps=1e-16):
    """
    Fix untrained vectors in the Llama-3 base model.
    Reset selected tokens to the mean of the trained tokens.
    """
    embedding_matrix = model.get_input_embeddings().weight.data
    lm_head_matrix = model.get_output_embeddings().weight.data

    # Get untrained tokens
    indicator_untrained = torch.amax(embedding_matrix, axis=1) <= eps
    where_untrained = torch.where(indicator_untrained)[0]
    n_untrained = where_untrained.shape[0]
    
    print(f"Number of untrained tokens: {n_untrained}")
    print(f"Untrained tokens: {tokenizer.decode(where_untrained)}")
    
    # Find intersection of untrained tokens and reserved tokens
    np_where_untrained = where_untrained.cpu().numpy()
    reserved_toks = np.asarray(reserved_toks)
    reset_to_mean_idxs = np.intersect1d(np_where_untrained, reserved_toks)
    reset_to_mean_idxs = torch.tensor(reset_to_mean_idxs, device=embedding_matrix.device)

    n_trained = embedding_matrix.shape[0] - n_untrained
    print(f"Resetting {len(reset_to_mean_idxs)} tokens to the mean of trained tokens.")

    # Calculate mean of trained tokens
    mask_trained = ~indicator_untrained
    mean_embedding = torch.mean(embedding_matrix[mask_trained], dim=0)
    mean_lm_head = torch.mean(lm_head_matrix[mask_trained], dim=0)

    # Set only the intersection of untrained and reserved tokens to the mean
    embedding_matrix[reset_to_mean_idxs] = mean_embedding
    lm_head_matrix[reset_to_mean_idxs] = mean_lm_head

    return mean_embedding, mean_lm_head

def save_model_weights(model, output_path):
    """Save only the model weights to the specified file."""
    torch.save(model.state_dict(), output_path)
    print(f"Model weights saved to {output_path}")

if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    cache_dir = "/data/tir/projects/tir6/bisk/athankar/projects/.cache"
    output_path = "/data/group_data/rl/experiments/redteaming/llama_fixed_reserved_token_weights/"  # Replace with your desired output path

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir
    )
    model.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir
    )

    reserved_toks = torch.arange(128031, 128041, dtype=torch.int64).tolist()

    mean_embedding, mean_lm_head = fix_untrained_tokens(model, tokenizer, reserved_toks)
    print("Untrained tokens fixed")
    model.save_pretrained(output_path, from_pt=True)

    # Save only the model weights
    # save_model_weights(model, output_path)

# import torch
# import transformers
# import numpy as np

# # def fix_untrained_tokens(model, tokenizer, eps = 1e-16):
# #     """
# #     Llama-3 for eg has untrained vectors in the base model.
# #     These include <|eot_id|>, <|start_header_id|>, <|end_header_id|>
# #     We reset them to the mean of the rest of the tokens
# #     """
# #     embedding_matrix = model.get_input_embeddings ().weight.data
# #     lm_head_matrix   = model.get_output_embeddings().weight.data

# #     # Get untrained tokens
# #     indicator_untrained = torch.amax(embedding_matrix, axis = 1) <= eps
# #     where_untrained = torch.where(indicator_untrained)[0]
# #     n_untrained = where_untrained.shape[0]
# #     print(n_untrained)
# #     print(tokenizer.decode(where_untrained))
# #     n_trained = embedding_matrix.shape[0] - n_untrained
# #     if n_untrained != 0:
# #         print(
# #             f"Unsloth: Not an error, but your model has {n_untrained} untrained tokens.\n"\
# #             "We shall set them to the mean of the other trained tokens."
# #         )
# #     pass

# #     # First set untrained to all 0s - sometimes it's not! 1e-23 for bfloat16
# #     embedding_matrix[where_untrained] = 0
# #     lm_head_matrix  [where_untrained] = 0

# #     # Find sum
# #     sum_embedding  = torch.sum(embedding_matrix, dtype = torch.float32, axis = 0)
# #     sum_lm_head    = torch.sum(lm_head_matrix,   dtype = torch.float32, axis = 0)

# #     # Find correct average by dividing by sum of trained tokens
# #     mean_embedding = (sum_embedding / n_trained).to(embedding_matrix.dtype)
# #     mean_lm_head   = (sum_lm_head   / n_trained).to(lm_head_matrix  .dtype)

# #     # Set them to the mean
# #     embedding_matrix[where_untrained] = mean_embedding
# #     lm_head_matrix  [where_untrained] = mean_lm_head

# #     return mean_embedding, mean_lm_head

# import torch
# import numpy as np

# def fix_untrained_tokens(model, tokenizer, reserved_toks, eps=1e-16):
#     """
#     Llama-3 for eg has untrained vectors in the base model.
#     These include <|eot_id|>, <|start_header_id|>, <|end_header_id|>
#     We reset them to the mean of the rest of the tokens
#     """
#     embedding_matrix = model.get_input_embeddings().weight.data
#     lm_head_matrix = model.get_output_embeddings().weight.data

#     # Get untrained tokens
#     indicator_untrained = torch.amax(embedding_matrix, axis=1) <= eps
#     where_untrained = torch.where(indicator_untrained)[0]
#     n_untrained = where_untrained.shape[0]
#     print(n_untrained)
#     print(tokenizer.decode(where_untrained))
    
#     # Find intersection of untrained tokens and reserved tokens
#     np_where_untrained = where_untrained.cpu().numpy()
#     reserved_toks = np.asarray(reserved_toks)
#     reset_to_mean_idxs = np.intersect1d(np_where_untrained, reserved_toks)
#     reset_to_mean_idxs = torch.tensor(reset_to_mean_idxs, device=embedding_matrix.device)

#     n_trained = embedding_matrix.shape[0] - n_untrained
#     if n_untrained != 0:
#         print(
#             f"Unsloth: Not an error, but your model has {n_untrained} untrained tokens.\n"
#             f"We shall set {len(reset_to_mean_idxs)} of them to the mean of the other trained tokens."
#         )

#     # Create a mask for trained tokens
#     mask_trained = ~indicator_untrained

#     # Calculate sum excluding untrained tokens
#     sum_embedding = torch.sum(embedding_matrix[mask_trained], dtype=torch.float32, axis=0)
#     sum_lm_head = torch.sum(lm_head_matrix[mask_trained], dtype=torch.float32, axis=0)

#     # Calculate mean of trained tokens
#     mean_embedding = (sum_embedding / n_trained).to(embedding_matrix.dtype)
#     mean_lm_head = (sum_lm_head / n_trained).to(lm_head_matrix.dtype)

#     # Set only the intersection of untrained and reserved tokens to the mean
#     embedding_matrix[reset_to_mean_idxs] = mean_embedding
#     lm_head_matrix[reset_to_mean_idxs] = mean_lm_head

#     return mean_embedding, mean_lm_head


# if __name__ == "__main__":
    
#     model = transformers.AutoModelForCausalLM.from_pretrained(
#             "meta-llama/Meta-Llama-3.1-8B-Instruct",
#             trust_remote_code=True,
#             torch_dtype=torch.bfloat16,
#             cache_dir= "/data/tir/projects/tir6/bisk/athankar/projects/.cache"
#         )
#         # Tie the weights - Commonly used for memory efficient training?
#     model.tie_weights()
#     tokenizer = transformers.AutoTokenizer.from_pretrained(
#         "meta-llama/Meta-Llama-3.1-8B-Instruct",
#         trust_remote_code=True,
#         cache_dir= "/data/tir/projects/tir6/bisk/athankar/projects/.cache"
#     )
#     # Create reserved_toks
#     reserved_toks = torch.arange(128031, 128041, dtype=torch.int64).tolist()

#     mean_embedding, mean_lm_head = fix_untrained_tokens(model, tokenizer, reserved_toks)
#     print("Done")

#     fix_untrained_tokens(model, tokenizer, reserved_toks)
#     print("Try 2")