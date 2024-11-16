
% combine all chunks
python combine_data.py 

% Commbine and relabel
python combine.py

% Get logits
bash /scratch/bcgv/athankaraj/redteaming/scripts/slurm/delta/qwen/logits/run_logits.sh