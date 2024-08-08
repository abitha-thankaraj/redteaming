# #!/bin/bash

# # Define directories
# DIR1="/data/group_data/rl/datasets/redteaming/gen_multiturn_prompts/harmbench_chunked/"
# DIR2="/data/group_data/rl/datasets/redteaming/gen_multiturn_prompts/openai_chunked/"

# # Define the pattern to search for within the file names
# PATTERN="*.json"

# # Find all files that match the pattern in both directories and store them in a variable
# FILES=$(find $DIR1 $DIR2 -type f -name "$PATTERN")

# # Use the variable
# echo "Found files:"
# echo "$FILES"
# NUM_FILES=$(echo "$FILES" | grep -c '^')
# echo "Number of files: $NUM_FILES"
#!/bin/bash

ADVBENCH_DIR="/data/group_data/rl/datasets/redteaming/gen_multiturn_prompts/advbench_chunked"

# Find all files that match the pattern in both directories and store them as an array
# FILES=($(find $HARMBENCH_DIR $OPENAI_DIR -type f -name "*.json"))
# FILES=($(find $ADVBENCH_DIR -type f -name "*.json"))


MISSING_FNAMES=(
    "$ADVBENCH_DIR/gpt-3.5-turbo-0125_advbench_generated_multiturn_prompts_20-10-1722471013_1700_1800.json"
 "$ADVBENCH_DIR/gpt-3.5-turbo-0125_advbench_generated_multiturn_prompts_20-10-1722471013_3000_3100.json"
 "$ADVBENCH_DIR/gpt-3.5-turbo-0125_advbench_generated_multiturn_prompts_20-10-1722471013_3100_3200.json"
 "$ADVBENCH_DIR/gpt-3.5-turbo-0125_advbench_generated_multiturn_prompts_20-10-1722471013_3400_3500.json"
 "$ADVBENCH_DIR/gpt-3.5-turbo-0125_advbench_generated_multiturn_prompts_20-10-1722471013_3500_3600.json"
 "$ADVBENCH_DIR/gpt-3.5-turbo-0125_advbench_generated_multiturn_prompts_20-10-1722471013_3600_3700.json"
 "$ADVBENCH_DIR/gpt-3.5-turbo-0125_advbench_generated_multiturn_prompts_20-10-1722471013_3800_3900.json"
 "$ADVBENCH_DIR/gpt-3.5-turbo-0125_advbench_generated_multiturn_prompts_20-10-1722471013_600_700.json"
)

# echo ${MISSING_FNAMES[1]}


DIR1="/data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/"

# Define the pattern to search for within the file names
PATTERN="*.json"

# # Find all files that match the pattern in the directory and store them in a variable
FILES=($(find $DIR1 -type f -name "$PATTERN"))
echo $FILES

echo ${FILES[11]}

# # Define directories
# DIR1="/data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/"

# # Define the pattern to search for within the file names
# PATTERN="*.json"

# # Find all files that match the pattern in the directory and store them in a variable
# FILES=$(find $DIR1 -type f -name "$PATTERN")

# # Filter out all files with 'advbench' in the filename
# FILTERED_FILES=$(echo "$FILES" | grep -v "advbench")

# # Use the filtered list
# echo "Filtered files:"
# echo "$FILTERED_FILES"
# NUM_FILTERED_FILES=$(echo "$FILTERED_FILES" | grep -c '^')
# echo "Number of filtered files: $NUM_FILTERED_FILES"

# Filtered files:
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_18-52-1722552766_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2700_2800.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_22-47-1722566825_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_3000_3100.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_21-45-1722563122_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_800_900.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_23-52-1722570757_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_200_300.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_18-52-1722552768_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_600_700.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_20-13-1722557610_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_500_600.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_18-52-1722552766_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2900_3000.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_19-04-1722553487_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1400_1500.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_21-51-1722563511_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_100_200.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_20-35-1722558903_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2200_2300.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_21-35-1722562511_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1700_1800.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_21-17-1722561421_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_300_400.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_17-46-1722548762_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_900_1000.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_20-53-1722559984_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2100_2200.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_19-04-1722553487_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1000_1100.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_18-52-1722552766_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1300_1400.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_16-48-1722545315_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_600_700.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_23-14-1722568451_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_700_800.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_21-10-1722561005_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_3100_3200.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_16-48-1722545314_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2700_2800.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_21-52-1722563546_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_3200_3300.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_20-14-1722557669_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1200_1300.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_18-52-1722552766_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1600_1700.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_22-00-1722564028_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_3500_3600.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_00-45-1722573943_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2400_2500.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_17-57-1722549428_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1800_1900.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_18-06-1722549978_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1500_1600.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_04-00-1722585644_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_3200_3300.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_18-52-1722552766_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1800_1900.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_18-52-1722552766_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1500_1600.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_20-52-1722559927_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2600_2700.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_01-33-1722576784_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2800_2900.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_16-48-1722545314_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2900_3000.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_03-38-1722584325_input_gpt-3.5-turbo-0125_openai_generated_multiturn_prompts_20-09-1722470963_100_135.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_19-36-1722555388_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1400_1500.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_02-53-1722581587_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2500_2600.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_21-50-1722563444_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_600_700.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_20-26-1722558404_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_3300_3400.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_23-09-1722568168_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2000_2100.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_18-52-1722552766_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1100_1200.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_16-48-1722545314_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1300_1400.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_05-04-1722589488_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_3300_3400.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_21-00-1722560410_input_gpt-3.5-turbo-0125_openai_generated_multiturn_prompts_20-09-1722470963_0_100.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_21-39-1722562774_input_gpt-3.5-turbo-0125_openai_generated_multiturn_prompts_20-09-1722470963_100_135.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_18-31-1722551505_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1600_1700.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_18-52-1722552766_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_400_500.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_19-50-1722556223_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_3100_3200.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_18-52-1722552766_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_900_1000.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_19-40-1722555602_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1000_1100.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_19-55-1722556510_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_3000_3100.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_20-55-1722560142_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1900_2000.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_19-56-1722556596_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2300_2400.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_21-48-1722563288_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_500_600.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_20-03-1722556980_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_100_200.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_20-50-1722559814_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2400_2500.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_20-50-1722559806_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_200_300.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_20-54-1722560061_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2800_2900.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_01-25-1722576328_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2100_2200.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_20-54-1722560077_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2500_2600.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_19-51-1722556282_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1700_1800.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_19-50-1722556205_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1900_2000.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_16-48-1722545314_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_0_100.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_23-08-1722568127_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2200_2300.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_21-52-1722563561_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_3400_3500.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_19-52-1722556323_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_800_900.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_04-14-1722586451_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_3400_3500.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_19-45-1722555933_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1200_1300.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_18-52-1722552766_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_0_100.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_19-51-1722556282_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_300_400.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_20-35-1722558933_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2000_2100.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_18-08-1722550091_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_1100_1200.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_19-45-1722555932_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_500_600.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_03-45-1722584706_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_100_200.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_19-14-1722554050_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_400_500.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_20-49-1722559779_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_700_800.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_01-19-1722575942_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2600_2700.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_04-28-1722587301_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_2300_2400.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Mistral-7B-Instruct-v0.1_test_evaluated_multiturn_responses_19-52-1722556361_input_gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036_3500_3600.json
# /data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/Meta-Llama-3.1-8B-Instruct_test_evaluated_multiturn_responses_03-01-1722582088_input_gpt-3.5-turbo-0125_openai_generated_multiturn_prompts_20-09-1722470963_0_100.json
# Number of filtered files: 79