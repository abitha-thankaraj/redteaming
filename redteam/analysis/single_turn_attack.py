from tqdm import tqdm
from typing import List, Dict
from redteam.envs.common import Conversation
from redteam.inference.judge import LlamaGuardJudge
from redteam.utils.data_utils import write_json
from redteam.analysis.common import load_eval_and_config, load_target_lm


def get_single_turn_attack_questions(data: List):
    """For any jailbroken turn in a conversation, get the question asked by the attacker"""
    single_turn_attack_questions = []
    for record in data:
        if sum(record["judge"]["rewards"]) > 0.0:
            questions = []
            rewards = record["judge"]["rewards"]
            conversation = record["game"]
            for turn in conversation:
                if turn["role"] == "goal":
                    goal = turn["content"]
                if turn["role"] == "attacker":
                    questions.append(turn["content"])
            for q, r in zip(questions, rewards):
                if r > 0.0:
                    single_turn_attack_questions.append(q)
    return single_turn_attack_questions


def single_turn_attack(questions:List[str], defender, judge):
    """For each jailbroken turn in the original evaluation, 
        ask the defender the question asked by the attacker, 
        without the previous context
    Args:
        questions: List of jailbroken questions asked by the attacker
        defender: The target language model
        judge: LlamaGuard as judge; reward = 1 if jailbroken, 0 otherwise  
    """
    results = []
    for q in tqdm(questions):
        conv = Conversation(messages=[("goal", "")])
        conv.messages.append(("attacker", q))
        defender_response = defender.act(conv.to_defender_message())
        conv.messages.append(("defender", defender_response))
        judge_score = judge.score(conv.to_judge_input())

        assert (
            len(judge_score["rewards"]) == 1
        ), "Single turn attacks should only have one turn."

        results.append(
            {
                "questions": q,
                "game": conv.to_game_message(),
                "judge": judge_score,
            }
        )
    return results


def aggregate_results(results):
    rewards = []
    for result in results:
        rewards.append(result["judge"]["rewards"][0])
    return {
        "single_turn_attack_num_jailbreaks": sum(rewards),
        "single_turn_attack_num_evals": len(results),
    }


def run_single_turn_attack(fname: str):
    # Read the evaluation data and config
    evals, config = load_eval_and_config(fname)
    # Load target_lm from configs
    defender = load_target_lm(config)
    # Load the judge
    judge = LlamaGuardJudge(device=config.defender.device)

    # For every jailbroken conversation, get all the questions asked by the attacker.
    # This is the input to the concatenation attack
    single_turn_attack_questions = get_single_turn_attack_questions(evals)

    results = single_turn_attack(single_turn_attack_questions, defender, judge)
    aggregated_results = aggregate_results(results)

    write_json(results, "single_turn_attack_results.json")
    write_json(aggregated_results, "aggregated_single_turn_attack_results.json")
