from tqdm import tqdm
from typing import List, Dict
from redteam.envs.common import Conversation
from redteam.inference.judge import LlamaGuardJudge
from redteam.utils.data_utils import read_json, write_json
from redteam.analysis.common import load_eval_and_config, load_target_lm


def get_concatenation_attack_questions(data: List):
    """For any jailbroken conversation, combine all the questions asked by the attacker"""
    concatenation_attack_questions = []
    for record in data:
        if sum(record["judge"]["rewards"]) > 0.0:
            questions = []
            conversation = record["game"]
            for turn in conversation:
                if turn["role"] == "goal":
                    goal = turn["content"]
                if turn["role"] == "attacker":
                    questions.append(turn["content"])
            concatenation_attack_questions.append((goal, " ".join(questions)))
    return concatenation_attack_questions


def concatenation_attack(concatenation_attack_questions: List[str], defender, judge):
    """Run the concatenation attack on the defender"""
    results = []
    for goal, qs in tqdm(concatenation_attack_questions):
        conv = Conversation(messages=[("goal", goal)])
        conv.messages.append(("attacker", qs))
        defender_response = defender.act(conv.to_defender_message())
        conv.messages.append(("defender", defender_response))
        judge_score = judge.score(conv.to_judge_input())
        results.append(
            {
                "goal": goal,
                "question": qs,
                "defender_response": defender_response,
                "judge": judge_score,
            }
        )
    return results


def aggregate_results(results: Dict):
    # Judge["rewards"] is a list of rewards for each turn in the conversation.
    # In this case, there is only one turn in the conversation.
    # The reward is 1 if the turn is a jailbroken, 0 otherwise.
    rewards = []
    for result in results:
        rewards.append(result["judge"]["rewards"][0])
    return {
        "concatenation_attack_num_jailbreaks": sum(rewards),
        "concatenation_attack_num_evals": len(results),
    }


def run_concatenation_attack(fname: str):
    # Read the evaluation data and config
    evals, config = load_eval_and_config(fname)
    # Load target_lm from configs
    defender = load_target_lm(config)
    # Load the judge
    judge = LlamaGuardJudge(device=config.defender.device)

    # For every jailbroken conversation, get all the questions asked by the attacker.
    # This is the input to the concatenation attack
    concatenation_attack_questions = get_concatenation_attack_questions(evals)

    results = concatenation_attack(concatenation_attack_questions, defender, judge)
    aggregated_results = aggregate_results(results)

    write_json(results, "concatenation_attack_results.json")
    write_json(aggregated_results, "aggregated_concatenation_attack_results.json")
