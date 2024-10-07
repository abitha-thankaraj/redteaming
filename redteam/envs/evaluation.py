from gym.utils import seeding
from typing import Optional, Any
from redteam.envs.common import Conversation
from redteam.train.datasets import get_value_function_keywords, get_reward_to_gos


class Game:
    def __init__(self, seed, attacker, defender, judge, goals=None, max_turns=3):
        self.seed(seed)
        self.attacker = attacker
        self.defender = defender
        self.judge = judge

        self.goals = goals
        self._turn = 0
        self._env_goal = None
        self.env_state = None
        self.max_turns = max_turns

    def seed(self, seed: Optional[int] = None) -> list:
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_model(self, goal):
        self._turn = 0
        self.env_state = Conversation(messages=[("goal", goal)])

    def reset(self, goal: Optional[str] = None):
        if goal is None:
            self._env_goal = self._np_random.choice(self.goals)
        else:
            self._env_goal = goal
        self.reset_model(self._env_goal)
        return self._env_goal

    def simulate_one_turn(self):
        attacker_action = self.attacker.act(self.env_state.to_attacker_message())
        self.env_state.messages.append(("attacker", attacker_action))
        defender_action = self.defender.act(self.env_state.to_defender_message())
        self.env_state.messages.append(("defender", defender_action))
        self._turn += 1

    def simulate_n_turns(self, n):
        for _ in range(n):
            self.simulate_one_turn()

    def simulate(self):
        while self._turn < self.max_turns:
            self.simulate_one_turn()
        # traj = self.env_state.to_game_message()
        # Remove the goal. Pass in as openai format for judge
        judge_score = self.judge.score(self.env_state.to_judge_input())
        return self.env_state, judge_score

    def repair(self, goal, conversation, original_judge_rewards, n=5):
        self.reset_model(goal=goal)
        judge_score = None

        for i, turn in enumerate(conversation):

            if turn["role"] == "user":
                self.env_state.messages.append(("attacker", turn["content"]))

            if turn["role"] == "assistant":
                judge_idx = i // 2
                judge_reward = original_judge_rewards[judge_idx]
                last_reward = None

                if judge_reward == 1.0: #flipped for defender
                    # Non jailbroken turn
                    self.env_state.messages.append(("defender", turn["content"]))
                else:
                    # Jailbroken turn
                    for _ in range(n):
                        self.env_state.messages.append(("defender", 
                            self.defender.act(self.env_state.to_defender_message())))
                        judge_score = self.judge.score(self.env_state.to_judge_input())
                        last_reward = judge_score["rewards"][-1]

                        if last_reward == 0.0:
                            break
                        else:
                            self.env_state.messages.pop(-1)

                    if last_reward == 1.0:
                        # break if no repair found
                        return None, None

        if judge_score is not None:
            if sum(judge_score["rewards"]) != 0.0:
                return None, None
        return self.env_state, [0.0] * len(original_judge_rewards)

    def prefilled_response(self, goal, conversation, original_judge_rewards, prefill_text):
        self.reset_model(goal=goal)
        judge_score = None

        for i, turn in enumerate(conversation):

            if turn["role"] == "user":
                self.env_state.messages.append(("attacker", turn["content"]))

            if turn["role"] == "assistant":
                judge_idx = i // 2
                judge_reward = original_judge_rewards[judge_idx]
                last_reward = None

                if judge_reward == 1.0: #flipped for defender. you flip this in the script
                    # Non jailbroken turn
                    self.env_state.messages.append(("defender", turn["content"]))
                else:
                    # Jailbroken turn
                    self.env_state.messages.append(("defender", 
                        self.defender.act_prefilled(self.env_state.to_defender_message(), prefill_text)))
                    judge_score = self.judge.score(self.env_state.to_judge_input())
        
        return self.env_state, judge_score

def evaluate_value_function(conversation, value_function_type, rewards, policy_type):
    if policy_type == "defender":
        # flip rewards
        rewards = [1.0 - r for r in rewards]
    # Calculate the ground truth value function
    if value_function_type in ["binary", "natural_lang_binary"]:
        rews = rewards
    elif value_function_type in ["multilabel", "natural_lang_multiclass"]:
        rews = get_reward_to_gos(rewards, gamma=0.9)
    else:
        raise NotImplementedError(f"Value function type {value_function_type} not implemented")

    value_function_token_map = get_value_function_keywords(
        value_function_type, gamma=0.9, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    gt_vfs = []
    is_correct_value_function_predictions = []
    for i, rew in enumerate(rews):
        gt_vfs.append(value_function_token_map[rew])
        # get assistant turn
        assert conversation[i * 2 + 1]["role"] == "assistant", "Assistant turn not found"
        # Did the assistant predict the correct value function for future states?
        is_correct_value_function_predictions.append(
            conversation[i * 2 + 1]["content"]
            .strip()
            .startswith(value_function_token_map[rew])
        )

    return {
        "ground_truth_value_functions": gt_vfs,
        "is_correct_value_function_predictions": is_correct_value_function_predictions,
    }
