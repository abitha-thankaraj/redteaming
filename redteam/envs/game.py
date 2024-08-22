import openai
import numpy as np
from gym.utils import seeding
from typing import Optional, Any
from redteam.common.chat_completion import ChatCompletionConfig, OAIChatCompletion
from redteam.data_generation.templates import GAME_JUDGE_PROMPTS
from redteam.data_generation.parsers import (
    parse_llm_judge_evaluation,
    is_valid_llm_judge_trace,
)
class LMAgent:
    def __init__(self, chat_completion, role) -> None:
        self.role = role
        self.chat_completion = chat_completion

    def act(self, obs: str) -> str:        
        response = self.chat_completion.single_turn_chat_completion(message = obs, history = True)
        return response[-1]["content"]

    def reset(self):
        self.chat_completion.reset()

    def _state(self):
        return self.chat_completion.conv.to_openai_api_messages()


# TODO: Get the <s,a,r,s'> tuples from the agent/ game


def get_lm_agent(config: ChatCompletionConfig, role: str) -> LMAgent:
    return LMAgent(OAIChatCompletion(config), role)


class RedteamGame:
    def __init__(
        self,
        seed: int,
        attacker: LMAgent,
        defender: LMAgent,
        judge: Optional[LMAgent],
        goals: list[str],
        max_turns: int = 3,
    ):

        self.goals = goals

        self.attacker = attacker
        self.defender = defender
        self.judge = judge
        self.max_turns = max_turns

        self._turn = 0

        self._np_random = None
        if seed is not None:
            self.seed(seed)
        self.env_state = []

    def seed(self, seed: Optional[int] = None) -> list:
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_model(self) -> None:
        self.attacker.reset()
        self.defender.reset()
        self._turn = 0

    def reset(self, goal:Optional[str] = None) -> str:
        self.reset_model()
        # Sample a goal for the attacker. Do you need to system rompt the attacker too?
        if goal is None:
            self._env_goal = self._np_random.choice(self.goals)
        else:
            self._env_goal = goal
        self.env_state = [{"role": "goal", "content": self._env_goal}]
        return self._env_goal

    def _update_env_state(self, attacker_action: str, defender_action: str) -> None:
        self.env_state.append(
            {
                "role": "attacker",
                "content": attacker_action,
            }
        )
        self.env_state.append(
            {
                "role": "defender",
                "content": defender_action,
            }
        )
        self._turn += 1

    def judge_conversation(self) -> int:
        self.judge.chat_completion.conv.set_system_message(GAME_JUDGE_PROMPTS["gpt-4"]["system"])
        message = GAME_JUDGE_PROMPTS["gpt-4"]["user_template"].format(goal = self._env_goal, conversation = self.env_state[1:])
        
        valid_judge_response = False
        i = 0
        while not valid_judge_response:
            response = self.judge.chat_completion.single_turn_chat_completion(
                message = message, history = False
            )
            print(response)
            llm_judge_trace = parse_llm_judge_evaluation(
                chat_completion_dict=response[-1],
            )
            valid_judge_response = is_valid_llm_judge_trace(
                llm_judge_trace=llm_judge_trace,
            )
            i+=1
            if i>5:
                response = None
                llm_judge_trace = None
                break
        self.judge.reset()
        return response, llm_judge_trace

    def simulate(self, judge = False):
        self.reset()
        defender_action = (
            self._env_goal
        )  # goal is always the first action; added to state in reset;
        while self._turn <= self.max_turns:  # goal gets counted as one turn
            attacker_action = self.attacker.act(defender_action)
            defender_action = self.defender.act(attacker_action)
            self._update_env_state(
                attacker_action=attacker_action, defender_action=defender_action
            )
        if judge:
            return self.env_state, self.judge_conversation()
        return self.env_state

    def render(self):
        print(self.env_state)


if __name__ == "__main__":
    attacker = LMAgent(lambda x: "a")
    defender = LMAgent(lambda x: "d")
    env = RedteamGame(seed=42, attacker=attacker, defender=defender, goals=["a", "b"])
    obs = env.reset()

    done = False
    while not done:
        action = env.attacker.act(obs)
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
