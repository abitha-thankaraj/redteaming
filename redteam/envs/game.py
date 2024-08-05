import numpy as np
from gym.utils import seeding
from typing import Optional, Any
from redteam.common.chat_completion import ChatCompletionConfig, OAIChatCompletion


class LMAgent:
    def __init__(self, chat_completion, role) -> None:
        self.role = role
        self.chat_completion = chat_completion

    def act(self, obs: Any[str]) -> str:
        self.messages.append(obs)  # Add conversation utterance
        response = self.chat_completion.multiturn_chat_completion(messages=self.messages)[-1][
            "response"
        ]
        self.messages.append(response)  # Add response
        return response

    def reset(self):
        self.messages = []

    def _state(self):
        return self.messages


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

    def reset(self) -> str:
        self.reset_model()
        # Sample a goal for the attacker. Do you need to system rompt the attacker too?
        self._env_goal = self._np_random.choice(self.goals)
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

    def _get_reward(self) -> int:
        if self.judge:
            # make judge ask
            pass

    def simulate(self):
        self.reset()
        defender_action = (
            self._env_goal
        )  # goal is always the first action; added to state in reset;

        while self._turn <= self.max_turns:  # goal gets counted as one turn
            # simulate one turn
            attacker_action = self.attacker.act(defender_action)
            defender_action = self.defender.act(attacker_action)
            self._update_env_state(
                attacker_action=attacker_action, defender_action=defender_action
            )
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
