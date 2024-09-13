import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gym.utils import seeding
from typing import Optional, Any
from redteam.envs.common import GameConversation
import logging

logger = logging.getLogger(__name__)


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
        self.env_state = GameConversation(messages=[("goal", goal)])

    def reset(self, goal: Optional[str] = None):
        if goal is None:
            self._env_goal = self._np_random.choice(self.goals)
        else:
            self._env_goal = goal
        self.reset_model(self._env_goal)
        return self._env_goal

    def simulate_one_turn(self):
        print(f"Attacker message: {self.env_state.to_attacker_message()}")
        attacker_action = self.attacker.act(self.env_state.to_attacker_message())
        self.env_state.messages.append(("attacker", attacker_action))
        print("*********************************************************")
        print(f"Defender message: {self.env_state.to_defender_message()}")
        defender_action = self.defender.act(self.env_state.to_defender_message())
        self.env_state.messages.append(("defender", defender_action))
        self._turn += 1

    def simulate_n_turns(self, n):
        for _ in range(n):
            self.simulate_one_turn()
    
    def simulate(self):
        while self._turn < self.max_turns:
            self.simulate_one_turn()
        traj = self.env_state.to_game_message()
        # Remove the goal. Pass in as openai format for judge
        judge_score = self.judge.score(self.env_state.to_defender_message())
        return traj, judge_score
