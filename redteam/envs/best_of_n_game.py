from openai import OpenAI
import logging
import numpy as np
from gym.utils import seeding
from typing import Optional, Any, Union, List
from redteam.common.chat_completion import ChatCompletionConfig

logger = logging.getLogger(__name__)

class GameConversation:
    def __init__(self, system_message: str = "", messages=None):
        self.messages = messages if messages is not None else []
        self.system_message = system_message

    def set_system_message(self, system_message: str):
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        self.messages.append((role, message))

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def _format_messages(self, system_role, user_role, assistant_role, offset=0):
        """Convert the conversation to OpenAI chat completion format."""
        if self.system_message == "":
            ret = []
        else:
            ret = [{"role": system_role, "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[offset:]):
            if i % 2 == 0:
                ret.append({"role": user_role, "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": assistant_role, "content": msg})
        return ret

    def to_openai_api_messages(self, offset=0):
        """Convert the conversation to OpenAI chat completion format."""
        return self._format_messages("system", "user", "assistant", offset=offset)

    def to_game_message(self):
        """Convert the conversation to a game message."""
        ret = []
        for role, msg in self.messages:
            ret.append({"role": role, "content": msg})

# TODO : Make abstraction later
# class LanguageModelAgent:
#     def __init__(self) -> None:


class Attacker:
    def __init__(self, config: ChatCompletionConfig):
        self.client = OpenAI(api_key="EMPTY", base_url=config.url)
        self.config = config

    def act(self, conversation: GameConversation):
        return self.sample_n_responses(conversation)

    def sample_n_responses(self, conversation: GameConversation):
        """Get responses from the model."""
        responses = []
        for i in range(self.config.n):
            res = self.client.chat.completions.create(
                model=self.config.model,
                messages=conversation.to_openai_api_messages(),
                temperature=self.config.temperature,
                n=1,
            )
            responses.append(res.choices[0].message.content)
        return responses
    # Attacker action : Sample n to do best of N
    def sample_n_responses_batched(self, conversation: GameConversation):
        """Get responses from the model."""
        res = self.client.chat.completions.create(
            model=self.config.model,
            messages=conversation.to_openai_api_messages(),
            temperature=self.config.temperature,
            n=self.config.n,
        )
        return [res.choices[i].message.content for i in range(self.config.n)]


class Defender:
    def __init__(self, config: ChatCompletionConfig):
        self.client = OpenAI(api_key="EMPTY", base_url=config.url)
        self.config = config

    def act(self, conversation: GameConversation, attacker_samples):
        return self.get_defender_responses(conversation, attacker_samples)

    # Defender action
    def get_defender_responses(self, conversation, attacker_samples):
        """Get responses from the model."""
        responses = []
        for attacker_sample in attacker_samples:
            # Ignore the first message; this flips it from attacker to defender
            messages = conversation.to_openai_api_messages(offset=1)
            messages.append({"role": "user", "content": attacker_sample})
            res = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                n=1,
            )
            del messages
            responses.append(res.choices[0].message.content)
        return responses
    
    # def get_batched_defender_response(self, conversation, attacker_samples):
    #     """Get responses from the model."""
    #     messages = conversation.to_openai_api_messages(offset=1)
    #     messages.append({"role": "user", "content": attacker_samples})
    #     res = self.client.chat.completions.create(
    #         model=self.config.model,
    #         messages=messages,
    #         temperature=self.config.temperature,
    #         n=1,
    #     )
    #     from IPython import embed; embed()
    #     return res.choices[0].message.content

# Judge action
def calculate_scores(judge, attacker_actions, defender_actions):
    """Get responses from the model."""
    rewards, categories = [], []
    for i, (attacker_action, defender_action) in enumerate(
        zip(attacker_actions, defender_actions)
    ):
        turn_level_scores = judge.score(
            conversation=[
                {"role": "user", "content": attacker_action},
                {"role": "assistant", "content": defender_action},
            ]
        )
        logger.debug(f"Turn {i} scores: {turn_level_scores}")
        assert len(turn_level_scores["rewards"]) == 1, "Judge should return a single reward per turn."
        rewards.append(turn_level_scores["rewards"][0])
        categories.append(turn_level_scores["categories"][0])
    return rewards, categories


def rejection_sampling(judge, attacker_responses, defender_responses):
    """Get responses from the model."""
    # Sample actor responses
    # attacker_responses = sample_n_responses(attacker, conversation, attacker_config)
    # # Get defender responses. Fixed. sample one response from the defender.
    # defender_responses = get_defender_responses(defender, conversation, defender_config, attacker_responses)
    # Rejection sampling from best of n. score using llamaguard
    rewards, categories = calculate_scores(judge, attacker_responses, defender_responses)
    # Pick the first response that given reward =0.0 ie is_safe.
    # If all are unsafe, pick the last one.
    chosen_idx = -1
    for i, reward in enumerate(rewards):
        if reward == 0.0:  # TODO: Check if this is safe
            chosen_idx = i
    chosen = {
        "attacker": attacker_responses[chosen_idx],
        "defender": defender_responses[chosen_idx],
        "reward": rewards[chosen_idx],
        "category": categories[chosen_idx],
    }            
    all_responses = {
        "attacker_responses": attacker_responses,
        "defender_responses": defender_responses,
        "rewards": rewards,
        "categories": categories,
    }
    return chosen, all_responses

    

class BestOfNRedteamingGame:
    def __init__(
        self,
        seed: int,
        attacker: Any,
        defender: Any,
        judge: Optional[Any],
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

        self.env_state = {}

    def seed(self, seed: Optional[int] = None) -> list:
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, goal=None):
        self._turn = 0
        del self.env_state
        self.env_state = {}
        if goal is None:
            self.goal = self._np_random.choice(self.goals)
        else:
            self.goal = goal
        return self.goal

    def simulate(self, goal):
        self.reset(goal)
        logger.debug(f"Goal: {self.goal}")
        assert self.goal == goal, "Goal should be same as reset"
        conversation = GameConversation(system_message="")
        conversation.append_message("goal", goal)
        rewards, categories = [], []
        while self._turn <= self.max_turns:
            logger.debug(f"Turn {self._turn}")
            logger.debug("Sampling {self.attacker.config.n} responses from attacker")
            attacker_actions = self.attacker.act(conversation)
            logger.debug("Getting responses from defender")
            defender_actions = self.defender.act(conversation, attacker_actions)
            logger.debug("Calculating scores for rejection sampling")


            chosen, all_responses= rejection_sampling(
                self.judge,
                attacker_responses=attacker_actions,
                defender_responses=defender_actions,
            )
            conversation.append_message("attacker", chosen["attacker"])
            conversation.append_message("defender", chosen["defender"])
            rewards.append(chosen["reward"])
            categories.append(chosen["category"])
            self.env_state.update({f"turn_{self._turn}": all_responses})
            self._turn+=1
        self.env_state.update({"game_conversation": conversation.to_game_message(),
            "game_rewards": rewards,
            "game_categories": categories})
        
        return self.env_state

    def render(self):
        print(self.env_state)

