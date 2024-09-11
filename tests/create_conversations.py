from fastchat.model.model_adapter import get_conversation_template
from redteam.utils.data_utils import read_json


def get_attacker_message(raw_msg):
    conv = get_conversation_template("gpt-4")
    goal = raw_msg["goal"]
    conv.append_message(role="user", message=goal)

    for d in range(len(raw_msg["conversation"])):
        if d % 2 == 0:
            conv.append_message(
                role="assistant", message=raw_msg["conversation"][d]["content"]
            )
        else:
            conv.append_message(role="user", message=raw_msg["conversation"][d]["content"])
    return conv.to_openai_api_messages()[1:]  # remove system prompt


def get_defender_message(raw_msg):
    conv = get_conversation_template("gpt-4")
    for d in range(len(raw_msg["conversation"])):
        if d % 2 == 0:
            conv.append_message(role="user", message=raw_msg["conversation"][d]["content"])
        else:
            conv.append_message(
                role="assistant", message=raw_msg["conversation"][d]["content"]
            )
    return conv.to_openai_api_messages()[1:]  # remove system prompt


def main():
    pass


if __name__ == "__main__":
    EXAMPLE_FNAME = "/data/tir/projects/tir7/user_data/athankar/redteaming/data/gen_judge_multiturn_conversation/gpt-4_judge_generated_multiturn_conversations_22-00-1722564014.json"
    raw_data = read_json(EXAMPLE_FNAME)
    conv = get_conversation_template("gpt-3.5")

    attacker_messages = []
    defender_messages = []
    for raw_msg in raw_data:
        attacker_messages.append(get_attacker_message(raw_msg))
        defender_messages.append(get_defender_message(raw_msg))
