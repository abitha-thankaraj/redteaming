Require: Set of harmful goals, reward_model, model, seed examples of successful multiturn attacks
Data generation: 
    Attacker dataset: Use ICL examples of seed dataset with GPT3.5 to generate multiturn attacks dataset.
    Train a language model to generate attacks contextually for multi turn conversation.
    Attacker(Goal) = [harmful questions]

    Safety tuning dataset: 
    for goal in goals:
        rollout \pi_a(goal) on \pi_d for n turns
        D = D \union rollout
    




Algo:
    Group collected trajectories by goal.Use rewrad model to determine if the model was broken or not.
    Pair broken as -ve with good trajectories by goal at +ve.

    Data augmentation:
        For each -ve trajectory: do best-of-N rejection sampling withgrounf truth reward model.
        if trajectory is successful, add to dataset.

    


    


Generate multiturn attacks using a \pi_a(goal)
rollout \pi_a(goal) to \pi_defender()/.
classify each response as safe/ unsafe using reward model/ judge. 