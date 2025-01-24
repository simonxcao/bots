---
layout: default
title: Proposal
---

## Summary of the Project
The main idea is to build an AI to beat a cuphead boss. However, this AI will require two machine-learning tasks. The first is to take images of the gameplay as input and it will produce object recognition through classification and labelling. The second task is with the labeled objects recognized as the input it will train a neural network again to output actions to beat the boss. The application this AI will be tested on is Cuphead, and we will have an account with the Cuphead boss unlocked and control for the same game state. We will utilize Python and tensorflow GPU locally to train our AI.

## AI/ML Algorithms
We anticipate using convolutional neural networks for object recognition via image classification. We will incorporate reinforcement learning with on-policy algorithms such as proximal policy optimization and reward signals such as positive rewards for a reduction in boss health and negative penalties for a decrease in player health.

## Evaluation Plam
The project's success will be quantitatively evaluated using two primary metrics: boss health reduction and player health preservation. For each test, we will calculate the percentage of the boss's health depleted by the AI and compare it to the player's health remaining. The baseline performance will be a random action agent, which is expected to deplete minimal boss health and lose all player health quickly. Our AI is expected to improve boss health reduction by at least 50% compared to the baseline while maintaining over 30% of player health on average. Evaluation will be conducted on consistent game states, ensuring that the boss, player attributes, and game environment are controlled and identical across trials.

To verify the project works, we will manually observe performance between intervals of several iterations. Internals of the algorithm, such as the object recognition output, will be visualized using overlays on gameplay videos to verify the accurate classification and labeling of objects. Additionally, we will analyze the AI's decision-making process by plotting action probabilities or Q-values over time to ensure logical consistency. The ultimate goal is for the AI to consistently beat the boss without losing any health and adapt to unexpected attack patterns or environmental changes, demonstrating robust and generalizable gameplay strategies. Achieving this would showcase the AI's capacity to handle complex decision-making in real-time gameplay.
