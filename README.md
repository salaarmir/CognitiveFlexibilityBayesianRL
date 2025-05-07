# Cognitive Flexibility in Bayesian Reinforcement Learning

This repository contains the code for my MSc thesis in Computational Neuroscience at the University of Nottingham. The project investigates the impact of over-training on cognitive flexibility in Bayesian reinforcement learning (RL) models, replicating the overtraining reversal effect (ORE) in reversal learning tasks. This work aims to provide insights into the development of habitual behaviours and the adaptive limitations of over-trained agents.

## Project Goal

To model and replicate cognitive flexibility in reinforcement learning agents, capturing the ability to rapidly adapt to changing environments, with a specific focus on the effects of over-training on task performance.

## Key Observations from Simulations

* **Reversal Learning Task**: Agents were trained on a two-choice task with periodically reversing reward structures, requiring rapid adaptation to changing conditions.
* **Impact of Over-Training**: Over-trained agents, exposed to an additional 100 trials under a fixed rule before reversal, developed more rigid, habitual behaviors, making them slower to adapt when the reward structure changed.
* **Decay Factor (γ)**: Higher decay rates led to slower adaptation as the agent relied more on past experiences, while lower decay rates promoted faster but less stable learning, favoring recent experiences over older ones.
* **Noise Level (η)**: Higher noise levels encouraged exploration and prevented overfitting to a specific strategy, while lower noise levels resulted in more predictable, conservative behaviors.

## Usage

To run a full simulation and generate results:

```bash
python main.py
```

To visualize the results:

```bash
python createPlots.py
```

## Relevance to Cognitive Flexibility Research

This project explores the impact of over-training on cognitive flexibility in Bayesian RL agents. It highlights how over-training can lead to the development of rigid, habitual behaviors, reducing the ability to adapt to new conditions. This has important implications for understanding both artificial and biological learning systems.

## Future Work

The current study provides an initial exploration of cognitive flexibility and the impact of over-training on learning adaptability using Bayesian RL models. Future research could include:

* **Meta-Learning Capabilities**: Developing RL models with better meta-learning capabilities, allowing agents to adjust their learning strategies based on accumulated experience. This could include algorithms that recognize patterns in task structures, accelerating the learning process over multiple reversals.
* **Dynamic Parameter Adjustment**: Implementing adaptive tuning for parameters like noise levels and decay factors, allowing agents to fine-tune their exploration-exploitation balance based on performance, potentially improving long-term adaptability.
* **More Complex Task Environments**: Incorporating more complex and realistic task environments, including multi-rule settings or stochastic reversal tasks, where rewards are provided probabilistically, forcing more frequent strategy adjustments and better approximating real-world learning scenarios.

## Contributors

Salaar Mir (Primary Developer)

## License

This project is released under the MIT License.
