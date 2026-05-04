# Final Project: Modern Deep RL Exploration (Option C)

**Student:** Barsha Kakshapati  
**Date:** May 3, 2026  
**Course:** Data Science Practicum – Reinforcement Learning  

---

## 1. Overview
**Thesis:** The transition from tabular methods to Deep Reinforcement Learning (Deep RL) represents a fundamental shift from manual state engineering to automated feature extraction, effectively solving the "discretization ceiling" that limited performance in the CartPole environment[cite: 1].

### Synthesis of Insights
In earlier labs, exploration of **Tabular Q-Learning** and **Dyna-Q** revealed a significant "discretization ceiling"[cite: 1]. In those models, the agent’s world was partitioned into 5,184 discrete "buckets"[cite: 1]. While this provided a stable learning environment, it stripped away the nuance required for optimal control in a continuous state space like **CartPole-v1**[cite: 1]. This project leverages **Proximal Policy Optimization (PPO)** via the **Stable-Baselines3** library to demonstrate how neural networks serve as powerful function approximators that overcome these resolution limits[cite: 1].

A critical insight gained is the role of **representation learning**[cite: 1]. Unlike tabular methods that treat every discrete state as an independent island, modern Deep RL algorithms utilize shared neural layers to generalize experience across similar states[cite: 1]. This project maps these library implementations back to the core principles of **Sutton & Barto**, specifically the challenge of managing the **bias-variance trade-off** in non-linear function approximation[cite: 1].

### Bridging Theory and Practice
While textbooks often focus on the mathematical convergence of algorithms like **REINFORCE**, this study explores why "engineering tricks"—such as **Generalized Advantage Estimation (GAE)**, **clipping**, and **observation normalization**—are essential for stability in practice[cite: 1]. Findings suggest that without these library-specific optimizations, even the most mathematically sound policy gradient methods struggle with high variance and catastrophic forgetting[cite: 1].

---

## 2. Code Description
### Implementation & Architectural Choices
This exploration utilized **Stable-Baselines3** to implement **PPO** on the **CartPole-v1** environment[cite: 1]. The architecture consists of a **Multi-Layer Perceptron (MLP)** with two hidden layers of 64 nodes each, using **ReLU** activation functions[cite: 1]. This choice provides enough capacity to model the non-linear relationship between the four continuous state features without over-fitting to early training noise[cite: 1].

### Key Hyperparameters
*   **Learning Rate ($3 \times 10^{-4}$):** Selected to balance convergence speed with stability[cite: 1].
*   **Clip Range (0.2):** A PPO-specific parameter that prevents the policy from moving too far in a single update, ensuring a "trust region"[cite: 1].
*   **Discount Factor ($\gamma = 0.99$):** Ensures the agent prioritizes long-term pole stability over immediate rewards[cite: 1].

### Experimental Setup
The setup involved running **10 independent trials** for each configuration to calculate **standard error**[cite: 1]. I implemented **Observation Normalization**, which scales continuous inputs to a mean of zero and standard deviation of one, significantly improving the convergence rate compared to unscaled data[cite: 1].

---

## 3. Results
### Performance Comparison
The PPO agent demonstrated superior **sample efficiency**, reaching the maximum stable return of **500.0** within approximately 20,000 steps[cite: 1].

| Configuration | Mean Return (800 Episodes) | Performance Impact |
| :--- | :--- | :--- |
| **Full PPO Model** | **492.4** | **Baseline**[cite: 1] |
| **Without GAE** | 315.2 | -36% (Increased Variance)[cite: 1] |
| **Without Obs. Normalization** | 185.6 | -62% (Slowed Convergence)[cite: 1] |

### Learning Dynamics
*   **Convergence Analysis:** Deep RL curves showed higher initial variance but reached a significantly higher peak return than the tabular baseline[cite: 1].
*   **Hyperparameter Sensitivity:** Testing learning rates revealed that while $10^{-3}$ learns faster, it eventually destabilizes. Conversely, $10^{-4}$ achieved the best long-term stability and consistent convergence[cite: 1].

---

## 4. Reflection
### The Gap Between Textbook and Practice
The most significant lesson learned is the extent of the **gap between textbook algorithms and practical RL**[cite: 1]. In **Sutton & Barto**, REINFORCE is presented with elegant simplicity[cite: 1]. However, in a deep implementation, these models are surprisingly "brittle"[cite: 1]. Without library-specific "engineering tricks" like **gradient clipping**, the agent often enters an "unrecoverable state" where the policy collapses[cite: 1].

### Theoretical Connections
The underperformance of **Dyna-Q** in earlier labs was particularly surprising[cite: 1]. This connected back to the theory of **model errors**; because the Dyna-Q model relied on flawed discretization, "planning" with virtual experience only reinforced existing errors[cite: 1]. This project changed my understanding of RL from a "search for the best algorithm" to a **"search for the best representation"**[cite: 1].

---

## References
*   **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press[cite: 1].
*   **Schulman, J., et al. (2017).** "Proximal Policy Optimization Algorithms." *arXiv preprint arXiv:1707.06347*[cite: 1].
*   **Raffin, A., et al. (2021).** "Stable-Baselines3: Reliable Reinforcement Learning Implementations." *Journal of Machine Learning Research*[cite: 1].
