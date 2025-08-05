# Offline Reinforcement Learning for Autonomous Driving (FATE-Aligned)

## 🚗 Project Overview

This project simulates **Offline Reinforcement Learning (OfflineRL)** for autonomous driving using pre-recorded driving sequences and expert trajectories. It incorporates **Batch-Constrained Q-Learning** principles and includes **fairness-aware reward shaping**, inspired by Microsoft's [FATE framework](https://www.microsoft.com/en-us/research/group/fate/).

It also includes a neural policy model trained using **Scikit-learn’s MLPClassifier** to mimic expert actions, forming the basis for policy learning from logged trajectory data.

---

## 🧠 Key Concepts

* **OfflineRL**: No real-time interaction with the environment — all training happens on pre-collected data.
* **Fairness-aware Rewards**: Extra incentives for socially beneficial decisions, e.g., slowing down near pedestrians.
* **Neural Policy Learning**: Use of a supervised classifier to model the driving policy.
* **Simulation-First Approach**: Uses synthetic driving data with velocity, obstacle distance, lane offset, pedestrian proximity, and traffic signals.

---

## 📦 Dependencies

```bash
pip install numpy matplotlib scikit-learn
```

---

## 🚀 How to Run (Colab / Local)

1. Copy or clone the Python file: `offline_rl_finance.py`
2. Run the file in Jupyter, VSCode, or Colab.
3. You’ll see:

   * Action distribution (fairness check)
   * Classification report from neural policy
   * Inference output
   * Reward distribution histogram

---

## 📝 Example Output

```bash
[Policy Action Distribution]
Maintain: 578 times (57.80%)
Slow Down: 243 times (24.30%)
Switch Lane: 179 times (17.90%)

[Neural Policy Evaluation]
Confusion Matrix:
[[...]]
Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.91      0.89       ...
           1       0.77      0.73      0.75       ...
           2       0.74      0.67      0.70       ...
```

---

## 📚 Dataset Note

This simulation uses synthetic data. For real use cases:

* Use the **Waymo Open Dataset**, **CARLA simulator**, or **D4RL** environments
* Replace the data generator with a parser for trajectory logs

---

## 🧠 Research Extensions

* Replace the MLPClassifier with a PyTorch actor-critic model
* Integrate with `d3rlpy` for full Offline RL agent training
* Add fairness metrics (e.g., action fairness by demographic subgroup)

---

## 👩‍💻 Author

This project was created by Yukta Chikate as a research simulation aligned with NVIDIA and Microsoft Research goals in fairness, agentic workflows, and RL.

> Inspired by Microsoft FATE, Waymo autonomous driving, and OfflineRL best practices.
