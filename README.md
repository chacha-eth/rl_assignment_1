# 🏆 Grid World RL Simulation

This project is a **Reinforcement Learning (RL) simulation** in a **Grid World environment**.  
The agent learns to **navigate toward a goal position** using **two RL algorithms**:

1. **Value Iteration** → A **Dynamic Programming** approach that iteratively computes an optimal policy.
2. **Q-Learning** → A **Model-Free RL** algorithm that updates Q-values based on the agent’s experience.

## **🔹 How the Simulation Works**
1. **Choose an RL Algorithm** (Value Iteration or Q-Learning).
2. **Set the Grid Size and Goal Position**.
3. **Train the Agent** → The model learns an optimal policy.
4. **Start the Simulation** → The agent follows the trained policy.
5. **Stop the Simulation Anytime**.

---

## **📌 Installation**
Make sure you have **Python 3.8+** installed.  
Clone this repository and install dependencies:
```bash
git clone https://github.com/chacha-eth/rl_assignment_1
cd rl_assignment_1
pip install -r requirements.txt

## **🚀 Start the Program**
Once you have installed the dependencies, run the simulation using **Streamlit**:
```bash
streamlit run app.py
```
![Optimal Policy Grid](/figures/figure2.png)

![Agent Moving](/figures/figure1.png)
