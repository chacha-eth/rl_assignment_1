# import streamlit as st
# import time
# import numpy as np
# from value_iteration import ValueIteration
# from q_learning import QLearning

# # Default settings
# DEFAULT_GRID_SIZE = 10
# DEFAULT_GOAL = (9, 9)
# ALGORITHMS = ["Value Iteration", "Q-Learning"]

# # Streamlit UI
# st.title("Grid World Simulation")

# st.markdown("""
# ### About This Simulation
# This simulation demonstrates **Reinforcement Learning (RL) in a Grid World environment**.  
# The agent learns to reach a **goal position** while avoiding unnecessary movements.  

# You can choose between two RL algorithms:
# - **Value Iteration**: A Dynamic Programming approach that iteratively improves state values.
# - **Q-Learning**: A Model-Free approach that updates Q-values based on agent experiences.

# ### How It Works
# 1. **Select the RL algorithm**, grid size, and goal position.
# 2. **Train the model** to compute the optimal policy.
# 3. **Start the simulation**, and the agent will navigate using the learned policy.
# 4. **Stop the simulation anytime** using the stop button.

# Try it out and see how the agent learns to reach the goal efficiently! 🚀
# """)

# # Ensure grid_size is stored in session state
# if "grid_size" not in st.session_state:
#     st.session_state.grid_size = DEFAULT_GRID_SIZE

# # User selects Grid Size
# grid_size = st.number_input("Grid Size (Min: 5, Max: 20)", min_value=5, max_value=20, value=st.session_state.grid_size, step=1)
# st.session_state.grid_size = grid_size  # Store updated grid size

# # Ensure goal state remains within grid size
# if "goal_x" not in st.session_state or st.session_state.goal_x >= grid_size:
#     st.session_state.goal_x = min(DEFAULT_GOAL[0], grid_size - 1)
# if "goal_y" not in st.session_state or st.session_state.goal_y >= grid_size:
#     st.session_state.goal_y = min(DEFAULT_GOAL[1], grid_size - 1)

# # User selects Goal Position
# goal_x = st.number_input("Goal X Position (0 to Grid Size - 1)", min_value=0, max_value=grid_size - 1, value=st.session_state.goal_x)
# goal_y = st.number_input("Goal Y Position (0 to Grid Size - 1)", min_value=0, max_value=grid_size - 1, value=st.session_state.goal_y)
# st.session_state.goal_x, st.session_state.goal_y = goal_x, goal_y

# goal_state = (goal_x, goal_y)

# # Ensure session state for policy and agent
# if "policy" not in st.session_state:
#     st.session_state.policy = np.full((grid_size, grid_size), ' ')
# if "agent" not in st.session_state:
#     st.session_state.agent = None

# # Train Model Button
# if st.button("Train Model"):
#     if st.session_state.goal_x >= grid_size or st.session_state.goal_y >= grid_size:
#         st.error("Goal state is outside the grid! Please adjust.")
#     else:
#         if st.selectbox("Select Algorithm", ALGORITHMS) == "Value Iteration":
#             agent = ValueIteration(grid_size, goal_state)
#             agent.run_value_iteration()
#             st.session_state.policy = agent.policy
#         else:
#             agent = QLearning(grid_size, goal_state)
#             agent.train_q_learning()
#             policy = np.full((grid_size, grid_size), ' ')
#             for i in range(grid_size):
#                 for j in range(grid_size):
#                     best_action = ["U", "D", "L", "R"][np.argmax(agent.q_table[i, j])]
#                     policy[i, j] = best_action
#             st.session_state.policy = policy

#         st.session_state.agent = agent  # Store trained model
#         st.success("Training Complete! Click 'Start Simulation' to begin.")

# # Simulation Controls
# start_simulation = st.button("Start Simulation")
# stop_simulation = st.button("Stop Simulation")

# # Display Policy Grid
# st.write("### Optimal Policy (Agent's Movement Direction)")
# grid_placeholder = st.empty()
# CELL_SIZE = 40

# def render_grid(state):
#     """ Render grid in Streamlit with agent movement """
#     grid_html = "<table style='border-collapse: collapse;'>"
    
#     for i in range(grid_size):
#         grid_html += "<tr>"
#         for j in range(grid_size):
#             cell_color = "white"
#             cell_text = st.session_state.policy[i, j]  # Use stored policy

#             if (i, j) == state:
#                 cell_color = "red"
#                 cell_text = "A"
#             elif (i, j) == goal_state:
#                 cell_color = "green"
            
#             grid_html += f"<td style='width: {CELL_SIZE}px; height: {CELL_SIZE}px; border: 1px solid black; background-color: {cell_color}; text-align: center; font-weight: bold;'>{cell_text}</td>"
        
#         grid_html += "</tr>"
    
#     grid_html += "</table>"
#     grid_placeholder.markdown(grid_html, unsafe_allow_html=True)

# # Run Simulation only if Start is pressed
# if start_simulation:
#     if start_simulation:
#         if st.session_state.agent is None:
#             st.error("Please train the model first!")
#         else:
#             state = (0, 0)
#             st.write(f"### Simulation Running using **{st.session_state.algorithm}**...")

#         for _ in range(grid_size * grid_size):
#             if stop_simulation:  # Stop immediately if user clicks "Stop Simulation"
#                 st.warning("Simulation Stopped!")
#                 break
            
#             render_grid(state)
#             time.sleep(0.3)

#             best_action = st.session_state.policy[state[0], state[1]]
#             if best_action in ["U", "D", "L", "R"]:
#                 state = st.session_state.agent.get_next_state(state, best_action)

#             if state == goal_state:
#                 render_grid(state)
#                 st.success("Agent Reached the Goal!")
#                 break
import streamlit as st
import time
import numpy as np
from value_iteration import ValueIteration
from q_learning import QLearning

# Default settings
DEFAULT_GRID_SIZE = 10
DEFAULT_GOAL = (9, 9)
ALGORITHMS = ["Value Iteration", "Q-Learning"]

# Streamlit UI
st.title("Grid World Simulation")

st.markdown("""
### About This Simulation
This simulation demonstrates **Reinforcement Learning (RL) in a Grid World environment**.  
The agent learns to reach a **goal position** while avoiding unnecessary movements.  

You can choose between two RL algorithms:
- **Value Iteration**: A Dynamic Programming approach that iteratively improves state values.
- **Q-Learning**: A Model-Free approach that updates Q-values based on agent experiences.

### How It Works
1. **Select the RL algorithm**, grid size, and goal position.
2. **Train the model** to compute the optimal policy.
3. **Start the simulation**, and the agent will navigate using the learned policy.
4. **Stop the simulation anytime** using the stop button.

Try it out and see how the agent learns to reach the goal efficiently! 🚀
""")

# Store algorithm selection in session state
if "algorithm" not in st.session_state:
    st.session_state.algorithm = ALGORITHMS[0]

st.session_state.algorithm = st.selectbox("Select Algorithm", ALGORITHMS, index=ALGORITHMS.index(st.session_state.algorithm))

# Store grid size in session state
if "grid_size" not in st.session_state:
    st.session_state.grid_size = DEFAULT_GRID_SIZE

grid_size = st.number_input("Grid Size (Min: 5, Max: 20)", min_value=5, max_value=20, value=st.session_state.grid_size, step=1)
st.session_state.grid_size = grid_size  # Update stored value

# Ensure goal position remains valid
if "goal_x" not in st.session_state or st.session_state.goal_x >= grid_size:
    st.session_state.goal_x = min(DEFAULT_GOAL[0], grid_size - 1)
if "goal_y" not in st.session_state or st.session_state.goal_y >= grid_size:
    st.session_state.goal_y = min(DEFAULT_GOAL[1], grid_size - 1)

goal_x = st.number_input("Goal X Position (0 to Grid Size - 1)", min_value=0, max_value=grid_size - 1, value=st.session_state.goal_x)
goal_y = st.number_input("Goal Y Position (0 to Grid Size - 1)", min_value=0, max_value=grid_size - 1, value=st.session_state.goal_y)
st.session_state.goal_x, st.session_state.goal_y = goal_x, goal_y

goal_state = (goal_x, goal_y)

# Ensure session state for policy and agent
if "policy" not in st.session_state:
    st.session_state.policy = np.full((grid_size, grid_size), ' ')
if "agent" not in st.session_state:
    st.session_state.agent = None

# Train Model Button
if st.button("Train Model"):
    if st.session_state.goal_x >= grid_size or st.session_state.goal_y >= grid_size:
        st.error("Goal state is outside the grid! Please adjust.")
    else:
        if st.session_state.algorithm == "Value Iteration":
            agent = ValueIteration(grid_size, goal_state)
            agent.run_value_iteration()
            st.session_state.policy = agent.policy
        else:
            agent = QLearning(grid_size, goal_state)
            agent.train_q_learning()
            policy = np.full((grid_size, grid_size), ' ')
            for i in range(grid_size):
                for j in range(grid_size):
                    best_action = ["U", "D", "L", "R"][np.argmax(agent.q_table[i, j])]
                    policy[i, j] = best_action
            st.session_state.policy = policy

        st.session_state.agent = agent  # Store trained model
        st.success(f"Training Complete! Algorithm: **{st.session_state.algorithm}**. Click 'Start Simulation' to begin.")

# Simulation Controls
start_simulation = st.button("Start Simulation")
stop_simulation = st.button("Stop Simulation")

# Display Policy Grid
st.write("### Optimal Policy (Agent's Movement Direction)")
grid_placeholder = st.empty()
CELL_SIZE = 40

def render_grid(state):
    """ Render grid in Streamlit with agent movement """
    grid_html = "<table style='border-collapse: collapse;'>"
    
    for i in range(grid_size):
        grid_html += "<tr>"
        for j in range(grid_size):
            cell_color = "white"
            cell_text = st.session_state.policy[i, j]  # Use stored policy

            if (i, j) == state:
                cell_color = "red"
                cell_text = "A"
            elif (i, j) == goal_state:
                cell_color = "green"
            
            grid_html += f"<td style='width: {CELL_SIZE}px; height: {CELL_SIZE}px; border: 1px solid black; background-color: {cell_color}; text-align: center; font-weight: bold;'>{cell_text}</td>"
        
        grid_html += "</tr>"
    
    grid_html += "</table>"
    grid_placeholder.markdown(grid_html, unsafe_allow_html=True)

# Run Simulation only if Start is pressed
if start_simulation:
    if st.session_state.agent is None:
        st.error("Please train the model first!")
    else:
        state = (0, 0)
        st.write(f"### Simulation Running using **{st.session_state.algorithm}**...")

        for _ in range(grid_size * grid_size):
            if stop_simulation:  # Stop immediately if user clicks "Stop Simulation"
                st.warning("Simulation Stopped!")
                break
            
            render_grid(state)
            time.sleep(0.3)

            best_action = st.session_state.policy[state[0], state[1]]
            if best_action in ["U", "D", "L", "R"]:
                state = st.session_state.agent.get_next_state(state, best_action)

            if state == goal_state:
                render_grid(state)
                st.success(f"Agent Reached the Goal using **{st.session_state.algorithm}**! 🎯")
                break
