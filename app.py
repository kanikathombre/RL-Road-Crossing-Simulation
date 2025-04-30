import streamlit as st
from environment import RoadCrossingEnv
from agent import QLearningAgent  

# Initialize session state
if 'env' not in st.session_state:
    st.session_state.env = RoadCrossingEnv() # Create environment instance
    st.session_state.agent = QLearningAgent(action_space=4) # Initialize agent with 4 actions
    st.session_state.state = st.session_state.env.reset() # Get initial state
    st.session_state.done = False # Game not finished
    st.session_state.steps = 0 # Step counter
    st.session_state.score = 0 # Cumulative score
    st.session_state.last_reward = 0 # Reward from last action

st.title("🧠 RL Road Crossing Simulation")
st.subheader("Initial Environment State:" if st.session_state.steps == 0 else "Next Step:")

# Render the grid
st.text(st.session_state.env.render_string())


# Display traffic lights
st.markdown(f"**Pedestrian Position:** 🚶 {st.session_state.env.pedestrian_pos}")
st.markdown(f"**Vehicle Positions:** 🚗 {st.session_state.env.vehicles}")
st.markdown(f"**Traffic Lights (0=Red, 1=Green):** {st.session_state.env.traffic_lights}")

# Action and Step
if not st.session_state.done:
    if st.button("🚶 Next Step"):
        action = st.session_state.agent.choose_action(st.session_state.state)
        next_state, rewards, done, _ = st.session_state.env.step(action)

        reward = rewards[0]  # Explicitly unpack reward

        # Update session state
        st.session_state.state = next_state
        st.session_state.done = done
        st.session_state.steps += 1
        st.session_state.score += reward
        st.session_state.last_reward = reward

        st.rerun()


# Show reward and score
st.markdown(f"**Reward (Last Step):** :green[{st.session_state.last_reward}]")
st.markdown(f"**Steps Taken:** {st.session_state.steps}")
st.markdown(f"**Cumulative Score:** :blue[{st.session_state.score}]")

# Restart button
if st.session_state.done:
    if st.button("🔁 Restart Simulation"):
        st.session_state.clear()
        st.rerun()
