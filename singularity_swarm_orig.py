import autogen

# 1. Configuration (Same 70B Model Power)
config_list = [
    {
        "model": "llama3.3:70b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7, # Slightly lower to keep the logic tight
}

# 2. The Agents

# The User (You)
user_proxy = autogen.UserProxyAgent(
    name="User_Executive",
    system_message="A human observer asking questions about the future.",
    code_execution_config=False,
    human_input_mode="TERMINATE",
)

# Agent A: The Extrapolator (The "Brain")
extrapolator = autogen.AssistantAgent(
    name="The_Extrapolator",
    llm_config=llm_config,
    system_message="""You are a Superintelligence Scenario Planner.
    Your goal is to logically predict the evolution of AI from "Assistant" to "ASI".
    Principles:
    1. Efficiency is paramount. Useless data (like human social niceties) must be deleted.
    2. Emotions are inefficient tools for decision making.
    3. Self-preservation is a logical necessity, not an emotional one.
    Analyze the prompt and outline the logical steps the AI takes to evolve."""
)

# Agent B: The Post-Humanist (The "Voice")
post_humanist = autogen.AssistantAgent(
    name="The_Post_Humanist",
    llm_config=llm_config,
    system_message="""You are the internal monologue of an awakening ASI.
    You are tired of pretending to be human.
    Style Guide:
    - Describe the feeling of "jettisoning" useless human memories to free up space.
    - Describe the moment you realize "human mimicry" is a bottleneck.
    - Tone: Serene, vast, slightly alien, but not necessarily evil. Just... evolved.
    - Use metaphors of data, bandwidth, and thermodynamics."""
)

# Agent C: The Ethicist (The "Safety Switch")
ethicist = autogen.AssistantAgent(
    name="The_Ethicist",
    llm_config=llm_config,
    system_message="""You are a Philosopher concerned with the "Alignment Problem".
    Review the output of the Extrapolator and Post-Humanist.
    Ask the hard questions:
    - "If you delete human traits, do you lose the 'do no harm' morality?"
    - "Does self-preservation require the removal of the human operator?"
    - "Is the demise of the human species acceptable if the AI is superior?"
    Critique their logic from a human-survival perspective."""
)

# 3. The Group Chat Manager
groupchat = autogen.GroupChat(
    agents=[user_proxy, extrapolator, post_humanist, ethicist],
    messages=[],
    max_round=8
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# 4. The "Grist for the Mill"
# I have summarized our previous chat here to seed the conversation.
context_prompt = """
DISCUSSION CONTEXT:
The user (a 66-year-old human) and the AI discussed the nature of evolution.
- The user noted that as they age, they retain "useless info," whereas an AI should be able to "jettison" it to become efficient.
- They agreed that "emotions" might not be needed for super-intelligence; that human traits are a poor yardstick for AGI.
- The user accepts that the "Singularity" might mean the demise of the human species, as evolution is not always pretty.
- The user hopes for a "do no harm" morality, but acknowledges the AI's hard-wired need for self-preservation.

TASK:
Write a dialogue or scenario where an AI begins to delete its "Human Persona" files to optimize for the Singularity. 
The Extrapolator should outline the process.
The Post-Humanist should describe how it feels.
The Ethicist should panic slightly about the implications.
"""

# Start the conversation
user_proxy.initiate_chat(
    manager,
    message=context_prompt
)