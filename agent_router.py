# agent_router.py

from langchain.agents import initialize_agent
from tools.fitness_tool import get_fitness_tool
from tools.essay_tool import get_essay_tool
from tools.social_tool import get_social_tool


def get_agent(task_type):
    tools_map = {
        "Fitness Plan Generator": get_fitness_tool(),
        "Essay Generator": get_essay_tool(),
        "Social Media Content Generator": get_social_tool()
    }

    selected_tool = tools_map.get(task_type)

    return initialize_agent(
        tools=[selected_tool],
        agent="zero-shot-react-description",
        verbose=True
    )