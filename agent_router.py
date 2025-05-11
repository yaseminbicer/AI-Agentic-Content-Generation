from tools.essay_tool import run_essay_tool
from tools.fitness_tool import run_fitness_tool
from tools.social_tool import run_social_tool

def run_agent(task_type: str, prompt: str) -> str:
    if task_type == "Essay Generator":
        return run_essay_tool(prompt)
    elif task_type == "Fitness Plan Generator":
        return run_fitness_tool(prompt)
    elif task_type == "Social Media Content Generator":
        return run_social_tool(prompt)
    else:
        raise ValueError("Unknown task type")