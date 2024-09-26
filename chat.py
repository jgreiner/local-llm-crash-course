from os import system

from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF",
    model_file="orca-mini-3b.q4_0.gguf"
)


def get_prompt(instruction: str) -> str:
    system = "You are an AI assistant that gives funny and short answers."
    prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    return prompt



question = "Which city iis the capital of India"


for word in llm(get_prompt(question), stream=True):
    print(word,flush=True,end="")
print()