from os import system
from typing import List

from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF",
    model_file="orca-mini-3b.q4_0.gguf"
)


def get_prompt(instruction: str, history: List[str] = None) -> str:
    system = "You are an AI assistant that gives short answers."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if history is not None:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer thr question: "
    prompt += f"{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt

history = []

answer = ""
question = "Which city is the capital of India?"
for word in llm(get_prompt(question), stream=True):
    print(word,flush=True,end="")
    answer += word
print()

history.append(answer)
print(''.join(history))

question = "And which one is it of the USA?"
for word in llm(get_prompt(question, history), stream=True):
    print(word,flush=True,end="")
    answer += word
print()