from typing import List
import chainlit as cl

from ctransformers import AutoModelForCausalLM


def get_prompt(instruction: str, history: List[str], answer_kind: str = "short") -> str:
    system = "You are an AI assistant that gives " + answer_kind + " answers."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if history is not None and len(history) > 0:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer thr question: "
    prompt += f"{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF",
        model_file="orca-mini-3b.q4_0.gguf"
    )
    print("A new chat session has started!")


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    print(message_history)
    msg = cl.Message(content="")
    await msg.send()

    prompt = get_prompt(message.content, message_history, "expensive")
    response = ""
    for word in llm(prompt, stream=True):
        #print(word)
        response += word
        await msg.stream_token(word)
    await msg.update()
    message_history.append(response)

"""
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
"""
