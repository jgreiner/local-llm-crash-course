from typing import List
import chainlit as cl
from ctransformers import AutoModelForCausalLM


class ChatSession:
    def __init__(self, model_name: str, model_file: str, answer_kind: str = "short"):
        self.answer_kind = answer_kind
        self.message_history: List[str] = []
        self.llm = AutoModelForCausalLM.from_pretrained(model_name, model_file=model_file)
        print("A new chat session has been initialized!")

    def get_prompt(self, instruction: str) -> str:
        system = f"You are an AI assistant that gives {self.answer_kind} answers."
        prompt = f"### System:\n{system}\n\n### User:\n"
        if self.message_history:
            prompt += f"This is the conversation history: {''.join(self.message_history)}. Now answer the question: "
        prompt += f"{instruction}\n\n### Response:\n"
        print(prompt)
        return prompt

    async def handle_message(self, message: cl.Message):
        print("history:")
        print(self.message_history)
        msg = cl.Message(content="")
        await msg.send()

        prompt = self.get_prompt(message.content)
        response = ""
        for word in self.llm(prompt, stream=True):
            response += word
            await msg.stream_token(word)
        await msg.update()
        print(response)
        self.message_history.append(response)


chat_session = ChatSession(
    model_name="zoltanctoth/orca_mini_3B-GGUF",
    model_file="orca-mini-3b.q4_0.gguf",
    answer_kind="expensive"
)


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("chat_session", chat_session)
    print("A new chat session has started!")


@cl.on_message
async def on_message(message: cl.Message):
    my_chat_session = cl.user_session.get("chat_session")
    await my_chat_session.handle_message(message)
