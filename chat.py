import chainlit as cl
from typing import List
from ctransformers import AutoModelForCausalLM


def get_prompt(instruction: str, history: List[str] = None) -> str:
    system = "You are an AI Assistant."
    prompt = f"### System:\n{system}\n\n### User:"
    if len(history) > 0:
        prompt += f"This is the conversation history : {''.join(history)}. Now Answer the question"
    prompt += f"\n{instruction}\n\n### Response:\n"
    return prompt


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content="")
    await msg.send()
    prompt = get_prompt(message.content, message_history)
    response = ""
    for word in llm(prompt, stream=True):
        await msg.stream_token(word)
        response += word
    await msg.update()
    message_history.append(response)


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        model_path_or_repo_id="zoltanctoth/orca_mini_3B-GGUF",
        model_file="orca-mini-3b.q4_0.gguf",
    )


# history = []

# question = "Which city is the capital of Indian?"

# answer = ""
# for word in llm(get_prompt(question), stream=True):
#     print(word, end="", flush=True)
#     answer += word
# print()

# history.append(answer)

# question = "And which city is of United States?"

# for word in llm(get_prompt(question, history), stream=True):
#     print(word, end="", flush=True)
# print()
