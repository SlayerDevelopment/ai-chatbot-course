from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    model_path_or_repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
    model_file="llama-2-7b-chat.Q4_0.gguf",
)


def get_prompt(instruction: str) -> str:
    system = "You are an AI Assistant."
    prompt = f"""<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"""

    # prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt


question = "Which city is the capital of Indian?"

for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
