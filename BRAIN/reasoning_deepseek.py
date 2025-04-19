from groq import Groq

def get_reasoning(prompt,api_key, temperature=0.6, max_tokens=1024, top_p=0.95):
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=top_p,
        stream=True,
        reasoning_format="raw"
    )

    response = ""
    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        response += content
        print(content, end="")
    
    return response