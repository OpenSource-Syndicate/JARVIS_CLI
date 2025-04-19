from groq import Groq
from rich.console import Console

console = Console()

def get_groq_response(user_message, api_key, system_message="you are a helpful assistant.", 
                      temperature=0.5, max_tokens=1024, top_p=1, stream=False, model="llama-3.3-70b-versatile"):
    """
    Get a response from the Groq LLM.
    
    Args:
        user_message (str): The message to send to the LLM
        system_message (str): The system message that sets the behavior of the assistant
        temperature (float): Controls randomness (0-1)
        max_tokens (int): Maximum number of tokens to generate
        top_p (float): Controls diversity via nucleus sampling
        stream (bool): Whether to stream the response
    
    Returns:
        str: The LLM's response
    """
    client = Groq(api_key=api_key)
    
    console.print(f"[bold magenta]Sending prompt to Groq model: {model}[/bold magenta]")
    console.print(f"[dim]System message: {system_message}[/dim]")
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message,
            }
        ],
        model=model,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=top_p,
        stop=None,
        stream=stream,
    )
    
    response = chat_completion.choices[0].message.content
    console.print(f"[bold green]Response:[/bold green] {response}")
    return response

