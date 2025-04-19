import google.generativeai as genai
from google.genai import types
import pathlib
import httpx
import pandas as pd
from TOOLS.DATABASE.data import ConversationDB
from TOOLS.AUDIO.tts import speak


def analyze_excel(excel_path, prompt="Summarize this document", api_key=None):
    """
    Analyze an Excel document by first converting it to CSV format
    
    Args:
        excel_path (str): Path to Excel file
        prompt (str): Analysis prompt for the model
        api_key (str): Google Generative AI API key
    
    Returns:
        str: Model response text
    """
    try:
        csv_path = convert_excel_to_csv(excel_path)
        result = analyze_csv(csv_path, prompt, api_key)
        speak(result)
        return result
    except Exception as e:
        return f"Error analyzing Excel: {str(e)}"


def analyze_csv(doc_path, prompt="Summarize this document", api_key=None):
    """
    Analyze a csv document using Gemini model
    
    Args:
        doc_path (str): URL or local path to CSV file
        prompt (str): Analysis prompt for the model
        api_key (str): Google Generative AI API key
        
    Returns:
        str: Model response text
    """
    genai.configure(api_key=api_key)
    db = ConversationDB()

    # Handle URL vs local path
    # Normalize and handle Windows paths
    doc_path = doc_path.strip('"')
    if doc_path.startswith('http'):
        filepath = pathlib.Path('temp.csv')
        filepath.write_bytes(httpx.get(doc_path).content)
    else:
        filepath = pathlib.Path(doc_path).resolve()

    try:
        # Prepare content parts
        csv_part = {
            'mime_type': 'text/csv',
            'data': filepath.read_bytes()
        }

        # Generate content using model
        model = genai.GenerativeModel('gemini-1.5-flash') # Using flash model
        response = model.generate_content([csv_part, prompt])
        
        # Store the processed data in database
        result_text = response.text if response.text else "No response generated"
        db.save_message(
            context=f"csv_processing:{filepath.name}",
            sender="assistant",
            message=result_text,
            timestamp=datetime.now().isoformat()
        )
        db.save_message(
            context=f"csv_processing:{filepath.name}",
            sender="system",
            message=f"Prompt: {prompt}\nResponse: {result_text}"
        )
        speak(result_text)

        return result_text
    except Exception as e:
        return f"Error analyzing CSV: {str(e)}"
    finally:
        # Clean up temp file if downloaded
        if doc_path.startswith('http') and filepath.exists():
            filepath.unlink()


def convert_excel_to_csv(excel_path, csv_path=None):
    """
    Convert Excel file to CSV format
    
    Args:
        excel_path (str): Path to Excel file
        csv_path (str, optional): Output CSV path. Defaults to None.
    
    Returns:
        str: Path to converted CSV file
    """
    try:
        df = pd.read_excel(excel_path)
        if not csv_path:
            csv_path = pathlib.Path(excel_path).with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        return str(csv_path)
    except Exception as e:
        return f"Error converting Excel to CSV: {str(e)}"