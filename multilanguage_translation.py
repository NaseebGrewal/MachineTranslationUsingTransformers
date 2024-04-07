from transformers import MarianMTModel, MarianTokenizer  
  
def translate_text(text, target_language):  
    # Define the model repository path  
    model_name = f'Helsinki-NLP/opus-mt-en-{target_language}'  
      
    # Load the tokenizer and model  
    tokenizer = MarianTokenizer.from_pretrained(model_name)  
    model = MarianMTModel.from_pretrained(model_name)  
  
    # Tokenize the text  
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))  
  
    # Decode the tokens to string  
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)  
  
    return translation  
  
# Example usage:  
if __name__ == "__main__":  
    # User input  
    english_text = input("Enter English text to translate: ")  
    target_language = input("Enter target language code (e.g., 'fr' for French): ")  
  
    # Translate the text  
    translated_text = translate_text(english_text, target_language)  
  
    # Output the translation  
    print(f"Translated text ({target_language}): {translated_text}")  