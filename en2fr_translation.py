from transformers import pipeline  
  
# Initialize the translation pipeline  
translator = pipeline("translation_en_to_fr")  
  
# Function to translate English to French  
def translate_to_french(text):  
    translation = translator(text)  
    return translation[0]['translation_text']  
  
# User input  
english_text = input("Enter text in English to translate to French: ")  
# english_text = "how are you"


# Translate and print the result  
french_translation = translate_to_french(english_text)  
print(f"french translation: {french_translation}")  
