import os
from openai import OpenAI

class ICMIOpenAIAPI:
    def __init__(self):
        # Setup the client with your API key
        APIKey = 'anonymized'
        # self.client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
        self.client = OpenAI(api_key = APIKey)
        #self.model_name = "gpt-3.5-turbo-0125"
        self.model_name = "gpt-4-turbo"
    
    def get_model_name(self):
        return self.model_name
    
    def rate_conversation(self, prompt, partNumber, turn, includeSideVideos, includeFrontalVideos):
        chat_completion = self.client.chat.completions.create(
            #model="gpt-4-0125-preview",  # Make sure this model is correct
            model = self.model_name,
            messages=[
                {"role": "user", "content": f"{prompt}"}
                ],
        )
        # Assuming you're looking to get the content of the first completion message
        return chat_completion.choices[0].message.content.strip()
