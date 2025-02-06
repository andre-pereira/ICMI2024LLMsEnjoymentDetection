import time
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel, Image, Part


class ICMIGeminiVertexAPI:
    def __init__(self):
        PROJECT_ID = "anonymized"
        REGION = "us-central1"
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model_name = "gemini-1.5-pro-preview-0409"  
        self.generative_multimodal_model = GenerativeModel(self.model_name)
    
    def get_model_name(self):
        return self.model_name

    def rate_conversation(self, prompt, partNumber, turn, includeSideVideos, includeFrontalVideos):
        max_retries = 15  # Adjust the number of retries as needed
        retry_delay = 5  # Initial retry delay in seconds

        video_uri_Frontal = f"gs://anonymized/anonymized/P{partNumber}/P{partNumber}_T{turn}.mp4"
        video_uri_Side = f"gs://anonymized/anonymized/P{partNumber}/P{partNumber}_T{turn}.mp4"
        video_Frontal = Part.from_uri(video_uri_Frontal, mime_type="video/mp4")
        video_Side = Part.from_uri(video_uri_Side, mime_type="video/mp4")

        contents = [prompt]

        if includeSideVideos:
            contents.append(video_Side)
        if includeFrontalVideos:
            contents.append(video_Frontal)

        for attempt in range(max_retries):
            try:
                rating = ""
                responses = self.generative_multimodal_model.generate_content(contents, stream=True)
                for response in responses:
                    rating += response.text
                return rating.strip()  # Return the rating if successful

            except Exception as e:  # Catch the specific error
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed. Error: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise e  # Re-raise the error if all retries fail