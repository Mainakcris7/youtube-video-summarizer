from youtube_transcript_api import YouTubeTranscriptApi
import requests
import os
from warnings import filterwarnings

filterwarnings('ignore')

session = requests.Session()
session.verify = False

# video_id = "o1ZOEVx_Tr8"
video_id = "keKEjMTFKSA"
ytt_api = YouTubeTranscriptApi(http_client=session)

print("Started...")
res = ytt_api.fetch(video_id, languages=['en', 'hi'])
result_dict = res.to_raw_data()

"""
Details of the Units
start: This value represents the exact moment (in seconds from the beginning of the video) when the transcribed text snippet starts.

duration: This value represents the length of time (in seconds) that the transcribed text snippet lasted.
"""
print(f"LANGUAGE: {res.language}")
# texts = [(result['start'], result['text']) for result in result_dict]
texts = [f"#TIME IN SECOND:{result['start']}# {result['text']}" for result in result_dict]

whole_text = " ".join(texts)
print(whole_text)

dir_path = "data"
file_name = "transcription.txt"

with open(file=os.path.join(dir_path, file_name), mode = 'w', encoding='utf-8') as f:
    f.write(whole_text)
    
print("Success")