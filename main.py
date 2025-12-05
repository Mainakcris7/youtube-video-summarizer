from youtube_transcript_api import YouTubeTranscriptApi
import requests

session = requests.Session()
session.verify = False

video_id = "o1ZOEVx_Tr8"
ytt_api = YouTubeTranscriptApi(http_client=session)

print("Started...")
res = ytt_api.fetch(video_id, languages=['en', 'hi'])
print(res.snippets)

"""
Details of the Units
start: This value represents the exact moment (in seconds from the beginning of the video) when the transcribed text snippet starts.

duration: This value represents the length of time (in seconds) that the transcribed text snippet lasted.
"""