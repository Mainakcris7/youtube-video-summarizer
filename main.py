from youtube_transcript_api import YouTubeTranscriptApi
import os
import json
from warnings import filterwarnings
from utils import extract_video_id, get_processed_response_time_slices
from llm.agents import translate_to_english
# from llm.translator import translate_to_english

filterwarnings('ignore')

languages = [
    'en',       # English
    'hi',       # Hindi
    'es',       # Spanish
    'fr',       # French
    'de',       # German
    'pt',       # Portuguese
    'ru',       # Russian
    'ar',       # Arabic
    'ja',       # Japanese
    'ko',       # Korean
    'zh-Hans',  # Chinese Simplified
    'zh-Hant',  # Chinese Traditional
    'id',       # Indonesian
    'it',       # Italian
    'bn',       # Bengali
]


def main():
    video_id = extract_video_id("https://www.youtube.com/watch?v=pi9-m8RNqJo")
    print(video_id)
    # video_id = "keKEjMTFKSA"
    ytt_api = YouTubeTranscriptApi()

    print("Transcription started...")
    res = ytt_api.fetch(video_id, languages=languages)
    result_dict = res.to_raw_data()
    print("Transcription done.")

    transcription_lang = res.language
    print(f"Transcription Language: {transcription_lang}")

    processed_output = get_processed_response_time_slices(
        result=result_dict, time=60)

    dir_path = "data"
    file_name_1 = "transcription.json"
    file_name_2 = "processed.json"
    file_name_3 = "translated.json"

    # with open(file=os.path.join(dir_path, file_name_1), mode='w') as f:
    #     json.dump(result_dict, f, indent=5)

    # with open(file=os.path.join(dir_path, file_name_2), mode='w') as f:
    #     json.dump(processed_output, f, indent=5)

    if 'english' not in transcription_lang.lower():
        translated_output = translate_to_english(
            chunks=processed_output, from_lang=res.language_code)
        with open(file=os.path.join(dir_path, 'translated_new.json'), mode='w') as f:
            json.dump(translated_output, f, indent=5)

    print("Success")


if __name__ == '__main__':
    main()
