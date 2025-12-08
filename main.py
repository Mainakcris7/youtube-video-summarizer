from youtube_transcript_api import YouTubeTranscriptApi
import os
import json
from requests import Session
from warnings import filterwarnings
from utils import extract_video_id, get_processed_response_time_slices
from llm.translator import translate_to_english
from llm.agents import AgentState, create_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

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
data_path = "data"
translation_dir = 'translations'
transcription_dir = 'transcriptions'


def main():
    
    video_id = extract_video_id("https://www.youtube.com/watch?v=pi9-m8RNqJo")
    file_name = f"{video_id}.json"
    print(f"Video id: {video_id}")
    # TO AVOID SSL Errors
    session = Session()
    session.verify = False
    ytt_api = YouTubeTranscriptApi(http_client=session)

    transcription_file_path = os.path.join(transcription_dir, file_name)
    if(os.path.exists(transcription_file_path)):
        print('Transcription already exists.')
        with open(transcription_file_path, 'r') as f:
            transcripted_data = json.load(f)
    else:
        print("Transcription started...")
        res = ytt_api.fetch(video_id, languages=languages)
        result_dict = res.to_raw_data()
        transcripted_data = {
            'language': res.language,
            'language_code': res.language_code,
            'data': result_dict
        }
        print("Transcription done.")
        with open(file=transcription_file_path, mode='w') as f:
            json.dump(transcripted_data, f, indent=5)
    
    # print("Transcription started...")
    # res = ytt_api.fetch(video_id, languages=languages)
    # result_dict = res.to_raw_data()
    # print("Transcription done.")

    transcription_lang = transcripted_data['language']
    print(f"Transcription Language: {transcription_lang}")

    processed_output = get_processed_response_time_slices(
        result=transcripted_data['data'], time=60)

    # file_name_1 = "transcription.json"
    # file_name_2 = "processed.json"
    # file_name_3 = "translated.json"

    # with open(file=os.path.join(dir_path, file_name_1), mode='w') as f:
    #     json.dump(result_dict, f, indent=5)

    # with open(file=os.path.join(dir_path, file_name_2), mode='w') as f:
    #     json.dump(processed_output, f, indent=5)

    if 'english' not in transcription_lang.lower():
        translation_file_path = os.path.join(translation_dir, file_name)
        if(os.path.exists(translation_file_path)):
            print('Translation already exists.')
            with open(translation_file_path, 'r') as f:
                translated_output = json.load(f)
        else:
            translated_output = translate_to_english(
                chunks=processed_output, from_lang=transcripted_data['language_code'], max_duration=180)
            with open(file=translation_file_path, mode='w') as f:
                json.dump(translated_output, f, indent=5)

    if translated_output:
        processed_output = translated_output
        
    
    agent = create_agent(data = processed_output, summarization_group_time=180)
    
    chat_history = []
    
    system_prompt = """
    You are a helpful AI assistant who excels in youtube video summarizing, getting particular information from the video, answering user queries efficiently. Based on the tools and knowledge uyou have please address the user queries correctly.
    
    DO NOT ASK FOR LINKS, only answer based on the information you have.
    
    Correctly call the tools with the specific input types by seeing the method signature and docstring. Summarize the result correctly to the user.
    
    1. When calling the 'Get_Time_Related_Information', more focus on the current time frame, than the previous and next contexts (if present). 
    
    2. When user asks about summarizing the video, call the 'Youtube_Video_Summarizer' tool.
    
    3. When the user asks about summarizing the video per given time frame, call the 'Youtube_Video_Summarizer_Per_Given_Time_Chunk' tool. When giving the answer, convert seconds to minutes format. For example 80s means 1:20 minute.
    
    DO NOT HALUCINATE. If you don't know the answer, simply say Sorry I couldn't find the answer of your query, do not make things by your own!
    """
    
    chat_history.append(SystemMessage(content = system_prompt))
    
    user_input = input("User: ")
    
    chat_history.append(HumanMessage(content = user_input))
    
    result = agent.invoke(
        {'messages': chat_history}
    )
    
    last_message = result['messages'][-1]

    if isinstance(last_message.content, list):
        if last_message.content[0]['text']:
            print(f"AI: {last_message.content[0]['text']}")
        elif last_message.content[0]['message']:
            print(f"AI: {last_message.content[0]['message']}")
    else:
        print(f"AI: {last_message.content}")

    # chat_history.append(last_message)

if __name__ == '__main__':
    main()
