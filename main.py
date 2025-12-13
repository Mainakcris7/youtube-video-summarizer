import os
import json
from requests import Session
from warnings import filterwarnings
from youtube_transcript_api import YouTubeTranscriptApi
from utils import extract_video_id, get_grouped_transcriptions, get_grouped_data
from llm.translator import translate_to_english
from llm.agents import create_agent
from llm.vector_store import create_vector_store
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

# Directory paths
TRANSLATION_DIR = 'translations'
TRANSCRIPTION_DIR = 'transcriptions'
VECTOR_DB_PATH = os.path.join('db', 'faiss_db')

# Per chunk time durations (in seconds, change as needed)
TRANSCRIBED_TEXT_TIME_DURATION = 60  # ~1 minute per chunk
TRANSLATION_TIME_DURATION = 120  # ~2 minutes per chunk
VECTOR_STORE_TIME_DURATION = 120  # ~2 minutes per chunk
SUMMARIZATION_TIME_DURATION = 120  # ~2 minutes per chunk


def main():

    if TRANSLATION_TIME_DURATION < TRANSCRIBED_TEXT_TIME_DURATION:
        raise ValueError(
            "Translation chunk time duration can't be lesser than transcribed text time duration!")

    if VECTOR_STORE_TIME_DURATION < TRANSCRIBED_TEXT_TIME_DURATION:
        raise ValueError(
            "Vector store chunk time duration can't be lesser than transcribed text time duration!")

    if SUMMARIZATION_TIME_DURATION < TRANSCRIBED_TEXT_TIME_DURATION:
        raise ValueError(
            "Summarization chunk time duration can't be lesser than transcribed text time duration!")

    # video_id = extract_video_id("https://www.youtube.com/watch?v=pi9-m8RNqJo")
    # video_id = extract_video_id("https://www.youtube.com/watch?v=JjRiW_HpMoM")

    video_url = input("Enter the YouTube video URL: ")
    video_id = extract_video_id(video_url)

    file_name = f"{video_id}.json"
    print(f"Video id: {video_id}")

    # TO AVOID SSL Errors (OPTIONAL)
    # session = Session()
    # session.verify = False
    # ytt_api = YouTubeTranscriptApi(http_client=session)
    ytt_api = YouTubeTranscriptApi()

    # Get transcriptions
    transcription_file_path = os.path.join(TRANSCRIPTION_DIR, file_name)
    if (os.path.exists(transcription_file_path)):
        print('Transcription already exists.')
        with open(transcription_file_path, 'r') as f:
            transcripted_data = json.load(f)
    else:
        print("Transcription started...")
        try:
            res = ytt_api.fetch(video_id, languages=languages)
        except Exception as e:
            print(f"Error during transcription: {e}\
                  \nIt can be due to the invalid video id or network issues.")
            return

        result_dict = res.to_raw_data()
        transcripted_data = {
            'language': res.language,
            'language_code': res.language_code,
            'data': result_dict
        }
        print("Transcription done.")
        with open(file=transcription_file_path, mode='w') as f:
            json.dump(transcripted_data, f, indent=5)

    transcription_lang = transcripted_data['language']
    print(f"Transcription Language: {transcription_lang}")

    # Merge the transcriptions into specified time chunks
    processed_output = get_grouped_transcriptions(
        transcriptions=transcripted_data['data'], time=TRANSCRIBED_TEXT_TIME_DURATION)

    translated_output = []

    # Translate the transcriptions, if not in english
    if 'english' not in transcription_lang.lower():
        translation_file_path = os.path.join(TRANSLATION_DIR, file_name)
        if (os.path.exists(translation_file_path)):
            print('Translation already exists.')
            with open(translation_file_path, 'r') as f:
                translated_output = json.load(f)
        else:
            translated_output = translate_to_english(
                chunks=processed_output, from_lang=transcripted_data['language_code'], max_duration=TRANSLATION_TIME_DURATION)
            with open(file=translation_file_path, mode='w') as f:
                json.dump(translated_output, f, indent=5)

    if translated_output:
        processed_output = translated_output

    # Create vector store for RAG
    create_vector_store(processed_output, max_duration=VECTOR_STORE_TIME_DURATION,
                        vector_db_path=os.path.join(VECTOR_DB_PATH, video_id))

    # Create agent
    agent = create_agent(data=processed_output, summarization_group_time=SUMMARIZATION_TIME_DURATION,
                         vector_db_path=os.path.join(VECTOR_DB_PATH, video_id))

    chat_history = []

    system_prompt = """
    You are a helpful AI assistant who excels in youtube video summarizing, getting particular information from the video, answering user queries efficiently. Based on the tools and knowledge uyou have please address the user queries correctly.
    
    DO NOT ASK FOR LINKS, only answer based on the information you have.
    
    Correctly call the tools with the specific input types by seeing the method signature and docstring. Summarize the result correctly to the user.
    
    1. When calling the 'Get_Time_Related_Information', more focus on the current time frame, than the previous and next contexts (if present). 
    
    2. When user asks about summarizing the video, call the 'Youtube_Video_Summarizer' tool.
    
    3. When the user asks about summarizing the video per given time frame, call the 'Youtube_Video_Summarizer_Per_Given_Time_Chunk' tool. When giving the answer, convert seconds to minutes format. For example 80s means 1:20 minute.
    
    4. When using 'Question_Answering', attach the timestamps in minutes form, after the response of the user query. TRY TO APPROXIMATE, the timestamp range (NARROW DOWN based on where you are finding the answer, for example, if you get the answer in the middle of the timestamp range (e.g, 150s to 250s), narrow down it to (175s to 225s). Example: For an initial range of 150s to 250s where the answer is central, provide a narrowed range like 175s to 225s, which translates to (Starts at 2:55 - Ends at 3:45) in the final output 
    e.g, 
    <RESPONSE>
    source: [1:20 to 1:45, 2:30 to 2:50]    
    
    DO NOT HALUCINATE. If you don't know the answer, simply say Sorry I couldn't find the answer of your query, do not make things by your own!
    """

    chat_history.append(SystemMessage(content=system_prompt))

    # Start chatting with the agent
    while True:
        user_input = input("User: ")

        if user_input in ['bye', 'exit']:
            print("Thank you!\nExitting...")
            break

        chat_history.append(HumanMessage(content=user_input))

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

        chat_history.append(last_message)


if __name__ == '__main__':
    main()
