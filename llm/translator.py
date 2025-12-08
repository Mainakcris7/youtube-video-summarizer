from dotenv import load_dotenv
import os
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import parse_segments

load_dotenv()

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite",
#     temperature=0.2
# )
llm = AzureChatOpenAI(
    api_key=os.environ["AZURE_OPENAI_GPT4O_API_KEY"],
    azure_deployment=os.environ["AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_GPT4O_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_GPT4O_API_VERSION"]
)

def translate_to_english(chunks: list[dict], from_lang: str, max_duration: int | float) -> list[dict]:
    """
    Translate each chunk to English while preserving time alignment and SEG markers.
    """

    translator_prompt_template = ChatPromptTemplate.from_messages([
        ('system',
         """
         You are a translation model. Translate the given text from '{from_lang}' into English.

         IMPORTANT RULES:
         - Keep segment markers EXACTLY as <SEG_X> (e.g., <SEG_1>)
         - Do NOT modify, remove, combine or reorder ANY markers
         - Translate ONLY the text after each marker
         - Some sentences may be incomplete within a block. You may use adjacent blocks for context, but must translate only the current block's content.
         - Output must follow:
           <SEG_X> translated sentence...
         """),
        ('user',
         """
         TEXT:
         {text_block}
         """)
    ])

    translator_chain = translator_prompt_template | llm | StrOutputParser()
    translated_output = []
    translated_segments = {}

    # Assign SEG IDs
    for i, chunk in enumerate(chunks):
        chunk["seg_id"] = i + 1  # <SEG_1>, <SEG_2>, ...

    # Merge into ~3-minute blocks
    blocks = []
    curr_block = [{
        'start': chunks[0]['start'],
        'end': chunks[0]['end'],
        'text': chunks[0]['text'],
        'seg_id': chunks[0]['seg_id'],
    }]  # list of dicts
    for chunk in chunks[1:]:
        if chunk['start'] - curr_block[0]['start'] <= max_duration:
            curr_block.append(chunk)
            # update end of first chunk to last chunk
            curr_block[0]['end'] = chunk['end']
        else:
            blocks.append(curr_block)
            curr_block = [{
                'start': chunk['start'],
                'end': chunk['end'],
                'text': chunk['text'],
                'seg_id': chunk['seg_id'],
            }]
    blocks.append(curr_block)

    print('Translation started...')
    for block in blocks:
        block_text = ""
        for w in block:
            block_text += f"<SEG_{w['seg_id']}> {w['text']}\n"

        output = translator_chain.invoke({
            'from_lang': from_lang,
            'text_block': block_text
        })

        parsed = parse_segments(output)
        translated_segments.update(parsed)

    # Reconstruct output
    for chunk in chunks:
        translated_output.append({
            "start": chunk["start"],
            "end": chunk["end"],
            "text": translated_segments[chunk["seg_id"]]
        })

    print('Translation success')
    return translated_output
