from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import parse_segments

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.1
)


def translate_to_english(chunks: list[dict], from_lang: str) -> list[dict]:
    """
    Translate each chunk to English while preserving time alignment and SEG markers.
    """

    translator_prompt_template = ChatPromptTemplate.from_messages([
        ('system',
         """
         You are a translation model. Translate the given text from {from_lang} into English.

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
    for i, w in enumerate(chunks):
        w["seg_id"] = i + 1  # <SEG_1>, <SEG_2>, ...

    # Merge into ~3-minute blocks
    blocks = []
    curr_block = [{
        'start': chunks[0]['start'],
        'end': chunks[0]['end'],
        'text': chunks[0]['text'],
        'seg_id': chunks[0]['seg_id'],
    }]  # list of dicts
    for d in chunks[1:]:
        if d['start'] - curr_block[0]['start'] <= 180:
            curr_block.append(d)
            # update end of first chunk to last chunk
            curr_block[0]['end'] = d['end']
        else:
            blocks.append(curr_block)
            curr_block = [{
                'start': d['start'],
                'end': d['end'],
                'text': d['text'],
                'seg_id': d['seg_id'],
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

        parsed = parse_segments(output)  # expects {seg_id: translated_text}
        translated_segments.update(parsed)

    # Reconstruct output
    for w in chunks:
        translated_output.append({
            "start": w["start"],
            "end": w["end"],
            "text": translated_segments[w["seg_id"]]
        })

    print('Translation success')
    return translated_output
