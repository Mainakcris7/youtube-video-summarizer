from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)


def translate_to_english(chunks: list[dict], from_lang: str) -> list[dict]:
    """
    Translate each chunks to english language

    :param chunks: Each chunks containing start, end, text (To be translated)
    :type chunks: list[dict]
    :param from_lang: The current language for each chunks' text
    :type from_lang: str
    :return: The translated text for each chunk in the same formt as the input 'chunks'
    :rtype: list[dict]
    """
    translator_prompt_template = ChatPromptTemplate.from_messages([
        ('system',
         'You are a helpful AI assistant. Your task is to efficiently translate the given text from {from_lang} to "English" language. Do not halucinate, or do not generate information that is not present in the given text. The given text may be incomplete, that is why I will add, its possibly previous chunk of text and the later chunk of text for your CONTEXT purpose only. NOTE: the previous chunk or later chunk can be empty, based on the position of the current chunk, handle it efficiently. Only provide the translated output, with no extra text!'),
        ('system',
         """
        This will be the input format - 
        <PREV_CHUNK_OF_TEXT>

        <CURR_CHUNK_OF_TEXT> # YOU NEED TO TRANSLATE THIS PIECE OF TEXT ONLY IN ENGLISH#

        <NEXT_CHUNK_OF_TEXT>
        """),
        ('user',
         """
        PREV CHUNK: {prev_chunk}

        TRANSLATE THIS CHUNK: {curr_chunk}

        NEXT CHUNK: {next_chunk}
        """)
    ])
    translator_chain = translator_prompt_template | llm | StrOutputParser()
    translated_output = []

    print('Translation started...')
    for i, chunk in enumerate(chunks):
        output = translator_chain.invoke(
            {
                'from_lang': from_lang,
                'prev_chunk': chunks[i - 1]['text'] if i > 0 else None,
                'curr_chunk': chunk['text'],
                'next_chunk': chunks[i + 1]['text'] if i < (len(chunks) - 1) else None
            }
        )

        translated_output.append({
            'start': chunk['start'],
            'end': chunk['end'],
            'text': output
        })
    print('Translation success')
    return translated_output
