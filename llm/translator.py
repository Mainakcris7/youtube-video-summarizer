from deep_translator import GoogleTranslator


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
    translator = GoogleTranslator(source=from_lang, target='en')
    translated_output = []

    print('Translation started...')
    for chunk in chunks:
        output = translator.translate(chunk['text'])

        translated_output.append({
            'start': chunk['start'],
            'end': chunk['end'],
            'text': output
        })
    print('Translation success')
    return translated_output
