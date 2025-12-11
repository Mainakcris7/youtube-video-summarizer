from urllib.parse import urlparse, parse_qs


def extract_video_id(url: str):
    """
    Extract video id from youtube video URL

    :param url: String representation of the youtube video URL
    """
    parsed = urlparse(url)

    # 1. Standard watch URL -> https://www.youtube.com/watch?v=ID
    if parsed.query:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]

    # 2. Shorts URL -> https://www.youtube.com/shorts/ID
    if "/shorts/" in parsed.path:
        return parsed.path.split("/shorts/")[1].split('/')[0]

    # 3. youtu.be URL -> https://youtu.be/ID
    if parsed.netloc == "youtu.be":
        return parsed.path.lstrip('/')

    # 4. Embed URL -> https://www.youtube.com/embed/ID
    if "/embed/" in parsed.path:
        return parsed.path.split("/embed/")[1].split('/')[0]

    return None


def get_grouped_transcriptions(transcriptions: list[dict], time: int | float) -> list[dict]:
    """
    Returns grouped transcription of the transcribed text with the given time window

    :param transcriptions: Dictionary version of the transcription result (obtained through res.to_raw_data())
    :type transcriptions: list[dict]
    :param time: The given time (in second) frame to which the transcription will be grouped
    :type time: int | float
    """
    curr_res = {
        'start': transcriptions[0]['start'],
        'end': transcriptions[0]['start'],
        'text': transcriptions[0]['text']
    }

    output = []
    for tr in transcriptions[1:]:
        if tr['start'] - curr_res['start'] <= time:
            curr_res['text'] += (" " + tr['text'])
            curr_res['end'] = tr['start']
        else:
            output.append(curr_res)
            curr_res = {
                'start': tr['start'],
                'end': tr['start'],
                'text': tr['text']
            }

    output.append(curr_res)
    return output

def get_grouped_data(data: list[dict], time: int | float) -> list[dict]:
    """
    Use it only after initital grouping using `get_grouped_transcriptions` has been done for the raw transcription data.
    
    Returns grouped transcription of the processed transcribed text with the given time window

    :param data: Dictionary version of the transcription result (obtained through res.to_raw_data())
    :type data: list[dict]
    :param time: The given time (in second) frame to which the transcription will be grouped
    :type time: int | float
    """
    curr_res = {
        'start': data[0]['start'],
        'end': data[0]['end'],
        'text': data[0]['text']
    }

    output = []
    for d in data[1:]:
        if d['start'] - curr_res['start'] <= time:
            curr_res['text'] += (" " + d['text'])
            curr_res['end'] = d['end']
        else:
            output.append(curr_res)
            curr_res = {
                'start': d['start'],
                'end': d['end'],
                'text': d['text']
            }

    output.append(curr_res)
    return output


def parse_segments(translated: str):
    result = {}
    parts = translated.split("<SEG_")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        seg_id, content = part.split(">", 1)
        seg_id = int(seg_id)

        result[seg_id] = content.strip()

    return result
