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


def get_processed_response_time_slices(result: list[dict], time: int | float) -> list[dict]:
    """
    Returns grouped result of the transcribed text with the given time window
    - For shorts, it is recommended to pass 10 (second)

    :param result: Dictionary version of the transcription result (obtained through res.to_raw_data())
    :type result: list[dict]
    :param time: The given time (in second) frame to which the transcription will be grouped
    :type time: int | float
    """
    curr_res = {
        'start': result[0]['start'],
        'end': result[0]['start'],
        'text': result[0]['text']
    }

    output = []
    for d in result[1:]:
        if d['start'] - curr_res['start'] <= time:
            curr_res['text'] += (" " + d['text'])
            curr_res['end'] = d['start']
        else:
            output.append(curr_res)
            curr_res = {
                'start': d['start'],
                'end': d['start'],
                'text': d['text']
            }

    output.append(curr_res)
    return output
