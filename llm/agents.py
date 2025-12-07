from dotenv import load_dotenv
import copy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)


def create_agent(data: list[dict]):
    processed_data = copy.deepcopy(data)

    @tool(name_or_callable='get_time_related_information', description='Useful when the user asks about what happened in the video, in a particular time', parse_docstring=True)
    def get_time_related_info(time_in_sec: int | float) -> str | None:
        """
        Useful when the user asks about what happened in the video, in a particular time.
        For better context, it will try to return the adjacent (previous and the next) transcribed text of the particular timestamp, along with the given timestamp text.

        For examle,
        User asks what happened during 3:20? -> 3 * 60 + 20 = 200 second.

        It will output something like -> 
        FROM 90s to 145s (PREVIOUS TIME CHUNK) (POSSIBLY NULL)
        # TRANSCRIBED TEXT OF WHAT HAPPENED DURING THIS PERIOD #

        FROM 150s to 210s (GIVEN TIME QUERY) (POSSIBLY NULL)
        # TRANSCRIBED TEXT OF WHAT HAPPENED DURING THIS PERIOD #

        FROM 215s to 275s (NEXT TIME CHUNK) (POSSIBLY NULL)
        # TRANSCRIBED TEXT OF WHAT HAPPENED DURING THIS PERIOD #

        IF NOTHING IS FOUND, it will return None

        :param time_in_sec: Timestamo of the video in second
        :type time_in_sec: int | float
        :return: Description
        :rtype: str | None
        """
        index = -1
        next_index = -1
        result = ""
        n = len(processed_data)

        for i, dict in enumerate(processed_data):
            if time_in_sec >= dict['start'] and time_in_sec <= dict['end']:
                index = i
                break
            # Only useful when exact match can't be computed (0-10) (12-21), but user asks for 11
            if time_in_sec <= dict['start'] and time_in_sec <= dict['end']:
                next_index = i
                break

        if index != -1:
            if index - 1 >= 0:
                result = f"""
                        FROM {processed_data[index - 1]['start']}s to {processed_data[index - 1]['end']}
                        {processed_data[index - 1]['text']}
                        """
            result += f"""
                        \n
                        FROM {processed_data[index]['start']}s to {processed_data[index]['end']}
                        {processed_data[index]['text']}
                        """
            if index + 1 < n:
                result += f"""
                        \n
                        FROM {processed_data[index + 1]['start']}s to {processed_data[index + 1]['end']}
                        {processed_data[index + 1]['text']}
                        """
        elif next_index != -1:
            if next_index - 1 >= 0:
                result = f"""
                        FROM {processed_data[next_index - 1]['start']}s to {processed_data[next_index - 1]['end']}
                        {processed_data[next_index - 1]['text']}
                        """
            result += f"""
                        \n
                        FROM {processed_data[next_index]['start']}s to {processed_data[next_index]['end']}
                        {processed_data[next_index]['text']}
                        """

        return result or None
