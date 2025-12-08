from dotenv import load_dotenv
import os
import copy
from utils import get_processed_response_time_slices
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import ToolNode

load_dotenv()

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0.2
# )

llm = AzureChatOpenAI(
    api_key=os.environ["AZURE_OPENAI_GPT4O_API_KEY"],
    azure_deployment=os.environ["AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_GPT4O_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_GPT4O_API_VERSION"]
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def create_agent(data: list[dict], summarization_group_time: int | float = 180):
    processed_data = copy.deepcopy(data)
    
    # Merge into ~3-minute blocks (For better summarization)
    summarization_data = get_processed_response_time_slices(processed_data, summarization_group_time)

    @tool(name_or_callable='Get_Time_Related_Information', description='Useful when the user asks about what happened in the video, in a particular time', parse_docstring=True)
    def get_time_related_info(time_in_sec: int | float) -> str | None:
        """
        Useful when the user asks about what happened in the video, in a particular time.
        For better context, it will try to return the adjacent (previous and the next) 
        transcribed text of the particular timestamp, along with the given timestamp text.

        For example,
        User asks what happened during 3:20? -> (3 * 60) + 20 = 200 second.

        It will output something like -> 
        FROM 90s to 145s (WHAT HAPPENED PREVIOUSLY) (POSSIBLY NULL)
        # TRANSCRIBED TEXT OF WHAT HAPPENED DURING THIS PERIOD #

        FROM 150s to 210s (GIVEN TIME QUERY) (POSSIBLY NULL)
        # TRANSCRIBED TEXT OF WHAT HAPPENED DURING THIS PERIOD #

        FROM 215s to 275s (NEXT TIME CHUNK) (POSSIBLY NULL)
        # TRANSCRIBED TEXT OF WHAT HAPPENED DURING THIS PERIOD #

        IF NOTHING IS FOUND, it will return None

        Args:
            time_in_sec (int | float): Timestamp of the video in second.

        Returns:
            str or None: Description of the events around the specified timestamp, 
                        or None if nothing is found.
        """
        if time_in_sec < 0:
            return "Invalid time stamp"
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
                        FROM {processed_data[index - 1]['start']}s to {processed_data[index - 1]['end']} (WHAT HAPPENED PREVIOUSLY)
                        {processed_data[index - 1]['text']}
                        """
            result += f"""
                        \n
                        FROM {processed_data[index]['start']}s to {processed_data[index]['end']} (GIVEN TIME QUERY)
                        {processed_data[index]['text']}
                        """
            if index + 1 < n:
                result += f"""
                        \n
                        FROM {processed_data[index + 1]['start']}s to {processed_data[index + 1]['end']} (WHAT HAPPENS NEXT)
                        {processed_data[index + 1]['text']}
                        """
        elif next_index != -1:
            if next_index - 1 >= 0:
                result = f"""
                        FROM {processed_data[next_index - 1]['start']}s to {processed_data[next_index - 1]['end']} (WHAT HAPPENED PREVIOUSLY)
                        {processed_data[next_index - 1]['text']}
                        """
            result += f"""
                        \n
                        FROM {processed_data[next_index]['start']}s to {processed_data[next_index]['end']} (WHAT HAPPENS NEXT)
                        {processed_data[next_index]['text']}
    
                        """
        return result or None

    @tool(name_or_callable='Youtube_Video_Summarizer', description='Useful when the user asks about summarizing the entire video. Call this tool whenever user asks about summarization.')
    def summarize_video() -> str:
        map_chain_template = [
            ('system', "You are a helpful AI assistant. Your job is to efficiently summarize the ducument chunk that is passed to you by the user. DON'T HALUCINATE, DON'T MAKE UP STORIES, summarize solely based on the text that is provided to you. DON'T LOSE MUCH INFORMATION, as these summaries will be re-summarized again, combined."),
            ('user', 'Please summarize the chunk: {document}')
        ]
        
        print("Summarizing...")
        
        map_chain_prompt_template = ChatPromptTemplate.from_messages(map_chain_template)
        map_chain = map_chain_prompt_template | llm | StrOutputParser()
        
        summaries = [map_chain.invoke(d['text']) for d in summarization_data]
        all_summaries = "\n".join(summaries)
        
        reduce_chain_template = [
            ('system', "You are a helpful AI assistant. Your job is to efficiently summarize the summaries (basically you need to provide summary of the summaries) that is passed to you by the user. DON'T HALUCINATE, DON'T MAKE UP STORIES, summarize solely based on the text that is provided to you!. DON'T LOSE MUCH INFORMATION, summarize efficienly."),
            ('user', 'Please summarize the summaries : {document}')
        ]

        reduce_chain_prompt_template = ChatPromptTemplate.from_messages(reduce_chain_template)
        reduce_chain = reduce_chain_prompt_template | llm | StrOutputParser()
        
        final_summary = reduce_chain.invoke(all_summaries)
        
        return final_summary
    
    @tool(name_or_callable='Youtube_Video_Summarizer_Per_Given_Time_Chunk', description='Useful for summarizing a YouTube video based on specific time segments or intervals specified by the user. For instance, it can summarize the content every two minutes. Use this when the user needs a detailed, segmented summary rather than a single, overall summary.', parse_docstring=True)
    def summarize_video_per_given_time(time_in_sec: int | float) -> list[str]:
        """Generates segment-by-segment summaries of a YouTube video.

        Divides the video into equal-length chunks defined by `time_in_sec`
        and summarizes the content of each segment chronologically.

        Args:
            time_in_sec: The length of each summary chunk in seconds (must be > 25).

        Returns:
            A :obj:`list` of :obj:`str`, where each string is the summary for 
            a chronological time segment.

        Example:
            >>> summarize_video_per_given_time(time_in_sec = 120)
            ['Summary of 0:00-2:00', 'Summary of 2:00-4:00', ...]
        """
        if time_in_sec <= 25:
            return "Error: Time frame must be greater than 10 seconds."
        
        summarization_data_time_wise = get_processed_response_time_slices(processed_data, time_in_sec)
        map_chain_template = [
            ('system', "You are a helpful AI assistant. Your job is to efficiently summarize the ducument chunk that is passed to you by the user. DON'T HALUCINATE, DON'T MAKE UP STORIES, summarize solely based on the text that is provided to you."),
            ('user', 'Please summarize the chunk: {document}')
        ]
        
        print("Summarizing...")
        
        map_chain_prompt_template = ChatPromptTemplate.from_messages(map_chain_template)
        map_chain = map_chain_prompt_template | llm | StrOutputParser()
        
        summaries = [f"SUMMARY OF {d['start']}s to {d['end']}s\n{map_chain.invoke(d['text'])}" for d in summarization_data_time_wise]
        
        return summaries
    
    
    # Creating the graph
    global llm
    llm = llm.bind_tools([get_time_related_info, summarize_video, summarize_video_per_given_time])
    
    def llm_node(state: AgentState) -> AgentState:
        result = llm.invoke(state['messages'])
        return {'messages': [result]}
    
    def should_call_tools(state: AgentState) -> str:
        if state['messages'][-1].tool_calls:
            return 'tool_call'
        else:
            return 'end'
    
    graph = StateGraph(AgentState)
    
    graph.add_node('llm_node', llm_node)
    graph.add_node('tools_node', ToolNode(tools = [get_time_related_info, summarize_video, summarize_video_per_given_time]))
    
    graph.add_edge(START, 'llm_node')
    graph.add_conditional_edges(
        'llm_node',
        should_call_tools,
        {
            'tool_call': 'tools_node',
            'end': END
        }
    )
    graph.add_edge('tools_node', 'llm_node')
    
    app = graph.compile()
    
    return app