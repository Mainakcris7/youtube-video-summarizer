# YouTubeInsight - (YouTube Video Transcription, Translation, and Summarization with LLMs and RAG) 🤖

A powerful AI-driven application that transcribes, translates, and intelligently summarizes YouTube videos. This tool leverages large language models (LLMs) and retrieval-augmented generation (RAG) to provide comprehensive summaries, answer specific questions about video content, and retrieve time-stamped information from videos.

## Features

- **🎥 Automatic Transcription**: Extract transcripts from YouTube videos in multiple languages
- **🌍 Multi-Language Support**: Automatically translate non-English transcriptions to English for processing
- **📝 Intelligent Summarization**: 
  - Generate comprehensive summaries of entire videos
  - Create time-segmented summaries (e.g., summaries every 2 minutes)
- **❓ Smart Q&A**: Ask questions about video content and get accurate answers with timestamps
- **⏱️ Time-Based Search**: Query what happened at specific timestamps in the video
- **💾 Vector Database**: Uses FAISS for efficient semantic search and context retrieval
- **🤖 Conversational AI Agent**: Interactive chat interface to explore video content

## Prerequisites

- Python 3.14 or higher
- [uv](https://docs.astral.sh/uv/) - Python package installer and dependency manager
- Azure OpenAI API credentials
- Internet connection (for YouTube and API access)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd youtube-video-summarizer
```

### 2. Install Dependencies

Using `uv`:

```bash
uv sync
```

This installs all required dependencies specified in `pyproject.toml`.

### 3. Configure Environment Variables

Copy the `.env.example` file and rename it to `.env`:

```bash
cp .env.example .env
```

Then, edit the `.env` file with your Azure OpenAI credentials:

```env
AZURE_OPENAI_GPT4O_API_KEY=<your-azure-openai-api-key>
AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME=<your-deployment-name>
AZURE_OPENAI_GPT4O_ENDPOINT=<your-azure-openai-endpoint>
AZURE_OPENAI_GPT4O_API_VERSION=<api-version-e.g-2024-08-01-preview>

AZURE_OPENAI_EMBEDDINGS_ADA_API_KEY=<your-embeddings-api-key>
AZURE_OPENAI_EMBEDDINGS_ADA_DEPLOYEMENT_NAME=<your-embeddings-deployment-name>
AZURE_OPENAI_API_EMBEDDINGS_ADA_ENDPOINT=<your-embeddings-endpoint>
AZURE_OPENAI_API_EMBEDDINGS_ADA_VERSION=<api-version>
```

**How to obtain Azure OpenAI credentials:**
1. Create an Azure account and set up Azure OpenAI service
2. Deploy a GPT-4 model (or similar) and an Text Embedding ADA model
3. Retrieve your API keys, endpoint URLs, and deployment names from the Azure portal
4. Paste them into the `.env` file

**Alternative: Using Google Gemini**

If you prefer to use **Google Gemini** instead of Azure OpenAI, you can use the `GOOGLE_API_KEY`:

```env
GOOGLE_API_KEY=<your-google-api-key>
```

The codebase includes commented-out code for Google Gemini integration in `llm/agents.py`, `llm/translator.py` and `llm/vector_store.py`. To switch to Gemini, uncomment the Google Gemini sections and comment out the Azure OpenAI sections in these files. Check the code comments for details on how to enable this alternative.

### 4. Run the Application

```bash
uv run main.py
```

## Usage

### Starting the Application

```bash
uv run main.py
```

You'll be prompted to enter a YouTube video URL:

```
Enter the YouTube video URL: https://www.youtube.com/watch?v=your_video_id
```

The application supports multiple URL formats:
- Standard watch URL: `https://www.youtube.com/watch?v=ID`
- Shorts URL: `https://www.youtube.com/shorts/ID`
- Shortened URL: `https://youtu.be/ID`
- Embed URL: `https://www.youtube.com/embed/ID`

### Interactive Chat Commands

Once the video is processed, interact with the AI agent using natural language queries:

#### Get Entire Video Summary
```
User: Summarize the entire video
```

#### Get Time-Segmented Summary
```
User: Summarize the video in 2-minute chunks
User: Give me summaries every 120 seconds
```

#### Ask Questions about Content
```
User: What is the main topic discussed?
User: What time does the discussion about X start?
User: Can you explain the concept of Y mentioned in the video?
User: What are the key takeaways from the video?
```

#### Get Information at Specific Timestamps
```
User: What happened at 5:30?
User: What does the video talk about at the 2-minute mark?
```

#### Exit the Application
```
User: bye
User: exit
```

## Recommendations

**📌 Use Long-Form YouTube Videos**

This application is optimized for **longer YouTube videos** (15+ minutes recommended). Here's why:

- **Better Context**: Longer videos provide more substantial content for meaningful summaries and Q&A
- **Improved RAG Performance**: The vector database benefits from richer, more diverse content across multiple segments
- **Cost Efficiency**: The tool caches transcriptions, translations, and vector stores, making subsequent queries more cost-effective on longer content
- **Time Segmentation**: Time-based summaries and timestamp queries work best when there's sufficient temporal content to segment
- **LLM Effectiveness**: The summarization and Q&A tools perform better with comprehensive information rather than minimal content

While the tool works with shorter videos, results will be more meaningful with extended content like tutorials, lectures, podcasts, or detailed explanations.

## Configuration Parameters

You can customize the processing behavior by modifying these parameters in `main.py`:

- `TRANSCRIBED_TEXT_TIME_DURATION` (default: 60 seconds) - Time window for grouping transcriptions
- `TRANSLATION_TIME_DURATION` (default: 120 seconds) - Time window for translation chunks
- `VECTOR_STORE_TIME_DURATION` (default: 120 seconds) - Time window for RAG document chunks
- `SUMMARIZATION_TIME_DURATION` (default: 120 seconds) - Time window for summarization chunks

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **youtube-transcript-api** | >=1.2.3 | Fetches transcripts directly from YouTube videos without needing external services |
| **langchain** | >=1.1.2 | Framework for building LLM applications with composable chains and agents |
| **langgraph** | >=1.0.4 | Builds stateful, multi-actor applications with LLMs using graph-based workflows |
| **langchain-community** | >=0.4.1 | Community integrations for LangChain (vector stores, tools, etc.) |
| **faiss-cpu** | >=1.13.1 | Meta's similarity search library for efficient vector indexing and retrieval |
| **deep-translator** | >=1.11.4 | Translates text between languages with support for multiple translation backends |
| **python-dotenv** | >=0.9.9 | Loads environment variables from `.env` files for secure credential management |

### LLM & Embedding Integrations

The project is configured to work with **Azure OpenAI APIs**:
- **GPT-4o** - For advanced language understanding, summarization, and reasoning
- **Text Embedding Ada** - For semantic similarity and vector embeddings

The codebase also includes commented-out support for **Google Gemini** (can be re-enabled if needed).

## Project Structure

```
youtube-video-summarizer/
├── main.py                 # Entry point for the application
├── utils.py               # Utility functions for URL parsing and data grouping
├── llm/
│   ├── agents.py         # AI agent definition with tools and workflow
│   ├── translator.py     # Translation logic for non-English transcripts
│   └── vector_store.py   # FAISS vector store creation and loading
├── transcriptions/        # Cached transcription files (auto-created)
├── translations/          # Cached translation files (auto-created)
├── db/                   # Vector database storage (auto-created)
├── .env.example          # Template for environment variables
├── pyproject.toml        # Project metadata and dependencies
└── README.md            # This file
```

## Workflow

1. **Transcription**: Extracts transcript from the YouTube video using the YouTube Transcript API
2. **Translation** (if needed): Translates non-English transcripts to English
3. **Chunking**: Groups transcription data into time-based segments for processing
4. **Vector Store Creation**: Embeds chunks and stores them in FAISS for semantic search
5. **Agent Setup**: Initializes an AI agent with tools for various operations
6. **Interactive Chat**: User can ask questions and get intelligent responses with timestamps

## Error Handling

The application gracefully handles:
- Invalid YouTube URLs
- Network connectivity issues
- Videos without available transcripts
- Missing or incomplete environment variables

## Caching

The application automatically caches:
- **Transcriptions** - Stored in `transcriptions/` folder
- **Translations** - Stored in `translations/` folder
- **Vector Databases** - Stored in `db/` folder

This prevents redundant API calls and speeds up subsequent queries for the same video.

## Limitations & Notes

- Requires video to have available transcripts (auto-generated or manual)
- Translation quality depends on the Azure OpenAI API's capabilities
- Timestamps are approximated based on the document chunks used in RAG
- Long videos may take time to process initially (subsequent queries are faster)

## Troubleshooting

### Issue: "Error during transcription"
- Verify the YouTube video ID is correct
- Check if the video has available captions/transcripts
- Ensure stable internet connection

### Issue: "Vector db is not present!"
- The vector store hasn't been created yet; this happens automatically during first run
- Ensure previous processing steps completed successfully

### Issue: API Key errors
- Verify `.env` file exists and is properly configured
- Double-check Azure OpenAI API keys and endpoints
- Ensure API resources are deployed and active in Azure portal

## Future Enhancements

- Support for multi-modal content (analyze video frames)
- Export summaries to PDF/Markdown
- Batch processing for multiple videos
- Enhanced translation with specialized domain models
- Custom summarization templates

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
