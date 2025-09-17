# AI Task Planner Streamlit App

This Streamlit application provides an interactive interface for planning complex tasks using AI. The app breaks down high-level tasks into actionable subtasks and incorporates best practices from web search results.
<img width="1366" height="722" alt="image" src="https://github.com/user-attachments/assets/95d8d76f-0ccc-4ba1-90ad-f185975b2131" />
<img width="814" height="633" alt="image" src="https://github.com/user-attachments/assets/983b442f-d271-4a40-b100-83f185f82e8e" />

## Features

- **Task Breakdown**: Automatically splits complex tasks into 5-7 actionable subtasks
- **Web Research**: Searches for current best practices related to your task
- **Task Refinement**: Refines initial subtasks based on search results
- **Downloadable Results**: Save your task plan as a JSON file

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ai-task-planner.git
cd ai-task-planner
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## API Keys

You'll need to sign up for the following services and obtain API keys:

- [Groq](https://console.groq.com/): Sign up and create an API key
- [Tavily](https://tavily.com/#api): Sign up for Tavily's search API

## Running the App

```bash
streamlit run app.py
```

This will start the Streamlit development server and open the app in your default web browser.

## Deployment

The app can be deployed to Streamlit Cloud:

1. Create a [Streamlit Cloud](https://streamlit.io/cloud) account
2. Connect your GitHub repository
3. Add your API keys as secrets in the Streamlit Cloud dashboard
4. Deploy!

## Architecture

This app uses:

- **LangGraph**: For orchestrating the multi-step planning workflow
- **LangChain**: For connecting to the Groq LLM and Tavily search
- **Groq LLM**: For natural language processing and task planning
- **Tavily**: For performing web searches to find best practices
- **Streamlit**: For the web interface

## License

MIT
