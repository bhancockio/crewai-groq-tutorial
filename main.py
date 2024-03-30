from crewai import Crew, Process
from agents import BookSummarizingAgents
from tasks import BookSummarizingTasks
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

# Initialize LLM
OpenAIGPT4 = ChatOpenAI(
    model="gpt-4",
)

# 1. Create agents
agents = BookSummarizingAgents()

book_summarizer = agents.book_summarizer()
framework_extractor = agents.framework_extractor()
action_items_agent = agents.action_items_agent()
quotes_extractor = agents.quotes_extractor()
chapter_summarizer = agents.chapter_summarizer()

# Book title
book_title = "The 7 Habits of Highly Effective People"

# 2. Creates tasks
tasks = BookSummarizingTasks()

summarize_book_task = tasks.summarize_book(
    book_title=book_title,
    agent=book_summarizer,
)
extract_frameworks_task = tasks.extract_frameworks(
    book_title=book_title,
    agent=framework_extractor,
)
identify_action_items_task = tasks.identify_action_items(
    book_title=book_title,
    agent=action_items_agent,
)
extract_quotes_task = tasks.extract_quotes(
    book_title=book_title,
    agent=quotes_extractor,
)
summarize_chapters_task = tasks.summarize_chapters(
    book_title=book_title,
    agent=chapter_summarizer,
)

# Setup Crew
crew = Crew(
    agents=[
        book_summarizer,
        framework_extractor,
        action_items_agent,
        quotes_extractor,
        chapter_summarizer
    ],
    tasks=[
        summarize_book_task,
        extract_frameworks_task,
        identify_action_items_task,
        extract_quotes_task,
        summarize_chapters_task
    ],
)

# Kick off the crew
results = crew.kickoff()

print("Crew usage", crew.usage_metrics)

print("Crew work results:")
print(results)
