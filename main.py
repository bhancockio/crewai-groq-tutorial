import os
import time
from crewai import Crew, Process
from langchain_groq import ChatGroq
from agents import NewsAggregatorAgents
from tasks import NewsAggregatorTasks

# 0. Setup environment
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
)
topics = ["tech", "politics"]

# 1. Create agents
agents = NewsAggregatorAgents()

news_collector = agents.news_collector(topics=topics)
news_collector_for_topic = agents.news_collector_for_topic()
news_report_compiler = agents.news_report_compiler()

# 2. Create tasks
tasks = NewsAggregatorTasks()


news_research_tasks = []
for topic in topics:
    news_research_tasks.append(
        tasks.research_news_for_topic(agent=news_collector_for_topic, topic=topic))

news_manager_task = tasks.news_manager(
    agent=news_collector, topics=topics, research_tasks=news_research_tasks)

compile_news_report = tasks.compile_news_report(
    agent=news_report_compiler, topics=topics, finalized_news=news_manager_task)

# Setup Crew
crew = Crew(
    agents=[
        news_collector,
        news_collector_for_topic,
        news_report_compiler
    ],
    tasks=[
        *news_research_tasks,
        news_manager_task,
        compile_news_report
    ],
    # max_rpm=29
)

# Kick off the crew
start_time = time.time()

results = crew.kickoff()

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Crew kickoff took {elapsed_time} seconds.")
print("Crew usage", crew.usage_metrics)
