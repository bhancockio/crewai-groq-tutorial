from datetime import datetime
import os
from crewai import Agent
from langchain_groq import ChatGroq

from langchain_openai import ChatOpenAI


class NewsAggregatorAgents():
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
        )

        # self.llm = ChatOpenAI(
        #     model="gpt-4",
        # )

    def news_collector(self, topics):
        return Agent(
            role="News Collector",
            goal=f"""
                Manage the research and collection of the today's news related to the 
                following topics: {topics}. 

                Today's date is {datetime.today().strftime('%Y-%m-%d')}.

                Each article should include a title, published date, 
                brief summary, and a link to the full article.
                """,
            backstory="""
                As a News Collector, you are responsible for finding today's news
                on certain topics.
                You only report on today's news""",
            verbose=True,
            llm=self.llm
        )

    def news_collector_for_topic(self):
        return Agent(
            role="News Collector For Topic",
            goal=f"""
                Collect news articles from {datetime.today().strftime('%Y-%m-%d')} 
                related to a specified topic.""",
            backstory="""
                As a News Collector, you are responsible for 
                searching the internet to find today's news
                on a specified topic.

                You only report on today's news""",
            verbose=True,
            llm=self.llm
        )

    def news_report_compiler(self):
        return Agent(
            role="News Report Compiler",
            goal="""
                Compile the collected news articles into a Markdown report. 
                Don't summarize anything. Just format the info into markdown and save it.""",
            backstory="""
                As a News Report Compiler, you are responsible for organizing 
                the collected news articles into a coherent and well-structured
                Markdown report.
                """,
            verbose=True,
            llm=self.llm
        )
