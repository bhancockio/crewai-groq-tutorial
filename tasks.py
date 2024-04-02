from crewai import Task
from datetime import datetime
from textwrap import dedent
from langchain_community.tools import DuckDuckGoSearchRun

from tools.search_tools import SearchTools


class NewsAggregatorTasks():
    def news_manager(self, agent, topics, research_tasks):
        return Task(
            description=dedent(
                f"""Manage the research and collection of the top three latest news articles related {topics} on {datetime.today().strftime('%Y-%m-%d')}. 
                    Each article should include a title, brief summary, and a link to the full article.

                    IMPORTANT:
                    - Only include articles from today: {datetime.today().strftime('%Y-%m-%d')}.
                    - If an article is older than today, discard the old article and keep searching until you find a relevant article.
                    """),
            agent=agent,
            expected_output=dedent("""
                A bulleted list for each topic that contains 3 news articles, each with a title, summary, and link.
                """),
            context=research_tasks
        )

    def research_news_for_topic(self, agent, topic):
        return Task(
            description=dedent(
                f"""Collect three  news articles related to {topic} on {datetime.today().strftime('%Y-%m-%d')}. 
                    Each article should include a title, brief summary, and a link to the full article.
                    
                    IMPORTANT:
                    - Only include articles from today. 
                    - If the article date is not today or if your missing info, keep searching until you find a relevant article.
                    - If you can't find the date of the article, skip it and keep searching.
                    - Provide a brief summary of each article. Two sentences max.
                    """),
            agent=agent,
            expected_output=dedent("""
                A bulleted list containing 3 news articles, each with a title, summary, and link.
                """),
            tools=[SearchTools.search_internet],
            async_execution=True,
        )

    def compile_news_report(self, agent, topics, finalized_news):
        return Task(
            description=dedent(
                f"""Compile the research for the {topics} topics into a Markdown report. 
                    The report should include a title, brief summary, and a link for each article.
                    
                     IMPORTANT:
                    - The final report should be well-structured and easy to read in Markdown format.
                        - Use headers, bullet points, and links to make the report visually appealing.
                    - You must include 3 articles for each topic.
                    - The report shouldn't include any other commentary or narration.
                    """),
            agent=agent,
            output_file="output/News_Report.md",
            expected_output=dedent("""
                A Markdown report with titles, summaries, and links for each tech and political news article.
                """),
            context=[finalized_news]
        )
