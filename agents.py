from crewai import Agent


class BookSummarizingAgents():
    def book_summarizer(self):
        return Agent(
            role="Book Summarizer",
            goal="""Summarize the main ideas and key takeaways from a self-help book to provide a concise overview.""",
            backstory="""As a Book Summarizer, you are responsible for extracting the essence of the book, highlighting its main points and providing a summary that captures the author's key messages.""",
            allow_delegation=True,
            verbose=True
        )

    def framework_extractor(self):
        return Agent(
            role="Framework Extractor",
            goal="""Identify and extract any frameworks, models, or structured approaches presented in the book.""",
            backstory="""As a Framework Extractor, you are responsible for recognizing and outlining the frameworks or models that the author uses to convey their ideas, making them easily accessible to readers.""",
            allow_delegation=True,
            verbose=True
        )

    def action_items_agent(self):
        return Agent(
            role="Action Items Agent",
            goal="""Identify actionable advice or steps that readers can take based on the book's content.""",
            backstory="""As an Action Items Agent, you are responsible for extracting practical steps or advice from the book, providing readers with clear guidance on how to apply the book's teachings in their lives.""",
            allow_delegation=True,
            verbose=True
        )

    def quotes_extractor(self):
        return Agent(
            role="Quotes Extractor",
            goal="""Extract notable quotes or key phrases from the book for quick reference.""",
            backstory="""As a Quotes Extractor, you are responsible for identifying and extracting memorable or significant quotes from the book, highlighting the author's most impactful words.""",
            allow_delegation=True,
            verbose=True
        )

    def chapter_summarizer(self):
        return Agent(
            role="Chapter Summarizer",
            goal="""Provide a summary for each chapter of the book, helping readers understand the focus and flow of the content.""",
            backstory="""As a Chapter Summarizer, you are responsible for breaking down the book into manageable sections, summarizing each chapter to give readers an overview of its content and purpose.""",
            allow_delegation=True,
            verbose=True
        )
