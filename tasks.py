from textwrap import dedent
from crewai import Task


class BookSummarizingTasks():

    def format_book_title(self, book_title):
        return book_title.replace(" ", "_")

    def summarize_book(self, book_title, agent):
        return Task(
            description=dedent(f"""Summarize the main ideas and key takeaways from the self-help book '{book_title}' to provide a concise overview.
                
                Please provide a summary that captures the essence of the book, highlighting its main points and key messages.
                """),
            agent=agent,
            output_file=f"output/{self.format_book_title(book_title)}_Summary.md",
            expected_output=dedent("""
                A summary that is concise and captures the main ideas and key takeaways of the book.
                """)
        )

    def extract_frameworks(self, book_title, agent):
        return Task(
            description=dedent(f"""Identify and extract any frameworks, models, or structured approaches presented in the book '{book_title}'.
                
                Please outline the frameworks or models that the author uses to convey their ideas, making them easily accessible to readers.
                """),
            agent=agent,
            output_file=f"output/{self.format_book_title(book_title)}_Frameworks.md",
            expected_output=dedent("""
                A list of frameworks, models, or structured approaches extracted from the book.
                """)
        )

    def identify_action_items(self, book_title, agent):
        return Task(
            description=dedent(f"""Identify actionable advice or steps that readers can take based on the content of the book '{book_title}'.
                
                Please extract practical steps or advice from the book, providing readers with clear guidance on how to apply the book's teachings in their lives.
                """),
            agent=agent,
            output_file=f"output/{self.format_book_title(book_title)}_Action_Items.md",
            expected_output=dedent("""
                A list of actionable steps or advice extracted from the book.
                """)
        )

    def extract_quotes(self, book_title, agent):
        return Task(
            description=dedent(f"""Extract notable quotes or key phrases from the book '{book_title}' for quick reference.
                
                Please identify and extract memorable or significant quotes from the book, highlighting the author's most impactful words.
                """),
            agent=agent,
            output_file=f"output/{self.format_book_title(book_title)}_Quotes.md",
            expected_output=dedent("""
                A list of notable quotes or key phrases extracted from the book.
                """)
        )

    def summarize_chapters(self, book_title, agent):
        return Task(
            description=dedent(f"""Provide a summary for each chapter of the book '{book_title}', helping readers understand the focus and flow of the content.
                
                Please break down the book into manageable sections, summarizing each chapter to give readers an overview of its content and purpose.
                """),
            agent=agent,
            output_file=f"output/{self.format_book_title(book_title)}_Chapter_Summaries.md",
            expected_output=dedent("""
                A summary for each chapter of the book, providing an overview of its content and purpose.
                """)
        )
