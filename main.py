import os
import time
import csv
from crewai import Crew
from langchain_groq import ChatGroq
from agents import EmailPersonalizationAgents
from tasks import PersonalizeEmailTask

# 0. Setup environment
from dotenv import load_dotenv
load_dotenv()

email_template = """
Hey [Name]!

Just a quick reminder that we have a Skool community where you can 
join us for weekly coaching calls every Tuesday at 6 PM Eastern time.
The community is completely free and we're about to hit the 500
user milestone. We'd love to have you join us!

If you have any questions or need help with your projects, 
this is a great place to connect with others and get support. 

If you're enjoying the AI-related content, make sure to check out 
some of the other videos on my channel. Don't forget to hit that 
like and subscribe button to stay updated with the latest content. 
Looking forward to seeing you in the community!

Best regards,
Brandon Hancock
"""

# 1. Create agents
agents = EmailPersonalizationAgents()

email_personalizer = agents.personalize_email_agent()
ghostwriter = agents.ghostwriter_agent()

# 2. Create tasks
tasks = PersonalizeEmailTask()

personalize_email_tasks = []
ghostwrite_email_tasks = []

# Path to the CSV file containing client information
csv_file_path = 'data/clients_small.csv'

# Open the CSV file
with open(csv_file_path, mode='r', newline='') as file:
    # Create a CSV reader object
    csv_reader = csv.DictReader(file)

    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Access each field in the row
        recipient = {
            'first_name': row['first_name'],
            'last_name': row['last_name'],
            'email': row['email'],
            'bio': row['bio'],
            'last_conversation': row['last_conversation']
        }

        # Create a personalize_email task for each recipient
        personalize_email_task = tasks.personalize_email(
            agent=email_personalizer,
            recipient=recipient,
            email_template=email_template
        )

        # Create a ghostwrite_email task for each recipient
        ghostwrite_email_task = tasks.ghostwrite_email(
            agent=ghostwriter,
            draft_email=personalize_email_task,
            recipient=recipient
        )

        # Add the task to the crew
        personalize_email_tasks.append(personalize_email_task)
        ghostwrite_email_tasks.append(ghostwrite_email_task)


# Setup Crew
crew = Crew(
    agents=[
        email_personalizer,
        ghostwriter
    ],
    tasks=[
        *personalize_email_tasks,
        *ghostwrite_email_tasks
    ],
    max_rpm=29
)

# Kick off the crew
start_time = time.time()

results = crew.kickoff()

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Crew kickoff took {elapsed_time} seconds.")
print("Crew usage", crew.usage_metrics)
