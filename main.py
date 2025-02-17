import os
import random
import pandas as pd
from dotenv import load_dotenv
import openai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables from the .env file
load_dotenv()

# OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment.")
openai.api_key = openai_api_key

# SMTP configuration
smtp_host = "smtp.gmail.com"
smtp_port = 587
smtp_username = os.getenv("SMTP_USERNAME")
smtp_password = os.getenv("SMTP_PASSWORD")
sender_email = os.getenv("SENDER_EMAIL")
receiver_email = os.getenv("RECEIVER_EMAIL")

for x in [smtp_host, smtp_username, smtp_password, sender_email, receiver_email]:
    print(x)

# Sample hospital names and branch names
hospitals = [
    "General Hospital", 
    "St. Mary's Hospital", 
    "City Hospital", 
    "County Medical Center", 
    "Regional Hospital"
]

branches = [
    "North Wing", 
    "South Wing", 
    "East Wing", 
    "West Wing", 
    "Central"
]

# Dictionary of Nelson rules and their descriptions.
nelson_rules = {
    "Rule 1": "One point is more than 3 standard deviations from the mean.",
    "Rule 2": "Nine or more consecutive points on the same side of the mean.",
    "Rule 3": "Six consecutive points steadily increasing or decreasing.",
    "Rule 4": "Fourteen consecutive points alternating up and down.",
    "Rule 5": "Two out of three consecutive points are more than 2 standard deviations from the mean on the same side.",
    "Rule 6": "Four out of five consecutive points are more than 1 standard deviation from the mean on the same side.",
    "Rule 7": "Fifteen consecutive points fall within 1 standard deviation of the mean.",
    "Rule 8": "Eight consecutive points are more than 1 standard deviation from the mean, with none within 1 SD."
}

# Generate sample DataFrame data
num_rows = 10
data = []
for _ in range(num_rows):
    rule = random.choice(list(nelson_rules.keys()))
    hospital = random.choice(hospitals)
    branch = random.choice(branches)
    transaction = random.randint(1000, 9999)  # Random transaction ID
    description = nelson_rules[rule]
    
    data.append({
        "Rule": rule,
        "Hospital": hospital,
        "Branch": branch,
        "Transaction": transaction,
        "Description": description
    })

df = pd.DataFrame(data)

# Construct the prompt for ChatGPT
prompt = f"""
You are an experienced data analyst and communications specialist. I have a DataFrame containing quality control events, where each record has the following fields:

- Rule: The specific Nelson rule that was broken.
- Hospital: The name of the hospital.
- Branch: The branch location.
- Transaction: A unique transaction ID.
- Description: A detailed description of the broken rule.

Below is the DataFrame data:

{df.to_string(index=False)}

Please analyze this DataFrame to identify any significant trends, anomalies, or patterns—such as frequently occurring rules or issues concentrated in specific hospitals or branches. Based on your analysis, draft a professional email addressed to senior management that summarizes the key findings and includes any recommendations or next steps.

The email should include:
- A clear subject line (e.g., "Important Findings on Quality Control Metrics").
- A professional greeting.
- A concise overview of the analysis and significant observations, using bullet points if necessary.
- Any actionable recommendations or next steps.
- A professional closing.

Please ensure the tone is formal and appropriate for communicating critical quality control issues to a senior audience.
"""

# Call the OpenAI ChatCompletion API to generate the email content
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
)

email_content = response.choices[0].message['content']
print("Generated Email:\n")
print(email_content)

# Prepare the email message using MIME
msg = MIMEMultipart()
msg["From"] = sender_email
msg["To"] = receiver_email
msg["Subject"] = "Important Findings on Quality Control Metrics"  # You can modify the subject if needed

msg.attach(MIMEText(email_content, "plain"))

# Send the email using SMTP
try:
    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()  # Secure the connection
        server.login(smtp_username, smtp_password)
        server.send_message(msg)
    print("Email sent successfully!")
except Exception as e:
    print("Error sending email:", e)
