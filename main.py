import os
import pandas as pd
import uuid
import argparse
import openai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import numpy as np

# Text analysis imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Import the example email for formatting reference
from example import example_email
EXAMPLE_EMAIL = example_email

def summarize_data(df):
    # --- Numeric Summary (Concise) ---
    numeric_col = df.columns[0]  # Assumes first column holds dollar amounts.
    numeric_mean = df[numeric_col].mean()
    numeric_min = df[numeric_col].min()
    numeric_max = df[numeric_col].max()
    numeric_summary = f"{numeric_col}: mean={numeric_mean:.2f}, min={numeric_min}, max={numeric_max}"

    # --- Raw Data Overview ---
    raw_overview = f"Dataset: {df.shape[0]} rows, {df.shape[1]} cols. Example row: {df.iloc[0].to_dict()}"

    # --- Frequency Analysis for String Columns (Top 2 per column) ---
    freq_summary = ""
    for col in df.columns[1:]:
        top_vals = df[col].value_counts().head(2).to_dict()
        freq_summary += f"{col}: {top_vals}; "
    
    # --- Group-By Aggregation if "Department" exists ---
    if "Department" in df.columns:
        group_df = df.groupby("Department")[numeric_col].agg(['mean', 'sum']).reset_index().head(2)
        group_summary = group_df.to_string(index=False)
    else:
        group_summary = "No Department column."

    # --- Text Analysis on String Columns ---
    # Combine text from all string columns (limit to 1000 rows)
    text_data = df[df.columns[1:]].astype(str).agg(" ".join, axis=1).head(1000)
    
    # Keyword Extraction using CountVectorizer (Top 10 keywords)
    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    X = vectorizer.fit_transform(text_data)
    keywords = vectorizer.get_feature_names_out()
    keyword_summary = "Keywords: " + ", ".join(keywords)
    
    # Topic Modeling using LDA (3 topics, Top 3 words per topic)
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(X)
    
    def get_topics(model, vec, n_top=3):
        topics = []
        feature_names = vec.get_feature_names_out()
        for topic in model.components_:
            top_indices = topic.argsort()[-n_top:][::-1]
            topics.append(", ".join(feature_names[i] for i in top_indices))
        return topics
    
    topics = get_topics(lda, vectorizer)
    topic_summary = "; ".join([f"Topic {i+1}: {t}" for i, t in enumerate(topics)])
    
    # Sentiment Analysis using NLTK VADER on the text sample
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    sentiments = text_data.apply(lambda x: sia.polarity_scores(x))
    avg_sentiment = pd.DataFrame(list(sentiments)).mean().to_dict()
    sentiment_summary = f"Avg Sentiment: {avg_sentiment}"

    # Combine all into one concise summary string
    final_summary = (
        f"Raw Overview: {raw_overview}\n"
        f"Numeric Summary: {numeric_summary}\n"
        f"Frequency Summary: {freq_summary}\n"
        f"Group Summary: {group_summary}\n"
        f"{keyword_summary}\n"
        f"Topic Summary: {topic_summary}\n"
        f"{sentiment_summary}"
    )
    return final_summary

def main(args):
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    openai.api_key = openai_api_key

    # SMTP configuration
    smtp_host = "smtp.gmail.com"
    smtp_port = 587
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    sender_email = os.getenv("SENDER_EMAIL")
    receiver_email = os.getenv("RECEIVER_EMAIL")
    
    for var in [smtp_host, smtp_username, smtp_password, sender_email, receiver_email]:
        print(var)
    
    # Read the anomaly data
    df = pd.read_csv("ANOMALY_TABLE.csv")
    data_summary = summarize_data(df)

    # Construct a concise prompt
    concise_prompt = f"""
You are an experienced data analyst and communications specialist. The dataset contains hospital finance anomalies.
Summary:
{data_summary}

Draft a professional email to senior management that:
- Has a clear subject line (e.g., "Important Findings on Hospital Finances"),
- Contains a professional greeting,
- Provides a concise overview of the key findings (using bullet points as needed),
- Lists actionable recommendations,
- Ends with a professional closing.

Use the following example for reference:
{EXAMPLE_EMAIL}
"""
    
    # Call the OpenAI ChatCompletion API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": concise_prompt}
        ]
    )
    
    generated_email = response.choices[0].message['content']
    print("Generated Email:\n", generated_email)
    
    if args.debug:
        # Write output to a file with a UUID-based filename in debug mode
        filename = f"outputs/{uuid.uuid4()}.txt"
        with open(filename, "w") as f:
            f.write(generated_email)
        print(f"Debug mode enabled. Email content written to file: {filename}")
    else:
        # Prepare and send the email via SMTP
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = "Important Findings on Hospital Finances"
        msg.attach(MIMEText(generated_email, "plain"))
        try:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
            print("Email sent successfully!")
        except Exception as e:
            print("Error sending email:", e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and send hospital finance anomaly email")
    parser.add_argument('--debug', action='store_true', help="If set, do not send email; output to a text file instead.")
    args = parser.parse_args()
    main(args)
