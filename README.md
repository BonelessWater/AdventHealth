# Hospital Finance Anomaly Email Report Generator

## Overview

This script is designed to **automatically generate professional email reports** on hospital finance anomalies by summarizing data and leveraging the OpenAI GPT model to draft the email.

### Key Features
- Summarizes anomaly data using a variety of techniques, including descriptive statistics, frequency analysis, topic modeling, and sentiment analysis.
- Automatically generates professional email content with a polished format.
- Supports both automated email delivery via SMTP and debug mode for offline email output.

---

## Workflow

The system follows a structured process consisting of six steps:

### 1. **Environment Setup**
- Loads environment variables from a `.env` file using `load_dotenv()`.
- Configures the OpenAI API key.
- Sets up SMTP credentials for email sending (SMTP host, port, username, password, sender email, and receiver email).

### 2. **Data Reading and Preprocessing**
- Reads anomaly data from a CSV file (`ANOMALY_TABLE.csv`) into a Pandas DataFrame.
- Dataset requirements:
  - The **first column** should contain numeric data (e.g., dollar amounts).
  - The **remaining columns** should contain descriptive string data.

### 3. **Data Summarization**
The system generates a concise and informative summary of the input data, including:

#### a. **Numeric Summary**
- Computes basic descriptive statistics (mean, minimum, and maximum) for the numeric column.

#### b. **Raw Data Overview**
- Provides an overview of the dataset, including the total number of rows and columns and an example row.

#### c. **Frequency Analysis for String Columns**
- Identifies the top 2 most common values in each string column, along with their frequency counts.

#### d. **Group-By Aggregation**
- If a "Department" column exists, performs a group-by operation on the numeric column to compute:
  - **Mean** values by department.
  - **Sum** totals by department.

#### e. **Text Analysis**
- Combines all string columns into a single text field per row (up to 1000 rows to manage token usage).
- Performs keyword extraction using `CountVectorizer` to identify the top 10 keywords.
- Applies **Latent Dirichlet Allocation (LDA)** for topic modeling, extracting 3 topics with the top 3 associated keywords per topic.
- Conducts **sentiment analysis** using NLTK's VADER, generating average sentiment scores for the sampled text.

#### f. **Final Summary**
- Combines all the above analyses into a concise summary string for downstream use.

### 4. **Prompt Construction**
- Constructs a detailed prompt for the GPT model, incorporating the summarized data and email formatting instructions.
- Includes an example email template for reference.

### 5. **Email Generation**
- Sends the prompt to the OpenAI ChatCompletion API (`gpt-3.5-turbo`) to generate email content.
- Displays the generated email for verification.

### 6. **Email Delivery or Debug Output**
- **Debug Mode:** When the `--debug` flag is set, saves the generated email as a text file with a UUID-based filename in the `outputs/` directory.
- **Email Mode:** Sends the generated email via SMTP using the configured credentials.
