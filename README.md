# Insurance Chatbot with Langchain and Pinecone

This implements a chatbot that utilizes Sentence Transformation and OpenAI's GPT-4o model to inform about insurance Policies in europe and Digitilization of insurance Industry. The chatbot aims to provide relevant responses to user queries by refining and enhancing their input queries, finding similar sentences using Sentence Transformation, and generating more contextually accurate conversation logs.

#### Given a knowledge base whose vectors are stored in a pinecone, the chatbot provides answers to the questions that are most relevant to the context.

![Screenshot 2023-08-08 184514](https://github.com/bbabina/Chatbot-with-Langchain-and-Pinecone/assets/74191100/4a7a22dc-ac70-426d-8765-0ef7f5c8f9b7)

#### Overview of data used to train the chatbot
![Screenshot 2023-08-08 184514](https://github.com/arjun8977/insurance/blob/main/Training_Data.png)


### Dependencies

The code uses the following Python libraries:

- `sentence_transformers`: Used to encode input sentences and find similar sentences.
- `pinecone`: Used to create and query an index of similar sentences.
- `openai`: Used to interact with the OpenAI GPT-3 model.
- `streamlit`: Used to create a user-friendly web application for interacting with the chatbot.

## Getting Started

### Installation

To set up the Language Chain Chatbot on your local machine, follow these steps:

1. Create a virtual environment

   ```bash
   virtualenv venv
   venv\Scripts\activate
   ```


2. Obtain API keys:

   - OpenAI API Key: Get your API key from the OpenAI platform and update it in `app.py` and `Store_Vectors_in_Pinecone.py`.
   - Pinecone API Key: Get your API key from the Pinecone platform and update it in `Store_Vectors_in_Pinecone.py`.

### Usage

Run the application using the following command:

```bash
streamlit run app.py
```

This will launch a Streamlit application where you can interact with the chatbot.
