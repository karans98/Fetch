# NLP using HuggingFace

## Problem statement - 

To build a tool that allows users to intelligently search for offers via text input from the user.

- If a user searches for a category (ex. diapers) the tool should return a list of offers that are relevant to that category.
- If a user searches for a brand (ex. Huggies) the tool should return a list of offers that are relevant to that brand.
- If a user searches for a retailer (ex. Target) the tool should return a list of offers that are relevant to that retailer.
- The tool should also return the score that was used to measure the similarity of the text input with each offer

## Approach - 

To address this problem we can build a Flask-based web application that leverages BERT embeddings and cosine similarity to provide users with relevant retail offers based on their input, whether it's a category, brand, or retailer name.

### Steps involved - 

1. Load the 3 csv files as Pandas DataFrames and create a list of relevant categories, brands, and retailers. 
2. Define a custom text pre-processing function that removes any character that is not a space or word
3. Load the pre-trained BERT model (bert-base-uncased) and tokenizer from the Hugging Face Transformers library.
4. Define a function to create embeddings for each class
5. Compute the cosine similarity between user input embeddings and embeddings of categories, brands, and retailers to identify the most similar category, brand, or retailer.
6. Once the input text has been classified, we can define functions to fetch offers from the dataframe.
7. Additionally, create a function to fetch categorical relevant offers by computing the cosine similarity score for offers that belong to the brand.
8. Create a Flask web application rendering an HTML template. The application can be run locally using the app.run() method, making it accessible through a web browser.

Hence, once the user searches by category/retailer/brand, the list of relevant offers is populated.

### Additional Points
- In a production pipeline, especially with large datasets, we can use a database management system (e.g., SQL database) for efficient querying and indexing. We can implement rigorous testing and use version control to track and facilitate collaboration. We can plan for horizontal scaling to handle increased traffic.

- For receipt classification or entity recognition, since the BERT model is already present in the pipeline we can fine-tune the BERT model for text classification or Named Entity Recognition (NER). 


## How to run - 

1. Clone this repository :[git clone https://github.com/karans98/NLP-HuggingFace.git]
2. Once cloning is complete, navigate to the cloned repository :[cd NLP-HuggingFace]
3. Install the Python libraries mentioned in the requirements.txt file :[pip install -r requirements.txt]
4. Run the Python file :[python app.py]
5. Click the link and query for related offers
