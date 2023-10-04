from transformers import AutoTokenizer, AutoModel
import torch
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from flask import Flask, request, render_template

# Read Data frames
offer_retailer_df = pd.read_csv('offer_retailer.csv')
categories_df = pd.read_csv('categories.csv')
brand_category_df = pd.read_csv('brand_category.csv')

# Remove any character that is not a word or space
def text_preprocess(text):
  text = re.sub(r'[^a-zA-Z\s]','',text)
  text = text.lower().strip()
  return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# List of all categories
categories = brand_category_df['BRAND_BELONGS_TO_CATEGORY'].dropna().unique().tolist()
categories = [text_preprocess(i) for i in categories]

# List of all retailers
retailers = offer_retailer_df['RETAILER'].dropna().unique().tolist()
retailers = [text_preprocess(i) for i in retailers]

# List of all brands
brands = offer_retailer_df['BRAND'].dropna().unique().tolist()
brands = [text_preprocess(i) for i in brands]

# Load the pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define a function to create embeddings for a list of words in a class
def create_embeddings(word_list):
    embeddings = []
    for word in word_list:
        word_tokens = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            word_output = model(**word_tokens)
        word_embedding = word_output.last_hidden_state.mean(dim=1)
        embeddings.append(word_embedding)
    return embeddings

# Create embeddings for each class
category_embeddings = create_embeddings(categories)
brand_embeddings = create_embeddings(brands)
retailer_embeddings = create_embeddings(retailers)

# Define a function to calculate maximum cosine similarity with precomputed embeddings
def calculate_max_cosine_similarity(input_embeddings, class_embeddings, class_keywords):
    max_similarity = -1.0
    best_matching_keyword = None
    for i, word_embedding in enumerate(class_embeddings):
        similarity = cosine_similarity(input_embeddings, word_embedding).item()
        if similarity > max_similarity:
            max_similarity = similarity
            best_matching_keyword = class_keywords[i]
    return max_similarity, best_matching_keyword

# Define a function to create input embeddings
def create_input_embeddings(input_text):
  input = text_preprocess(input_text)
  processed_input = remove_stopwords(input)
  # Tokenize and obtain embeddings for the input text
  input_tokens = tokenizer(processed_input, return_tensors="pt", padding=True, truncation=True)
  with torch.no_grad():
      input_output = model(**input_tokens)
  input_embeddings = input_output.last_hidden_state.mean(dim=1) 
  return input_embeddings

# Fetch offers based on retail name
def retailer_offers(retailer_name):
  offer_retailer_df['RETAILER'] = offer_retailer_df['RETAILER'].dropna().apply(text_preprocess)
  offers = offer_retailer_df['OFFER'][offer_retailer_df['RETAILER']==retailer_name]
  return offers

# Fetch offers based on brand name
def brand_offers(brand_name):
  offer_retailer_df['BRAND'] = offer_retailer_df['BRAND'].dropna().apply(text_preprocess)
  offers = offer_retailer_df['OFFER'][offer_retailer_df['BRAND']==brand_name]
  return offers

# Fetch offers based on category name
def category_offers(category_name):
  brand_category_df['BRAND_BELONGS_TO_CATEGORY'] = brand_category_df['BRAND_BELONGS_TO_CATEGORY'].dropna().apply(text_preprocess)
  offer_retailer_df['BRAND'] = offer_retailer_df['BRAND'].dropna().apply(text_preprocess)
  brand_category_df['BRAND'] = brand_category_df['BRAND'].dropna().apply(text_preprocess)
  brand = brand_category_df['BRAND'][brand_category_df['BRAND_BELONGS_TO_CATEGORY'] == category_name].tolist()
  offer_brands = [item for item in brand if item in brands]
  if len(offer_brands) == 0:
    return pd.Series(["The brands that belong to this category do not have offers currently!"])
  else:
    offer = offer_retailer_df['OFFER'][offer_retailer_df['BRAND'].isin(offer_brands)]
    return offer

# Read all the category offers and fetch relevant offers 
def relevant_offers(cat_offers, max_key):
  cosine_sim = {}
  key_embeddings = create_input_embeddings(max_key)
  for offer in cat_offers:
    offer_embeddings = create_input_embeddings(offer)
    cosine_score = cosine_similarity(offer_embeddings,key_embeddings)
    cosine_sim[offer] = cosine_score
  sorted_offers = sorted(cosine_sim.items(), key=lambda x: x[1], reverse=True)
  off = [i[0] for i in sorted_offers]
  return off

def prediction(input_embeddings):
  # Calculate maximum cosine similarities with precomputed class embeddings
  max_similarity_category, best_matching_category = calculate_max_cosine_similarity(input_embeddings, category_embeddings, categories)
  max_similarity_brand, best_matching_brand = calculate_max_cosine_similarity(input_embeddings, brand_embeddings, brands)
  max_similarity_retailer, best_matching_retailer = calculate_max_cosine_similarity(input_embeddings, retailer_embeddings, retailers)
  # Determine the class with the highest maximum similarity
  max_similarity = max(max_similarity_category, max_similarity_brand, max_similarity_retailer)
  score = 0
  key = "existing Category/Brand/Retailer"
  if max_similarity < 0.65:
    offers = ["No related offers"]
  else:
    if max_similarity == max_similarity_category:
      score = max_similarity_category
      key = best_matching_category
      off = category_offers(best_matching_category)
      offers = relevant_offers(off,best_matching_category)

    elif max_similarity == max_similarity_brand:
      score = max_similarity_brand
      key = best_matching_brand
      offers = brand_offers(best_matching_brand)
    else:
      score = max_similarity_retailer
      key = best_matching_retailer
      offers = retailer_offers(best_matching_retailer)

  return(score,key,offers)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/currentoffers', methods=['POST'])
def currentoffers():
    user_input = request.form['user_input']
    user_embeddings = create_input_embeddings(user_input)
    sim_score,cat,offers = prediction(user_embeddings)
    show_line = True
    return render_template('index.html', output=offers,score=round(sim_score,2),cat=cat,show_line=show_line)

if __name__ == '__main__':
    app.run()

  