from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
import pickle
import torch

app = Flask(__name__)

# Load the model and embeddings
embeddings = pickle.load(open('models/embeddings.pkl', 'rb'))
sentences = pickle.load(open('models/sentences.pkl', 'rb'))
rec_model = pickle.load(open('models/rec_model.pkl', 'rb'))

def recommendation(input_paper):
    # Calculate cosine similarity scores between the embeddings of input_paper and all papers in the dataset.
    cosine_scores = util.cos_sim(embeddings, rec_model.encode(input_paper))

    # Get the indices of the top-k most similar papers based on cosine similarity.
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=5, sorted=True)

    # Retrieve the titles of the top similar papers.
    papers_list = []
    for i in top_similar_papers.indices:
        papers_list.append(sentences[i.item()])

    return papers_list

@app.route('/')
def home():
    return render_template('index1.html')
