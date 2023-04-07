from flask import Flask, request
from get_embeddings import mean_pooling_bert
from similar_documents_retrieval import get_similar_docs_by_cos_sims
from detailed_analysis import get_plagiarised_pairs
from datetime import datetime
from program import detailed_plagiarism
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})

@app.route('/perform-plagiarism-detailed-analysis', methods=['POST'])
def plagiarism_detailed_analysis():
    '''
    Input from API body:
    {'text': 'lorem ipsum lorem ipsum 1000x'}
    '''
    inputted_text = request.get_json()
    print(inputted_text)
    text = inputted_text['text']
    plagiarism_cases = detailed_plagiarism(sus_text=text, n=5, threshold=.85, clustering=True)
    print(plagiarism_cases)
    return plagiarism_cases

if __name__ == '__main__':
    app.run(debug=True)