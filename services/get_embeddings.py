import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = BertModel.from_pretrained('indobenchmark/indobert-base-p1', output_hidden_states=True)

def document_mean_pooling_bert(document):
    """
    Calculate mean pooling of the BERT embeddings for a list of sentences.

    Arguments:
    - document: document to encode.

    Returns:
    - A single numpy array representing the mean pooling of the BERT embeddings for the document.
    """

    # Concatenate the sentences into a single string
    # document = " ".join(sentences)

    # Encode the document using BERT
    input_id = tokenizer.encode(document, return_tensors="pt")

    # Pass the input_id through the BERT model
    with torch.no_grad():
        output = model(input_id)

    # Extract the hidden states of the tokens in the document
    hidden_states = output[2][-1]

    # Calculate the mean pooling of the hidden states for the document
    document_embedding = torch.mean(hidden_states, dim=1)

    return document_embedding


def get_bert_embeddings(text):
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    
    with torch.no_grad():
        outputs = model(tokens_tensor)
        last_hidden_state = outputs[-1]

    token_vecs = last_hidden_state[-2][0]
    # Get the vector representation with 1x768 dimension
    embeddings = torch.mean(token_vecs, dim=0)
    return embeddings

'''
# Function to get the BERT embeddings for a given text
def get_mean_pooled_bert_embeddings(text):
    # Tokenize the text
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    # Get the BERT embeddings
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0][:, 0, :]
    # Mean pool the embeddings to get a single vector representation
    mean_pooled_embeddings = torch.mean(last_hidden_states, dim=1)
    return mean_pooled_embeddings
'''


