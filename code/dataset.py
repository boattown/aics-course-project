import pandas as pd
import torch
from transformers import VisualBertModel, BertModel, BertTokenizer

def clean_up_df(object_name):
    """Removes punctuation from object names

    Removes unnecessary punctuation and replaces underscore with space.

    Args:
        object_name: String. Applied to each object name in a DataFrame column.
    
    Returns:
        A string without punctuation.

    """

    clean_object_name = ''
    for char in object_name:
        if char == '_':
            clean_object_name += ' '
        elif char == '.':
            break
        else:
            clean_object_name += char

    return clean_object_name

def create_df(file):
    """Creates a pandas DataFrame with the data.

    Create a pandas DataFrame with the data of the file and applies clean_up_df to the object names.

    Args:
        file: String. The path of the data file.
    
    Returns:
        A DataFrame with names of objects, affordances and their annotation.

    """
    
    df = pd.read_csv(file)
    df.rename(columns = {'Unnamed: 0':'Object'}, inplace = True)
    df['Object'] = df['Object'].apply(clean_up_df)
    df.columns = ['Object','ImageNet synset','grasp','lift','throw','push','fix','ride','play','watch','sit on','feed','row','pour from','look through','write with', 'type on']

    return df

def get_lists_and_dicts(df):
    """Creates lists of objects and affordances and dictionaries for mapping them to an id.

    Create a pandas DataFrame with the data of the file and applies clean_up_df to the object names.

    Args:
        df: Pandas DataFrame.
    
    Returns:
        unique_objects: A list of unique objects.
        unique_affordances: A list of unique affordances.
        word_to_index: A dictionary that maps objects and affordances to an id.
        index_to_word: A dictionary that maps id to objects and affordances.
    
    """

    unique_objects = list(df['Object'])
    unique_affordances = [affordance.lower() for affordance in df.columns[2:]]

    word_to_index = {}
    index_to_word = {}

    for i, word in enumerate(unique_objects + unique_affordances):
        word_to_index[word] = i
        index_to_word[i] = word

    return unique_objects, unique_affordances, word_to_index, index_to_word

def get_gold_data(df):
    """Pairs objects with affordances and their truth value.

    Creates a list of tuples from the data in the DataFrame. Each tuple consists of object, affordance and truthvalue.

    Args:
        df: Pandas DataFrame.
    
    Returns:
        gold_data_pairs: A list of tuples.
    
    """
    
    gold_data_pairs = []
    for _, row in df.iterrows():
        for i, value in enumerate(row):
            if type(value) == str:
                pass
            else:
                gold_data_pairs.append((row[0],df.columns[i].lower(),value))
    return gold_data_pairs

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_string(text):
    """Pairs objects with affordances and their truth value.

    Adds [CLS] and [SEP] tokens to the string and tokenizes it.

    Args:
        text: String. The word to tokenize.
    
    Returns:
        String. The tokenized string.
    
    """
    
    marked_text = "[CLS] " + text + " [SEP]"
    return tokenizer.tokenize(marked_text)

def get_bert_embedding_dict(data_pairs):
    """Creates a dictionary that maps objects and affordances to BERT representations.

    Tokenizes objects and affordances and passes them to BERT. The mean of the penultimate layer 
    of the hidden state is used to represent the input. The words are mapped to their BERT representation in a dictionary.

    Args:
        data_pairs: List. The list of tuples with pairs of objects and affordances.
    
    Returns:
        Dictionary. A dictionary that maps objects and affordances to BERT's representations.
    
    """
    
    bert_model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True
                                  )

    bert_word_to_embedding = {}

    with torch.no_grad():
        
        for subset in data_pairs:
        
            for obj, affordance, _ in subset:

                if obj not in bert_word_to_embedding.keys():
                    tokenized_obj = tokenize_string(obj)
                    indexed_obj = tokenizer.convert_tokens_to_ids(tokenized_obj)
                    segments_ids = [1] * len(tokenized_obj)
                    tokens_tensor = torch.tensor([indexed_obj])
                    segments_tensor = torch.tensor([segments_ids])

                    outputs = bert_model(tokens_tensor, segments_tensor)
                    hidden_states = outputs[2]
                    token_vecs = hidden_states[-2][0] # I take the penultimate layer
                    obj_embedding = torch.mean(token_vecs, dim=0) # I take the mean over the vectors for each token to get a representation of the whole input

                    bert_word_to_embedding[obj] = obj_embedding

                if affordance not in bert_word_to_embedding.keys():
                    tokenized_affordance = tokenize_string(affordance)
                    indexed_affordance = tokenizer.convert_tokens_to_ids(tokenized_affordance)
                    segments_ids = [1] * len(tokenized_affordance)
                    tokens_tensor = torch.tensor([indexed_affordance])
                    segments_tensor = torch.tensor([segments_ids])

                    outputs = bert_model(tokens_tensor, segments_tensor)
                    hidden_states = outputs[2]
                    token_vecs = hidden_states[-2][0]
                    affordance_embedding = torch.mean(token_vecs, dim=0)

                    bert_word_to_embedding[affordance] = affordance_embedding

    return bert_word_to_embedding



def get_visual_bert_embedding_dict(data_pairs):
    """Creates a dictionary that maps objects and affordances to VisualBERT representations.

    Tokenizes objects and affordances and passes them to VisualBERT. The mean of the penultimate layer 
    of the hidden state is used to represent the input. The words are mapped to their VisualBERT representation in a dictionary.

    Args:
        data_pairs: List. The list of tuples with pairs of objects and affordances.
    
    Returns:
        Dictionary. A dictionary that maps objects and affordances to VisualBERT's representations.
    
    """
    
    visual_bert_model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre",output_hidden_states=True)

    visual_bert_word_to_embedding = {} # I create this embeddings dictionary so I can easily map words to embeddings

    with torch.no_grad():
        
        for subset in data_pairs:
        
            for obj, affordance, _ in subset:

                if obj not in visual_bert_word_to_embedding.keys():
                    tokenized_obj = tokenize_string(obj)
                    indexed_obj = tokenizer.convert_tokens_to_ids(tokenized_obj)
                    segments_ids = [1] * len(tokenized_obj)
                    tokens_tensor = torch.tensor([indexed_obj])
                    segments_tensor = torch.tensor([segments_ids])

                    outputs = visual_bert_model(tokens_tensor, segments_tensor)
                    hidden_states = outputs[2]
                    token_vecs = hidden_states[-2][0] # I take the penultimate layer
                    obj_embedding = torch.mean(token_vecs, dim=0) # I take the mean over the vectors for each token to get a representation of the whole input

                    visual_bert_word_to_embedding[obj] = obj_embedding

                if affordance not in visual_bert_word_to_embedding.keys():
                    tokenized_affordance = tokenize_string(affordance)
                    indexed_affordance = tokenizer.convert_tokens_to_ids(tokenized_affordance)
                    segments_ids = [1] * len(tokenized_affordance)
                    tokens_tensor = torch.tensor([indexed_affordance])
                    segments_tensor = torch.tensor([segments_ids])

                    outputs = visual_bert_model(tokens_tensor, segments_tensor)
                    hidden_states = outputs[2]
                    token_vecs = hidden_states[-2][0]
                    affordance_embedding = torch.mean(token_vecs, dim=0)

                    visual_bert_word_to_embedding[affordance] = affordance_embedding

    return visual_bert_word_to_embedding
