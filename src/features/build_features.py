
import numpy as np
from tqdm import tqdm

def preprocess_text(tokenizer, list_sentences, max_len) -> dict:
    """ 
    Function processes a list of sentences to tokenize the text (thanks to the tokenizer in parameter) and returns a dictionnary including :
    - the list of token id: key = 'input_ids'
    - the list of attention masks: key = 'attention_mask'
    """
    input_id = []
    attention_mask = []
    for text in tqdm(list_sentences):
        encode = tokenizer.encode_plus(text, 
                                        add_special_tokens = True, 
                                        padding="max_length", 
                                        truncation=True, 
                                        return_attention_mask = True, 
                                        return_token_type_ids = True,
                                        max_length=max_len)

        input_id.append(encode['input_ids'])
        attention_mask.append(encode['attention_mask'])
    results = {}
    results['input_ids'] = np.array(input_id)
    results['attention_mask'] = np.array(attention_mask)
    return results
