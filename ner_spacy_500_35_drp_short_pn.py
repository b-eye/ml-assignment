import spacy
from spacy.training.example import Example
import random 
import pandas as pd 
from copy import deepcopy, copy
from spacy.util import minibatch, compounding
from spacy.language import Language, PipeCallable
import re
import matplotlib.pyplot as plt

EN_NLP      = spacy.load('en_core_web_sm')
STOPWORDS   = EN_NLP.Defaults.stop_words


def process_text(text:str):
    processed_text = list()

    for token in text.split():
        processed_token = ''.join(e.lower() for e in token if e.isalnum())
        if processed_token not in STOPWORDS:
            processed_text.append(processed_token)

    return ' '.join(processed_text)

def process_product_names(product_names:str, split_individual_tokens:bool = False):
    processed_product_names = list()

    for product_name in product_names.split(sep='@'):
        processed_product_name = product_name.lower()
        processed_product_name = process_text(processed_product_name)

        processed_product_names.append(processed_product_name)

        if split_individual_tokens:
            processed_product_names = processed_product_name.split()

    return processed_product_names

def create_labeled_data(data_frame:pd.DataFrame, product_list, max_count):
    count       = 0
    train_data  = list()

    for _, current_row_item in data_frame.iterrows():
        ent_dict = dict()

        if count < max_count:
            # review = process_review(item['review'])

            processed_text  = str(current_row_item['processed_text'])
            visited_items   = list()
            entities        = list()

            for token in processed_text.split():
                if token in product_list:
                    for i in re.finditer(token, processed_text):
                        if token not in visited_items:
                            entity = (i.span()[0], i.span()[1], 'PRODUCT')
                            visited_items.append(token)
                            entities.append(entity)

            if len(entities) > 0:
                ent_dict['entities'] = entities
                train_item = (processed_text, ent_dict)
                train_data.append(train_item)
                count+=1

    return train_data, count

def create_data_set(data_frame:pd.DataFrame):
    return_data_frame = data_frame.loc[data_frame['product_name'].str.len() > 1]
    return_data_frame = deepcopy(return_data_frame.loc[return_data_frame['text'].str.len() > 5])

    return return_data_frame, len(return_data_frame)


def create_blank_nlp_model():
    nlp_model = spacy.load("en_core_web_sm")

    # Add NER pipe
    if "ner" not in nlp_model.pipe_names:
        ner_pipe = nlp_model.create_pipe("ner")
        nlp_model.add_pipe('ner', last=True)
    else:
        ner_pipe = nlp_model.get_pipe("ner")

    return nlp_model, ner_pipe

def add_labels_to_ner_pipe(data_set, ner_pipe:PipeCallable):
    for text, annotations in data_set:
        for ent in annotations.get("entities"):
            ner_pipe.add_label(ent[2])

    doc = EN_NLP.make_doc(text)
    tags = spacy.training.offsets_to_biluo_tags(doc, annotations.get("entities"))

    for tag in tags:
        ner_pipe.add_label(tag)

    ner_pipe.add_label("PRODUCT")
    ner_pipe.add_label("B-PRODUCT")
    ner_pipe.add_label("O-PRODUCT")
    ner_pipe.add_label("I-PRODUCT")
    ner_pipe.add_label("U-PRODUCT")


def train_ner(train_data_set, n_iter):
    nlp_model, ner_pipe = create_blank_nlp_model()
    add_labels_to_ner_pipe(train_data_set, ner_pipe)
    # nlp_model.begin_training()

    loss_function_x = list()
    loss_function_y = list()

    other_pipes = [pipe for pipe in nlp_model.pipe_names if pipe != "ner"]
    with nlp_model.disable_pipes(*other_pipes):       
        

        for c_iter in range(n_iter):

            current_train_data = train_data_set
            random.shuffle(current_train_data)
            losses = {}

            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data_set, size=compounding(4.0, 32.0, 1.001))

            for batch in batches:
                examples_list = list()
                for text, annotation in batch:
                    current_doc       = nlp_model.make_doc(text)
                    '''
                    spans_list = list()
                    with current_doc.retokenize() as retokenizer:
                        for entity in annotation['entities']:
                            start_idx = entity[0]
                            end_idx   = entity[1]
                            label_idx = entity[2]

                            current_span = current_doc.char_span(start_idx, end_idx, label=label_idx)
                            
                            if current_span is not None:
                                retokenizer.merge(current_span)
                                spans_list.append(current_span)
                    '''
                    # current_doc.ents = spans_list
                    try:
                        current_example = Example.from_dict(current_doc, annotation)
                    except Exception as excpt:
                        to_be_removed_from_set = list()
                        exception_string = str(excpt)
                        if "Trying to set conflicting doc.ents" in exception_string:
                            print('\n\n\n', 'Exception RE')
                            pattern = r"\(\d+,\s*\d+,\s*'PRODUCT'\)"
                            found_entities = re.findall(pattern, exception_string)
                            if found_entities:
                                for match in found_entities:
                                    current_start_idx = int(str(match).split(',')[0][1:])
                                    current_end_idx   = int(str(match).split(',')[1][1:])
                                    to_be_removed_from_set.append(text[current_start_idx:current_end_idx])

                        
                        print(to_be_removed_from_set)
                        exit()

                    examples_list.append(current_example)

                nlp_model.update(examples_list, drop=0.35, losses=losses)

                del examples_list

            print(f'[{c_iter}] \t', "Losses", losses)
            loss_function_x.append(c_iter)
            loss_function_y.append(losses['ner'])

    return nlp_model, loss_function_x, loss_function_y



if __name__ == "__main__":
    # Load the small English model
    # nlp = spacy.load("en_core_web_sm")
    
    downloaded_df = pd.read_excel('downloaded_text_from_links.xlsx')
    downloaded_df['product_name'].fillna('', inplace=True)

    # Preprocess Text
    downloaded_df['processed_text'] = downloaded_df.apply(lambda row: process_text(str(row['text'])), axis=1)

    # Preprocess Products
    print('----------------------------------------', '\n')
    print('Preprocessing Products')
    product_list = list() 
    for _, current_row in downloaded_df.iterrows():
        product_list += process_product_names(str(current_row['product_name']), split_individual_tokens=True)
    product_set = set(product_list)

    # Load 100 most common furniture product names 
    with open('top_100_furniture_store_product_names.txt', 'r') as top_furniture_products_file:
        file_contents = top_furniture_products_file.read()
        processed_top_100 = process_product_names(file_contents, split_individual_tokens=True)
        processed_top_100 = set(processed_top_100)
    print('Top 100 Furniture product names')
    # print(processed_top_100)

    product_set = product_set.union(processed_top_100)

    print('Union product names')
    # print(product_set)

    # Remove spans
    product_set = {x for x in product_set if not x.isdigit()} # remove numbers

    # Leave this uncommented for 'Long'
    '''
    
    product_set.remove('wood')
    product_set.remove('chair')
    product_set.remove('bar')
    product_set.remove('table')
    product_set.remove('wall')
    product_set.remove('stand')
    product_set.remove('board')
    product_set.remove('drawer')
    product_set.remove('gift') 
    product_set.remove('desk') 
    product_set.remove('robe') 
    product_set.remove('stool') 
    product_set.remove('bed') 
    '''

    # Leave this uncommented for 'Short'
    product_set.remove('lakewood')      # wood
    product_set.remove('chairs')        # chair
    product_set.remove('barrel')        # bar
    product_set.remove('armchair')      # arm / chair
    product_set.remove('wardrobe')      # war
    product_set.remove('stdesktable')   # table
    product_set.remove('tables')        # table
    product_set.remove('wallpaper')     # wall
    product_set.remove('standing')      # stand    
    product_set.remove('nightstand')    # stand
    product_set.remove('sideboard')     # board
    product_set.remove('barstool')      # bar
    product_set.remove('wooden')        # wood
    product_set.remove('egift')         # gift
    # product_set.remove('standabledesk') # desk
    # product_set.remove('wardrobe')      # desk
    # product_set.remove('barstool')      # stool
    product_set.remove('bedside')       # bed
    product_set.remove('drawers')       # drawer
    product_set.remove('headboard')       # board


    print('----------------------------------------', '\n')
    print('Filtering Data')
    # filtered_dataset, size_of_data_set = create_data_set(downloaded_df)
    filtered_dataset       = deepcopy(downloaded_df.loc[downloaded_df['text'].str.len() > 5])
    size_of_data_set        = len(filtered_dataset)

    filtered_train_dataset = deepcopy(filtered_dataset[5:])
    size_of_train_data_set = len(filtered_train_dataset)

    filtered_test_dataset  = deepcopy(filtered_dataset[:5])
    size_of_test_data_set  = len(filtered_test_dataset)

    print('Size of the filtered data set: ', size_of_data_set)
    print('Size of the filtered train data set: ', size_of_train_data_set)
    print('Size of the filtered test data set: ', size_of_test_data_set)

    # Create Training Data
    print('----------------------------------------', '\n')
    print('Creating Train/Test Data Sets')

    # full_data_set, size_of_full_data_set = create_labeled_data(filtered_dataset, product_set, size_of_data_set)
    # slice_idx = -5
    # train_data_set = full_data_set[:slice_idx]
    # test_data_set  = full_data_set[slice_idx:]
    
    train_data_set, train_size =  create_labeled_data(filtered_train_dataset, product_set, size_of_train_data_set)
    test_data_set,  test_size  =  create_labeled_data(filtered_test_dataset, product_set, size_of_test_data_set)

    # Train Model
    print('----------------------------------------', '\n')
    print('Training')
    
    final_model, loss_function_x, loss_function_y = train_ner(train_data_set, n_iter=500)

    print('----------------------------------------', '\n')
    print('Saving To Disk')
    model_save_name = "ner_model_500it_35_drp_short_pn"
    final_model.to_disk(model_save_name)
    plt.plot(loss_function_x, loss_function_y)
    plt.title(f'Loss Function {model_save_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(f'loss_{model_save_name}.png')

    # Test
    print('----------------------------------------', '\n')
    print('Testing')

    testing_model = spacy.load(model_save_name)
    
    for test_text in test_data_set:
        current_test_text   = test_text[0]
        current_annotations = test_text[1]
        list_of_products = list()

        for entity in current_annotations['entities']:
            list_of_products.append(current_test_text[entity[0]:entity[1]])


        # test_doc = final_model(current_test_text)
        test_doc = testing_model(current_test_text)
        print('----------------------------------------', '\n')
        print('Corrent PRODUCT', list_of_products)
        print()
        print("Entities", [(ent.text, ent.label_) for ent in test_doc.ents])
        print('----------------------------------------', '\n')