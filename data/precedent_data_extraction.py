import re
import lzma
import json
import bisect

import pandas as pd

from datetime    import date
from joblib      import Parallel, delayed
from rapidfuzz   import fuzz
from rapidfuzz   import process
from tqdm        import tqdm
from collections import defaultdict

from transformers import RobertaTokenizerFast, RobertaForTokenClassification
import torch

tqdm.pandas()

############################################################################################################
# ----- Step 0: Get case data -----
############################################################################################################

def make_case_data(n = None):
    '''
    To enable later parralalization, we create a json of all CAP opinion texts.

    From each case we retain:
    1. case id
    2. case text
    3. case court
    4. case decision date


    Inputs:
    n: If supplied, limit to first n CAP cases. If none, include all cases.
    '''

    file  = "./data/data.jsonl.xz"
    c     = 0
    data = []
    with lzma.open( file ) as in_file:
        for line in in_file:

            # break after n
            c += 1
            if (c % 1000 == 0): print(c,' of 1761068 cases', end='\r')
            if n:
                if c > n: break

            # convert to case dictionary
            case = json.loads(str(line,'utf8'))

            # try to get info, skip case if missing
            try:
                case_id       = case['id']
                case_text     = case['casebody']['data']['opinions'][0]['text']
                case_cites    = case['cites_to']
                case_court    = case['court']['name']
                case_date     = case['decision_date']
                case_citation = case['citations']
                case_name     = case['name_abbreviation']
            except: continue

            # if no citations, continue
            if len(case_cites) == 0: continue

            case_citation = [x for x in case_citation if x['type'] == 'official'][0]['cite']
            case_year     = case_date.split('-')[0]
            case_citation = f"{case_name}, {case_citation} ({case_year})"

            case_dict = {'id'      : case_id,
                         'name'    : case_name,
                         'text'    : case_text,
                         'cites_to': case_cites,
                         'court'   : case_court,
                         'date'    : case_date,
                         'cite'    : case_citation
                        }

            data.append(case_dict)

    # save as json
    meta  = 'Created on ' + str(date.today()) + '. Contains all CAP case texts and ids for extracting passages'
    cases = {'meta': meta,
             'data': data}

    with open('./data/case_data.json', 'w') as fp:
        json.dump(cases, fp)

def get_citation_dict(n = None):
    '''
    Returns mapping of case citations to CAP ids. Stores mapping as json.
    This function only needs to be used once when we get new data since the mapping is stored.
    '''

    # Create ID-Citation Dictionary
    citation_dict = {}


    file  = "./data/data.jsonl.xz"
    c     = 0
    with lzma.open( file ) as in_file:
        for line in in_file:
            c += 1
            if (c % 1000 == 0): print(c,' of 1761068 cases', end='\r')
            if n:
                if c > n: break

            # convert to case dictionary
            case = json.loads(str(line,'utf8'))

            # skip if degenerate
            try: case_id = case['id']
            except: continue   # ... continue with next case

            # loop over citations
            for citation in case['citations']:

                # append to citation dictionary
                citation_dict.update({citation['cite'] : case_id})
            
            # add case name abbreviation to citation dictionary
            citation_dict.update({case['name_abbreviation'] : case_id})

    # save as json
    with open('./data/citation_dict.json', 'w') as fp:
        json.dump(citation_dict, fp)

############################################################################################################
# ----- Step 1: Tokenize Cases -----
############################################################################################################

def legal_sentence_tokenize(text, tokenizer, model, device, batch_size = 6):
    '''
    Tokenizes text into sentences using custom ROBERTA tokenizer.
    '''

    # tokenize
    token_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False, pad_to_multiple_of=254, padding=True)
    # reshape into batches
    token_ids = token_ids.reshape(-1, 254)

    cls_token = torch.tensor([tokenizer.cls_token_id])
    sep_token = torch.tensor([tokenizer.sep_token_id])

    out_ids, out_labels = [], []
    for batch in token_ids:
        input_ids = torch.cat((cls_token, batch, sep_token))
        input_ids = input_ids.to(device)
        logits = model(input_ids.unsqueeze(dim=0)).logits #.detach().cpu()
        labels = logits[0][1:-1].argmax(dim=1).detach().cpu().tolist()
        out_labels.extend(labels)
        out_ids.extend(input_ids[1:-1].detach().cpu().tolist())

    result = []
    sent   = []
    for w, pred in zip(out_ids, out_labels):
        # padding token
        if w == 1:
            continue
        sent.append(w)
        if pred == 1:
            result.append(tokenizer.decode(sent))
            sent = []
    if len(sent) > 0:
        result.append(tokenizer.decode(sent))
    return result

def tokenize_cases_gpu(chunk_id, num_gpu = 6):
    """
    Calls sentence tokenizer on specific chunk_id. This is used to parralelize the tokenization process across gpus.

    Inputs:
    chunk_id: id of chunk to run on. Total number of chunks equal to num_gpu
    num_gpu: total number of gpus to use
    """

    device = 'cuda:' + str(chunk_id)

    # load tokenizer model
    model_name_or_path = "roberta_sentence_tokenizer"
    tokenizer          = RobertaTokenizerFast.from_pretrained(model_name_or_path)
    model              = RobertaForTokenClassification.from_pretrained(model_name_or_path)
    model              = model.to(device)

    # load CAP case texts
    with open('./data/case_data.json') as f: case_texts = json.load(f)
    case_texts = case_texts['data']

    # Split the case_texts into chunks
    chunks = [case_texts[i::num_gpu] for i in range(num_gpu)]

    case_texts = chunks[chunk_id]

    case_texts_tokenized = [legal_sentence_tokenize(x['text'], tokenizer, model, device) for x in tqdm(case_texts, smoothing = 0)]

    # replace case texts with tokenized versions
    for i, case in enumerate(case_texts):
        case['text'] = case_texts_tokenized[i]
    
    with open(f'./data/chunks/case_data_tokenized_{chunk_id}.json', 'w') as fp:
        json.dump(case_texts, fp)
    
def combine_chunks():
    '''
    Combine chunks of case_data_tokenized_i.json into case_data_tokenized.json
    '''
    # load case_data_tokenized_i.json for i in range(0, 7)
    case_data = []
    for i in range(0, 8):
        with open(f'./data/chunks/case_data_tokenized_{i}.json', 'r') as f:
            case_data += json.load(f)

    # save case_data_tokenized.json
    meta = 'Created on ' + str(date.today()) + '. Tokenized case data from CAP.'

    json_output = {'meta': meta,
                   'data': case_data
                  }

    with open('./data/case_data_tokenized.json', 'w') as f:
        json.dump(json_output, f)

############################################################################################################
# ----- Step 2: Extract quoted precedent -----
############################################################################################################

def extract_case_precedent(case, citation_dict, regex_quote_pattern):
    '''
    
    Sub-routine: finds all quoted portions of text in each sentence of the case and converts citations to other opinions to CAP id.

    Args:
    case: dictionary representing a single case, with 'text' field being a list of sentences
    citation_dict: dictionary mapping citations to CAP ids
    regex_quote_pattern: compiled regex pattern to find quotes

    Returns: list of quotes and list of CAP citations
    '''

    # Get list of citations from dict
    citations = [citation['cite'] for citation in case['cites_to']]

    # Convert text citations to CAP ids
    citations = [citation_dict[citation] for citation in citations if citation in citation_dict]

    # Occasionally the case will cite itself. We remove these citations.
    citations = [citation for citation in citations if citation != case['id']]

    # Sometimes there are duplicated citations, we remove these
    citations = list(set(citations))

    # Initialize variables for quoted text
    output                    = []
    min_length                = 5
    last_matched_quote        = None

    # Capture all matches to regex in the case_text
    matches = regex_quote_pattern.finditer(case['text'])

    for match in matches:
        
        matched_text = match.group(1)
        
        if len(matched_text.split()) > min_length:             
            span = match.span(1)
            output.append({'text': matched_text, 'span': span, 'last_match': last_matched_quote})
            last_matched_quote = {'span': span}

    output = {'dest_id'    : case['id'], 
              'citations'  : citations, 
              'quoted_text': output, 
              'dest_date'  : case['date'], 
              'dest_court' : case['court'],
              'dest_name'  : case['name'],
              'dest_cite'  : case['cite']
              }

    return output

def precedent_extraction_wrapper(filename_used = './data/citations_used.json', num_cores = 30, batch_size = 1000):
    '''
    Wrapper function that loads cases and extracts the quoted precedent with optional parralelization
    '''

    # load citation dict
    with open('./data/citation_dict.json') as f: citation_dict = json.load(f)

    # load CAP case texts
    with open('./data/case_data.json') as f: case_texts = json.load(f)
    case_texts = case_texts['data']

    # pre-compile regular expression for faster execution
    # this matches any text in quotes
    regex_quote_pattern = re.compile(r'(?:“|")(.*?)(?:”|")')

    # extract quoted precedent
    citation_used = Parallel(n_jobs=num_cores, batch_size = batch_size)(delayed(extract_case_precedent)(
        case, citation_dict, regex_quote_pattern) for case in tqdm(case_texts))

    # keep only cases where we find quotes and CAP cases are cited
    citation_used = [x for x in citation_used if (len(x['citations'])>0 and len(x['quoted_text'])>0)]

    print('Number of cases with quoted precedent: ', len(citation_used))
    print('Number of quotations found: ', sum([len(x['quoted_text']) for x in citation_used]))

    # save as json
    meta  = 'Created on ' + str(date.today()) + '. All quoted text and citations to CAP opinions'
    json_output = {'meta': meta,
                   'data': citation_used}

    with open(filename_used, 'w') as fp:
        json.dump(json_output, fp)

############################################################################################################
# ----- Step 3: Citation Matcher ----- #
############################################################################################################

def citation_matcher(case, quoted_precedent):
    '''
    Sub-Routine: Given a list of quoted precedent from precedent_extractor,
    match these to sentences in the corresponding origin case

    Inputs:
    case: case dict from CAP with 'text' being a list of pre-tokenized sentences
    quoted_precedent: list of case citations generated from precedent_extractor
    '''

    case_id   = case['id']
    case_text = case['text'] 

    # Keep only sentences longer than 10 words
    choices = [x for x in case_text if len(x.split()) > 10]

    extracted    = []
    score_cutoff = 90

    # Iterate over all quotations from this case
    # Find sentence that is close to query subject to score_cutoff
    for info in quoted_precedent:
        dest_id = info['dest_id']
        
        for query in info['quoted_text']:
            match = process.extract(query['text'],
                                    choices,
                                    scorer=fuzz.token_set_ratio,
                                    score_cutoff=score_cutoff)
            
            # If match is found, find match with highest score and append to extracted
            if match: 
                # Get best match
                match = [x for x in match if x[1] == max([x[1] for x in match])]
                match = match[0][0]  # Extract sentence from match

                extracted.append({'dest_id'     : dest_id, 
                                  'source_id'   : case_id, 
                                  'passage'     : match, 
                                  'query'       : query,
                                  'dest_date'   : info['dest_date'],
                                  'dest_court'  : info['dest_court'],
                                  'dest_name'   : info['dest_name'],
                                  'dest_cite'   : info['dest_cite'],
                                  'source_date' : case['date'], 
                                  'source_court': case['court'],
                                  'source_name' : case['name'],
                                  'source_cite' : case['cite']
                                  })
    
    if len(extracted) == 0: 
        return None
    
    return extracted

def origin_wrapper(filename_origin = './data/citations_origin.json', num_cores = 30,  batch_size = 1000):
    '''
    Step 2
    Wrapper function to find quotations in the case of origin.
    Groups unique passages by source_id and assigns a passage_id to each unique passage
    '''

    # load CAP case texts
    with open('./data/case_data_tokenized.json') as f: case_texts = json.load(f)
    case_texts = case_texts['data']

    print('Number of cases: ', len(case_texts))

    # load previously identified quotations
    with open('./data/citations_used.json') as f: citations_used = json.load(f)
    citations_used = citations_used['data']

    print('Number of cases with quoted precedent: ', len(citations_used))

    # for convinence generate dictionary of quotations
    quotation_dict = {}
    for case in citations_used:

        info     = case.copy()
        case_ids = case['citations']
        case_ids = set(case_ids) # remove duplicates from citations
        info.pop('citations')
        
        # make dict of citation dictionaries
        for case_id in case_ids:
            if case_id not in quotation_dict: quotation_dict.update({case_id : [info]})
            else: quotation_dict[case_id].append(info)

    # keep only case texts with cited passages
    case_texts = [x for x in case_texts if x['id'] in quotation_dict.keys()]
        
    print('Number of cases with cited passages: ', len(case_texts))
    print('Number of potential training examples: ', sum([len(x['quoted_text']) for item in quotation_dict.values() for x in item]) / 1000000, ' million')

    # extract quotations from source case texts
    citations_origin = Parallel(n_jobs=num_cores, batch_size=batch_size)(delayed(citation_matcher)
                                    (case = case, quoted_precedent = quotation_dict[case['id']]) for case in tqdm(case_texts))

    # remove None values and flatten
    citations_origin = [x for x in citations_origin if x != None]
    citations_origin = [item for sublist in citations_origin for item in sublist]

    print('Number of training examples: ', len(citations_origin))

    # save as json
    meta        = 'Created on ' + str(date.today()) + '.Quotations matched to source opinions'
    json_output = {'meta' : meta, 'data': citations_origin}
    with open(filename_origin, 'w') as fp: json.dump(json_output, fp)

############################################################################################################
# ----- Step 4: Get Desination Context  ----- #
############################################################################################################

def assign_passage_id(passages_origin_filename = './data/citations_origin.json', passage_dict_filename = './data/passage_dict.json'):
    '''
    Assigns a unique passage_id to each unique passage and saves the mapping of passage_id to passage.
    '''

    # load previously identified quotations
    with open(passages_origin_filename) as f: citations_origin = json.load(f)
    citations_origin = citations_origin['data']

    # create dictionary of unique passages
    unique_passages = [{'source_id': x['source_id'], 'passage': x['passage']} for x in citations_origin]
    unique_passages = list(set(tuple(passage.items()) for passage in unique_passages))
    unique_passages = [dict(passage) for passage in unique_passages]

    print('Number of unique passages: ',len(unique_passages))
    
    grouped_passages = defaultdict(list)

    for passage in unique_passages:
        source_id  = passage['source_id']
        passage_id = f"{source_id}_{len(grouped_passages[source_id])}"
        grouped_passages[source_id].append({'passage_id': passage_id, 'passage': passage['passage']})

    # create list of unique passages with passage_id
    grouped_passages_list = [{'source_id': k, 'passages': v} for k, v in grouped_passages.items()]

    # for convenience, make a dict of grouped_passages_list
    grouped_passages_dict = {item['source_id']: item['passages'] for item in grouped_passages_list}

    # replace passage strings in citations_origin with passage_id
    updated_citations_origin = []

    for citation in citations_origin:
        source_id        = citation['source_id']
        passage          = citation['passage']
        updated_citation = citation.copy()                # make a copy of the entry
        
        # find the correct passage_id
        passage_id = None
        for p in grouped_passages_dict[source_id]:
            if p['passage'] == passage:
                passage_id = p['passage_id']
                break
        
        # if passage_id is found, replace passage with passage_id
        if passage_id is not None:
            updated_citation.pop('passage')                   # remove the passage
            updated_citation['passage_id'] = passage_id       # add the passage_id
            updated_citations_origin.append(updated_citation) # append the updated entry
        else:
            updated_citations_origin.append(citation)

    # save as updated citations origin
    meta = 'Created on ' + str(date.today()) + '. Training data with passage_id instead of passage but without destination_context.'
    passages_origin_filename = './data/citations_origin_updated.json'
    json_output = {'meta' : meta, 'data': updated_citations_origin}
    with open(passages_origin_filename, 'w') as fp: json.dump(json_output, fp)

    grouped_passages_dict = {passage['passage_id']: passage['passage'] for value in grouped_passages_dict.values() for passage in value}

    # save mapping of passages to passage ids
    meta = 'Created on ' + str(date.today()) + '. Dictionary of unique passages with passage ids and sources.'
    json_output = {'meta' : meta, 'data': grouped_passages_dict}
    with open(passage_dict_filename, 'w') as fp: json.dump(json_output, fp)

def get_destination_context(case, training_example, max_words = 300, min_chars = 5):
    '''
    Sub-Routine: Returns sentences before a quotation up to a maximum length of max_words
    '''

    # get case text
    case_text = case['text']

    # get cumulative sum of sentence lengths (will be used to find sentence index)
    cumulative_sum     = 0
    cumulative_lengths = [cumulative_sum := cumulative_sum + len(sentence) for sentence in case_text]
    
    # find location quote in case_text
    quote       = training_example['query']['text']
    quote_start = training_example['query']['span'][0]
    quote_idx   = bisect.bisect_right(cumulative_lengths, quote_start)

    # special case: quote is in the last sentence
    if quote_idx == len(case_text): 
        training_example.update({'destination_context': ''})
        return training_example

    # find preceeding text in sentence containing quote
    text_ahead_of_quote = case_text[quote_idx].split(quote,1)[0]

    # add quote to training_example
    training_example.update({'quote': quote})

    # find location of previous quote in case_text
    # Select sentences before the quote's sentence and after the last quote's sentence
    if training_example['query']['last_match'] is not None:
        last_quote_end        = training_example['query']['last_match']['span'][1]
        last_quote_idx        = bisect.bisect_right(cumulative_lengths, last_quote_end)
        preceding_sentences   = case_text[last_quote_idx:quote_idx-1]

    else:
        preceding_sentences   = case_text[:quote_idx-1]

    # Check if the preceding context is too short
    preceeding_context = ''.join(preceding_sentences) + text_ahead_of_quote
    if  len(preceeding_context) < min_chars: 
        training_example.update({'destination_context': ''})
        return training_example

    # Initialize context sentences list and word count
    context_sentences = []
    text_length       = len(text_ahead_of_quote.split())

    # Add sentences to training example up to max_words
    for sentence in reversed(preceding_sentences):
        context_sentences.append(sentence)
        text_length += len(sentence.split())
        if len(sentence.split()) + text_length >= max_words:
            break

    # flip add the context from the sentence containing the quote
    context_sentences.reverse()
    context_sentences.append(text_ahead_of_quote)

    # join context sentences
    context_sentences = ''.join(context_sentences)

    # add context to training_ex
    training_example.update({'destination_context': context_sentences})

    # remove query
    training_example.pop('query', None)

    return training_example

def context_wrapper(training_filename = './data/training_data.csv.gz', num_cores = 30, batch_size = 1000):
    '''
    Parralelize the get_destination_context function
    '''

    #load previously identified quotations if not provided
    with open('./data/citations_origin_updated.json') as f: citations_origin = json.load(f)
    citations_origin = citations_origin['data']

    # load CAP case texts
    with open('./data/case_data_tokenized.json') as f: case_texts = json.load(f)
    case_texts = case_texts['data']

    # retain only case texts that are cited
    dest_ids   = set([x['dest_id'] for x in citations_origin])
    case_texts = [x for x in case_texts if x['id'] in dest_ids]
    
    # create dictionary of case texts for faster lookup
    case_texts = {x['id']:x for x in case_texts}

    # get context for each citation
    citation_context  = Parallel(n_jobs=num_cores, batch_size=batch_size)(delayed(get_destination_context)
                                    (case = case_texts[ex['dest_id']], training_example = ex) for ex in tqdm(citations_origin))
    
    # remove training examples where context is too short
    citation_context = [x for x in citation_context if len(x['destination_context']) > 0]
    
    print('Number of training examples: ', len(citation_context))

    # remove query from all examples
    for item in citation_context: item.pop('query', None)

    # save as gzipped csv
    training_df = pd.DataFrame(citation_context)
    training_df.to_csv(training_filename, index = False, compression = 'gzip')

def make_smaller_training_data():
    '''
    Loads training data, removes source citations and saves data for top-25k, top-100K and top-250k passages
    '''

    # load training data
    training_data = pd.read_csv('./data/training_data.csv.gz')

    print('Loaded Training Data' )

    for count in [10000, 20000, 50000]:
        # Find the n most common passage_id values
        top_passage_ids = training_data['passage_id'].value_counts().nlargest(count).index

        # Filter the DataFrame to get rows with the most common passage_id values
        output_df = training_data[training_data['passage_id'].isin(top_passage_ids)]

        print(output_df.shape)

        # save as gzipped csv
        output_df.to_csv(f'./data/top_{count}_training_data.csv.gz', index = False, compression = 'gzip')
