import json 
import streamlit as st 
import os 
from openai import OpenAI 
import re 
from groq import Groq


################# ChromaDB Functions ###############################
def index_json_files_using_chromadb(collection, json_directory, run_till=50):
    # Loop through each JSON file in the directory
    count = 0
    progress_text = "Indexing Trials..."
    my_bar = st.progress(0, progress_text)
    list_of_files = os.listdir(json_directory)
    for filename in list_of_files:
        if filename.endswith('.json'):
            file_path = os.path.join(json_directory, filename)
            with open(file_path, 'r') as file:
                trial_data = json.load(file)

                # Create a string representation of the trial data
                trial_text = f"{trial_data['BriefTitle']} {trial_data['EligibilityCriteria']} {trial_data['BriefSummary']} {trial_data['Conditions']} {trial_data['Interventions']} {trial_data['PrimaryOutcomes']} {trial_data['BaselineMeasures']}"

                # Store the embedding and metadata in ChromaDB
                collection.add(
                    documents=[trial_text],
                    metadatas=[{
                        "NCTId": trial_data["NCTId"],
                        "BriefTitle": trial_data["BriefTitle"],
                        "EligibilityCriteria": trial_data["EligibilityCriteria"],
                        "BriefSummary": trial_data["BriefSummary"],
                        "Conditions": trial_data["Conditions"],
                        "Interventions": trial_data["Interventions"],
                        "PrimaryOutcomes": trial_data["PrimaryOutcomes"],
                        "BaselineMeasures": trial_data["BaselineMeasures"]
                    }],
                    ids=[trial_data["NCTId"]]
                )
        count += 1
        
        if run_till!=-1:
            my_bar.progress(count/run_till, progress_text)
            if count == run_till:
                break
        else:
            my_bar.progress(count/len(list_of_files), progress_text)
    my_bar.empty()
    st.success("Indexing complete!")

def query_clinical_trials_using_chromadb(collection, trial_data, query_text, top_k=2):

    # Query the ChromaDB for similar trials
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k+1
    )

    # Print the results
    for i in range(1, len(results['ids'][0])):
        with st.expander(f"Result {i}"):
            tab1, tab2= st.tabs(['Similar Trial', 'Explain Similarity'])

            #Tab1: Info about similar trial
            with tab1:
                st.write("Similar Trial")
                st.write("ID:", results['ids'][0][i])
                st.write("Distance:", results['distances'][0][i])
                md = results['metadatas'][0][i]
                st.write(f"**Brief Title:** {md.get('BriefTitle')}\n\n")
                st.write(f"**Brief Summary:** {md.get('BriefSummary')}\n\n")
                st.write(f"**Conditions:** {md.get('Conditions')}\n\n")
                st.write(f"**Interventions:** {md.get('Interventions')}\n\n")
                st.write(f"**Primary Outcomes:** {md.get('PrimaryOutcomes')}\n\n")
                st.write(f"**Baseline Measures:** {md.get('BaselineMeasures')}\n\n")
                st.write(f"**Eligibility Criteria:** {md.get('EligibilityCriteria')}\n\n")


            #Tab2 Query Formation and Explain Similarity
            #call openai api
            system_message = "You are an AI assistant helping a researcher find similar clinical trials."
            system_message += "You will be given one clinical trial as reference and another clinical trial as query."
            system_message += "You need to explain why and how these two trials are similar and different."
            system_message += "Keep your arguments concise and present in bullet points. Avoid headings and long paragraphs."

            #build the reference trial 
            reference_trial = "\n" 
            reference_trial += f"\n<Brief Title> {trial_data['BriefTitle']}\n"
            reference_trial += f"\n<Brief Summary> {trial_data['BriefSummary']}\n"
            reference_trial += f"\n<Conditions> {trial_data['Conditions']}\n"
            reference_trial += f"\n<Interventions> {trial_data['Interventions']}\n"
            reference_trial += f"\n<Primary Outcomes> {trial_data['PrimaryOutcomes']}\n"
            reference_trial += f"\n<Eligibility Criteria> {trial_data['EligibilityCriteria']}\n"

            #build the query trial
            query_trial = "\n"
            query_trial += f"\n<Brief Title> {md.get('BriefTitle')}\n"
            query_trial += f"\n<Brief Summary> {md.get('BriefSummary')}\n"
            query_trial += f"\n<Conditions> {md.get('Conditions')}\n"
            query_trial += f"\n<Interventions> {md.get('Interventions')}\n"
            query_trial += f"\n<Primary Outcomes> {md.get('PrimaryOutcomes')}\n"
            query_trial += f"\n<Eligibility Criteria> {md.get('EligibilityCriteria')}\n"

            query = f"This is the reference clinical trial:\n {reference_trial} \n"
            query += f"This is the query clinical trial:\n {query_trial} \n"

            response = ask_gpt4_omni(system_message, query.format(reference_trial=reference_trial, query_trial=query_trial))
            
            with tab2:
                #st.write(system_message)
                #st.write(query.format(reference_trial=reference_trial, query_trial=query_trial))
                st.write(response)

########################################################################

############### APP + LLama Index Functions ###############################
def index_json_files_using_llamaindex(chroma_collection, json_directory, vector_store_path, run_till=-1):
    
    #1.1 load json docs
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import JSONNodeParser 
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import TokenTextSplitter
    import chromadb
    from llama_index.core import VectorStoreIndex 
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import StorageContext

    json_docs = SimpleDirectoryReader(json_directory).load_data()

    #1.2 Process and Transform Data - Save as Nodes (Chunks of Documents)
    pipeline = IngestionPipeline(transformations=[JSONNodeParser()])
    nodes = pipeline.run(documents = json_docs)

    #1.3 create and assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection) 
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create a new index
    index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
    print("Created new index.")


def get_query_trial(index, json_directory):
    filenames = [f for f in os.listdir(json_directory) if f.endswith('.json')]
    if 0 <= index < len(filenames):
        file_path = os.path.join(json_directory, filenames[index])
        with open(file_path, 'r') as file:
            return json.load(file)
    return None

def query_clinical_trials_using_llamaindex(chroma_collection, vector_store_path, trial_query, top_k, **kwargs):
    # create a custom query engine and query
    from llama_index.core import get_response_synthesizer 
    from llama_index.core.retrievers import VectorIndexRetriever 
    from llama_index.core.query_engine import RetrieverQueryEngine 
    from llama_index.core.postprocessor import SimilarityPostprocessor 
    from llama_index.core import StorageContext
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import VectorStoreIndex
    import chromadb

    #1.3 create and assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection) 
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if os.path.exists(vector_store_path):
        # Load the existing index
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, 
                                                   embed_model=kwargs.get('embed_model', None))
        #print("Loaded existing index.")

        #configure retriever 
        retriever = VectorIndexRetriever(
            index = index,
            similarity_top_k = top_k + 1,
        )

        #configure response synthesizer
        response_synthesizer = get_response_synthesizer(response_mode="no_text")

        #assemble query engine
        query_engine = RetrieverQueryEngine.from_args(
            retriever = retriever,
            response_synthesizer = response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_threshold = 0.8)],
        )

        #query
        # trial_query = f"""
        # BriefTitle: {trial_data['BriefTitle']}\n
        # EligibilityCriteria: {trial_data['EligibilityCriteria']}\n 
        # BriefSummary: {trial_data['BriefSummary']}\n
        # Conditions: {trial_data['Conditions']}\n
        # Interventions: {trial_data['Interventions']}\n
        # PrimaryOutcomes: {trial_data['PrimaryOutcomes']}\n
        # BaselineMeasures: {trial_data['BaselineMeasures']}
        # """
        # #print(trial_query)
        response = query_engine.query(trial_query) 
        
        #response file names 
        file_names = []
        for key, value in response.metadata.items():
            file_names.append(value['file_name'])

        return file_names[1:]

    else:
        print("Index not found. Please index the JSON files first.")

def display_trial_info_from_id(file_name, query_trial, json_directory):
    import json 
    with open(json_directory + "/" + file_name, "r") as file:
        similar_trial = json.load(file)

    tab1, tab2= st.tabs(['Similar Trial', 'Explain Similarity'])
    with tab1:
            st.write(f"**ID:** {similar_trial['NCTId']}")
            st.write(f"**Brief Title:** {similar_trial['BriefTitle']}")
            st.write(f"**Brief Summary:** {similar_trial['BriefSummary']}")
            st.write(f"**Conditions:** {similar_trial['Conditions']}")
            st.write(f"**Interventions:** {similar_trial['Interventions']}")
            st.write(f"**Primary Outcomes:** {similar_trial['PrimaryOutcomes']}")
            st.write(f"**Baseline Measures:** {similar_trial['BaselineMeasures']}")
            st.write(f"**Eligibility Criteria:** {similar_trial['EligibilityCriteria']}")

    #Tab2 Query Formation and Explain Similarity
    #call openai api

    system_message = "You are an AI assistant helping a researcher find similar clinical trials."
    system_message += "You will be given one clinical trial as reference and another clinical trial as query."
    system_message += "You need to explain why and how these two trials are similar and different."
    system_message += "Keep your arguments concise and present in bullet points. Avoid headings and long paragraphs."

    #build the query trial 
    query_trial_text = "\n" 
    query_trial_text += f"\n<Brief Title> {query_trial['BriefTitle']}\n"
    query_trial_text += f"\n<Brief Summary> {query_trial['BriefSummary']}\n"
    query_trial_text += f"\n<Conditions> {query_trial['Conditions']}\n"
    query_trial_text += f"\n<Interventions> {query_trial['Interventions']}\n"
    query_trial_text += f"\n<Primary Outcomes> {query_trial['PrimaryOutcomes']}\n"
    query_trial_text += f"\n<Eligibility Criteria> {query_trial['EligibilityCriteria']}\n"

    #build the similar trial (querying about it)
    similar_trial_text = "\n"
    similar_trial_text += f"\n<Brief Title> {similar_trial['BriefTitle']}\n"
    similar_trial_text += f"\n<Brief Summary> {similar_trial['BriefSummary']}\n"
    similar_trial_text += f"\n<Conditions> {similar_trial['Conditions']}\n"
    similar_trial_text += f"\n<Interventions> {similar_trial['Interventions']}\n"
    similar_trial_text += f"\n<Primary Outcomes> {similar_trial['PrimaryOutcomes']}\n"
    similar_trial_text += f"\n<Eligibility Criteria> {similar_trial['EligibilityCriteria']}\n"

    query = f"This is the reference clinical trial:\n {query_trial_text} \n"
    query += f"This is the query clinical trial:\n {similar_trial_text} \n"

    response = ask_gpt4_omni(system_message, query.format(reference_trial=query_trial_text, 
                                                        query_trial=similar_trial_text))

    with tab2:
        st.write(response)


################################################################################################

def build_three_shot_prompt(row, similar_trials, json_directory, ref_col_name):
    #prompt structure
    system_message = "You are a helpful assistant with experience in the clinical domain and clinical trial design. \
    You'll be asked queries related to clinical trials. These inquiries will be delineated by a '##Question' heading. \
    Inside these queries, expect to find comprehensive details about the clinical trial structured within specific subsections, \
    indicated by '<>' tags. These subsections include essential information such as the trial's title, brief summary, \
    condition under study, inclusion and exclusion criteria, intervention, and outcomes."

    #baseline measure defintion
    # system_message += "In answer to this question, return a list of probable baseline features (separated by commas) of the clinical trial. \
    # Baseline features are the set of baseline or demographic characteristics that are assessed at baseline and used in the analysis of the \
    # primary outcome measure(s) to characterize the study population and assess validity. Clinical trial-related publications typically \
    # include a table of baseline features assessed  by arm or comparison group and for the entire population of participants in the clinical trial."
    system_message += "In answer to this question, return a list of probable baseline features (each feature should be enclosed within a pair of backticks \
    and each feature should be separated by commas from other features) of the clinical trial. \
    Baseline features are the set of baseline or demographic characteristics that are assessed at baseline and used in the analysis of the \
    primary outcome measure(s) to characterize the study population and assess validity. Clinical trial-related publications typically \
    include a table of baseline features assessed  by arm or comparison group and for the entire population of participants in the clinical trial."


    #additional instructions
    system_message += "You will be given three examples. In each example, the question is delineated by '##Question' heading and the corresponding answer is delineated by '##Answer' heading. \
    Follow a similar pattern when you generate answers. Do not give any additional explanations or use any tags or headings, only return the list of baseline features."

    example_question = build_example_questions(similar_trials, json_directory, ref_col_name)

    #divide row information to generate the query
    title = row['BriefTitle']
    brief_summary = row['BriefSummary']
    condition = row['Conditions']
    eligibility_criteria = row['EligibilityCriteria']
    intervention = row['Interventions']
    outcome = row['PrimaryOutcomes']

    question = "##Question:\n"
    question += f"<Title> \n {title}\n"
    question += f"<Brief Summary> \n {brief_summary}\n"
    question += f"<Condition> \n {condition}\n"
    question += f"<Eligibility Criteria> \n {eligibility_criteria}\n"
    question += f"<Intervention> \n {intervention}\n"
    question += f"<Outcome> \n {outcome}\n"
    question += "##Answer:\n"

    return system_message, example_question + question

def build_example_questions(similar_trials, json_directory, ref_col_name):
    question = ""
    for trial in similar_trials:
        with open(json_directory + '/' + trial) as f:
            example = json.load(f)

        question += "##Question:\n"
        question += f"<Title> \n {example['BriefTitle']}\n"
        question += f"<Brief Summary> \n {example['BriefSummary']}\n"
        question += f"<Condition> \n {example['Conditions']}\n"
        question += f"<Eligibility Criteria> \n {example['EligibilityCriteria']}\n"
        question += f"<Intervention> \n {example['Interventions']}\n"
        question += f"<Outcome> \n {example['PrimaryOutcomes']}\n"
        question += "##Answer:\n"
        question += f"{example[ref_col_name]}\n\n"

    return question

def run_generation_single_openai(message, model_name, openai_token, temperature=0.0):

    client = OpenAI(
        api_key=openai_token
    )

    response = client.chat.completions.create(
      model=model_name,
      messages=message,
      seed = 42,
      temperature=temperature,
      stream=False,
      max_tokens=1000
    )
    return response.choices[0].message.content

def ask_gpt4_omni(system_message, question):
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": question},
      ],
      seed = 42,
      temperature=0.0,
      stream=False,
      max_tokens=500
    )
    return response.choices[0].message.content

def run_generation_single_hf_models(message, hf_url, huggingface_token, temperature=0.0):

    base_url = "https://api-inference.huggingface.co/models/" 

    client = OpenAI(
        base_url=base_url + hf_url + "/v1/",
        api_key =huggingface_token 
    )

    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=message,
        seed = 42,
        temperature=temperature,
        stream=False,
        max_tokens=1000
    )
    return chat_completion.choices[0].message.content

def run_generation_single_hf_models_groq(message, hf_url, groq_token, temperature=0.0):

    #base_url = "https://api-inference.huggingface.co/models/" 

    client = Groq(
        api_key=groq_token
    )

    #print(groq_token)

    chat_completion = client.chat.completions.create(
        messages=message,
        model="llama3-70b-8192",
        seed=42,
        temperature=temperature,
        stream=False,
        max_tokens=1000
    )

    # chat_completion = client.chat.completions.create(
    #     model="tgi",
    #     messages=message,
    #     seed = 42,
    #     temperature=temperature,
    #     stream=False,
    #     max_tokens=1000
    # )
    return chat_completion.choices[0].message.content

def system_user_template(system_message, question):
    return [{"role": "system", "content": system_message},
            {"role": "user", "content": question}]

def get_question_from_row(row):
    title = row['BriefTitle']
    brief_summary = row['BriefSummary']
    condition = row['Conditions']
    eligibility_criteria = row['EligibilityCriteria']
    intervention = row['Interventions']
    outcome = row['PrimaryOutcomes']

    question = ""
    question += f"<Title> \n {title}\n"
    question += f"<Brief Summary> \n {brief_summary}\n"
    question += f"<Condition> \n {condition}\n"
    question += f"<Eligibility Criteria> \n {eligibility_criteria}\n"
    question += f"<Intervention> \n {intervention}\n"
    question += f"<Outcome> \n {outcome}\n"

    return question

def build_gpt4_eval_prompt(reference, candidate, qstart):
    system = """
        You are an expert assistant in the medical domain and clinical trial design. You are provided with details of a clinical trial.
        Your task is to determine which candidate baseline features match any feature in a reference baseline feature list for that trial. 
        You need to consider the context and semantics while matching the features.

        For each candidate feature:

            1. Identify a matching reference feature based on similarity in context and semantics.
            2. Remember the matched pair.
            3. A reference feature can only be matched to one candidate feature and cannot be further considered for any consecutive matches.
            4. If there are multiple possible matches (i.e. one reference feature can be matched to multiple candidate features or vice versa), choose the most contextually similar one.
            5. Also keep track of which reference and candidate features remain unmatched.

        Once the matching is complete, provide the results in a JSON format as follows:
        {
        "matched_features": [
            ["<reference feature 1>", "<candidate feature 1>"],
            ["<reference feature 2>", "<candidate feature 2>"]
        ],
        "remaining_reference_features": [
            "<unmatched reference feature 1>",
            "<unmatched reference feature 2>"
        ],
        "remaining_candidate_features": [
            "<unmatched candidate feature 1>",
            "<unmatched candidate feature 2>"
        ]
        }
    """

    question = f"\n Here is the trial information: \n\n"
    question += f"{qstart}"

    question += f"\n\nHere is the list of reference features: \n\n"
    ir = 1
    for ref_item in reference:
        question += (
            f"{ir}. {ref_item}\n"
        )
        ir += 1


    question += f"\nCandidate features: \n\n"
    ic = 1
    for can_item in candidate:
        question += (
            f"{ic}. {can_item}\n"
        )
        ic += 1

    return system, question

def extract_elements_v2(s):
  """
  Extracts elements enclosed within backticks (`) from a string.

  Args:
    s: The input string.

  Returns:
    A list of elements enclosed within backticks.
  """
  pattern = r"`(.*?)`"
  elements = re.findall(pattern, s)
  return elements

def run_evaluation_with_gpt4o(system_message, question, openai_token):

    client = OpenAI(
        api_key=openai_token
    )

    response = client.chat.completions.create(
      model="gpt-4o",
      response_format = { "type": "json_object" },
      messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": question},
      ],
      seed = 42,
      temperature=0.0,
      stream=False,
      max_tokens=1000
    )
    return response.choices[0].message.content

def match_to_score(matched_pairs, remaining_reference_features, remaining_candidate_features):
    """
    Calculates precision, recall, and F1 score based on the given matched pairs and remaining features.

    Parameters:
    matched_pairs (list): A list of matched feature pairs.
    remaining_reference_features (list): A list of remaining reference features.
    remaining_candidate_features (list): A list of remaining candidate features.

    Returns:
    dict: A dictionary containing the precision, recall, and F1 score.
    """
    precision = len(matched_pairs) / (len(matched_pairs) + len(remaining_candidate_features)) # TP/(TP+FP)
    recall = len(matched_pairs) /  (len(matched_pairs) + len(remaining_reference_features)) #TP/(TP+FN)
    
    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall) # F1

    return {"precision": precision, "recall": recall, "f1": f1}
