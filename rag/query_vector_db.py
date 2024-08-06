import sys
from pathlib import Path
from enum import Enum
root_path = Path("..")
root_path = root_path.absolute().parent
sys.path.append(str(root_path/"rag"))
from rag_utils import constraint_path, problem_descriptions_vector_db_path, constraint_vector_db_path, objective_descriptions_vector_db_path
import pandas as pd
from typing import List, Tuple, Dict, Union
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

openai_key = "###"
openai_org = "###"

class RAGFormat(Enum):
    PROBLEM_DESCRIPTION_OBJECTIVE = 1
    PROBLEM_DESCRIPTION_CONSTRAINTS = 2
    CONSTRAINT_FORMULATION = 3
    OBJECTIVE_FORMULATION = 4

constraint_df = pd.read_pickle(constraint_path)

def load_vector_db(vector_db_path: Path, model_name: str = "text-embedding-3-large") -> Chroma:
    """
    Loads the vector database from the specified directory.

    Args:
        vector_db_path (Path): The path to the vector database directory.
        model_name (str): The model name for generating embeddings.

    Returns:
        Chroma: The loaded vector database.
    """
    embedding_function = OpenAIEmbeddings(model=model_name, openai_api_key=openai_key, organization=openai_org)
    return Chroma(persist_directory=str(vector_db_path), embedding_function=embedding_function)

problem_desciption_vector_db = load_vector_db(problem_descriptions_vector_db_path)
constraint_vector_db = load_vector_db(constraint_vector_db_path)
objective_descriptions_vector_db = load_vector_db(objective_descriptions_vector_db_path)

def get_rag_from_problem_description(description: str, format_type: RAGFormat, top_k: int = 3) -> str:
    """
    Generates RAG (Retrieval-Augmented Generation) text based on a problem description.

    Args:
        description (str): The problem description.
        format_type (RAGFormat): The format type for RAG text.
        top_k (int, optional): The number of top similar documents to consider. Defaults to 3.

    Returns:
        str: The generated RAG text.
    """
    similar_documents = problem_desciption_vector_db.similarity_search_with_score(description, k=top_k+1)
    rag_text = ""
    similar_documents_remove_duplicates = [document for document in similar_documents if document[0].page_content != description][:top_k]
    for i in range(top_k):
        document = similar_documents_remove_duplicates[i][0]
        document.metadata['key']
        if format_type == RAGFormat.PROBLEM_DESCRIPTION_OBJECTIVE:
            rag_text += f"Problem Description:\n{document.page_content}\n\nObjective:\n{constraint_df[constraint_df.problem_name == document.metadata['key']].iloc[0].objective_description}\n\n"
        elif format_type == RAGFormat.PROBLEM_DESCRIPTION_CONSTRAINTS:
            rag_text += f"Problem Description:\n{document.page_content}\n\nConstraints:\n"
            for row in constraint_df[constraint_df.problem_name == document.metadata['key']].itertuples():
                if row.constraint_description == "auxiliary constraint":
                    continue
                rag_text += f"{row.constraint_description}\n\n"
        elif format_type == RAGFormat.CONSTRAINT_FORMULATION:
            for row in constraint_df[constraint_df.problem_name == document.metadata['key']].itertuples():
                rag_text += f"{row.constraint_description}\n{row.constraint_formulation}\n\n"
        elif format_type == RAGFormat.OBJECTIVE_FORMULATION:
            rag_text += f"Objective:\n{constraint_df[constraint_df.problem_name == document.metadata['key']].iloc[0].objective_description}\n{constraint_df[constraint_df.problem_name == document.metadata['key']].iloc[0].objective_formulation}\n\n"
    return rag_text

def get_rag_from_constraint(constraint_description: str, format_type: RAGFormat, current_problem_name: str | None = None,  top_k: int = 10) -> str:
    """
    Generates RAG text based on a constraint description.

    Args:
        constraint_description (str): The constraint description.
        format_type (RAGFormat): The format type for RAG text.
        current_problem_name (str | None, optional): The name of the current problem. Defaults to None.
        top_k (int, optional): The number of top similar documents to consider. Defaults to 10.

    Returns:
        str: The generated RAG text.
    """
    assert format_type in [RAGFormat.CONSTRAINT_FORMULATION]

    similar_documents = constraint_vector_db.similarity_search_with_score(constraint_description, k=top_k+20)
    rag_text = ""
    similar_documents_remove_duplicates = [document for document in similar_documents if document[0].page_content != constraint_description]
    similar_documents_remove_duplicates = [x for x in similar_documents_remove_duplicates if constraint_df.iloc[x[0].metadata['key']].problem_name != current_problem_name][:top_k]

    for i in range(min(top_k, len(similar_documents_remove_duplicates))):
        x = similar_documents_remove_duplicates[i][0]
        row = constraint_df.iloc[x.metadata['key']]
        rag_text += f"{row.constraint_description}\n{row.constraint_formulation}\n\n"
    return rag_text

def get_rag_from_objective(objective_description: str, format_type: RAGFormat, current_problem_name: str | None = None, top_k: int = 10) -> str:
    """
    Generates RAG text based on an objective description.

    Args:
        objective_description (str): The objective description.
        format_type (RAGFormat): The format type for RAG text.
        current_problem_name (str | None, optional): The name of the current problem. Defaults to None.
        top_k (int, optional): The number of top similar documents to consider. Defaults to 10.

    Returns:
        str: The generated RAG text.
    """
    assert format_type in [RAGFormat.OBJECTIVE_FORMULATION]

    similar_documents = objective_descriptions_vector_db.similarity_search_with_score(objective_description, k=top_k+20)
    rag_text = ""
    similar_documents_remove_duplicates = [document for document in similar_documents if document[0].page_content != objective_description]
    similar_documents_remove_duplicates = [x for x in similar_documents_remove_duplicates if constraint_df.iloc[int(x[0].metadata['key'])].problem_name != current_problem_name][:top_k]

    for i in range(min(top_k, len(similar_documents_remove_duplicates))):
        x = similar_documents_remove_duplicates[i][0]
        row = constraint_df.iloc[int(x.metadata['key'])]
        rag_text += f"Objective:\n{row.objective_description}\n{row.objective_formulation}\n\n"
    return rag_text

def jaccard_similarity(set1: Union[set, List[Tuple[str, str]], str], set2: Union[set, List[Tuple[str, str]], str]) -> float:
    """
    Calculates the Jaccard similarity between two sets, lists of pairs, or descriptions.

    Args:
        set1 (Union[set, List[Tuple[str, str]], str]): The first set, list of pairs, or description.
        set2 (Union[set, List[Tuple[str, str]], str]): The second set, list of pairs, or description.

    Returns:
        float: The Jaccard similarity coefficient.
    """
    if isinstance(set1, str) and isinstance(set2, str):
        set1 = set(set1.split())
        set2 = set(set2.split())
    elif isinstance(set1, list) and isinstance(set2, list):
        set1 = set(set1)
        set2 = set(set2)
    assert isinstance(set1, set) and isinstance(set2, set)

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def get_rag_from_problem_categories(description: str, labels: Dict[str, Dict], format_type: RAGFormat, current_problem_name: str | None = None, top_k: int = 3) -> str:
    """
    Generates RAG text based on problem categories and similarity criteria.

    Args:
        description (str): The problem description.
        labels (Dict[str, Dict]): The labels for problem categories.
        format_type (RAGFormat): The format type for RAG text.
        current_problem_name (str | None, optional): The name of the current problem. Defaults to None.
        top_k (int, optional): The number of top similar problems to consider. Defaults to 3.

    Returns:
        str: The generated RAG text.
    """
    # Generate the pairs for the specified problem
    pairs_x = [(labels['types'][i], labels['domains'][j]) for i in range(len(labels['types'])) for j in range(len(labels['domains']))]
    types_x = set(labels['types'])
    domains_x = set(labels['domains'])

    # Store similarities
    similarities = []

    # Iterate over other problems
    for other_problem_name in set(constraint_df.problem_name.unique()) - {current_problem_name}:
        y_row = constraint_df[constraint_df.problem_name == other_problem_name].iloc[0]
        y_labels = y_row['labels']
        description_y = y_row['description']

        # Generate the pairs for the other problem
        pairs_y = [(y_labels['types'][i], y_labels['domains'][j]) for i in range(len(y_labels['types'])) for j in range(len(y_labels['domains']))]
        types_y = set(y_labels['types'])
        domains_y = set(y_labels['domains'])

        # Calculate Jaccard similarities
        pairs_similarity = jaccard_similarity(pairs_x, pairs_y)
        types_similarity = jaccard_similarity(types_x, types_y)
        domains_similarity = jaccard_similarity(domains_x, domains_y)
        description_similarity = jaccard_similarity(description, description_y)

        # Combine similarities (sum of individual categories)
        combined_similarity = types_similarity + domains_similarity

        similarities.append((other_problem_name, pairs_similarity, combined_similarity, description_similarity))

    # Rank the problems based on the criteria
    ranked_problems = sorted(similarities, key=lambda x: (x[1], x[2], x[3]), reverse=True)

    # Create the RAG text
    rag_text = ""
    for problem, _, _, _ in ranked_problems[:top_k]:
        problem_df = constraint_df[constraint_df.problem_name == problem]
        if format_type == RAGFormat.PROBLEM_DESCRIPTION_OBJECTIVE:
            rag_text += f"Problem: {problem_df.iloc[0]['description']}\n\nObjective:\n{problem_df.iloc[0]['objective_description']}\n\n"
        elif format_type == RAGFormat.PROBLEM_DESCRIPTION_CONSTRAINTS:
            rag_text += f"Problem: {problem_df.iloc[0]['description']}\n\nConstraints:\n"
            for row in problem_df.itertuples():
                if row.constraint_description == "auxiliary constraint":
                    continue
                rag_text += f"{row.constraint_description}\n\n"
        elif format_type == RAGFormat.CONSTRAINT_FORMULATION:
            for row in problem_df.itertuples():
                rag_text += f"{row.constraint_description}\n{row.constraint_formulation}\n\n"
        elif format_type == RAGFormat.OBJECTIVE_FORMULATION:
            rag_text += f"Objective:\n{problem_df.iloc[0]['objective_description']}\n{problem_df.iloc[0]['objective_formulation']}\n\n"

    return rag_text
