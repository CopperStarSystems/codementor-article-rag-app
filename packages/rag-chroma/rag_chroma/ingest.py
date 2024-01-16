import requests
from bs4 import BeautifulSoup, SoupStrainer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from rag_chroma.chain import EMBEDDING_FUNCTION, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME

ARTICLE_URLS = [
    "https://www.codementor.io/@copperstarconsulting/getting-started-with-dependency-injection-using-castle-windsor-4meqzcsvh",
    "https://www.codementor.io/@copperstarconsulting/improve-testability-by-wrapping-static-classes-zfd45xl65",
    "https://www.codementor.io/@copperstarconsulting/test-driven-workflow-in-two-easy-steps-bfjln9sl4",
    "https://www.codementor.io/@copperstarconsulting/5-ways-test-driven-development-benefits-your-projects-as26987y0",
    "https://www.codementor.io/@copperstarconsulting/dynamic-test-cases-with-c-and-nunit-8tovg10gn",
    "https://www.codementor.io/@copperstarconsulting/intro-to-unit-testing-c-code-with-nunit-and-moq-part-1-y2b9iv8iq",
    "https://www.codementor.io/@copperstarconsulting/dependency-injection-a-gentle-introduction-697qjipog"
]

ARTICLE_CONTENT_CLASS = "md-viewer"

test_db_path = CHROMA_DB_PATH
vector_db = Chroma(persist_directory=test_db_path, collection_name=CHROMA_COLLECTION_NAME,
                   embedding_function=EMBEDDING_FUNCTION)


def retrieve_article_content(article_url):
    response = requests.get(article_url)
    return response.content


def extract_article_text(article_content):
    attrs_dict = {"class": ARTICLE_CONTENT_CLASS}
    strainer = SoupStrainer("div", attrs_dict)
    soup = BeautifulSoup(article_content, "html.parser", parse_only=strainer)
    return soup.get_text()


def get_metadata(article_url, article_content) -> dict:
    output = {"source_url": article_url}
    field_dictionary = {
        "article_likes": {"class": "btn-heart__number", "tag": "div", "inner_tag": "div"},
        "author_name": {"class": "author__name", "tag": "div", "inner_tag": "span"},
        "article_comments": {"class": "btn-frame-only engager__comment", "tag": "a", "inner_tag": "span"},
        "article_title": {"class": "article__header", "tag": "div", "inner_tag": "h1"}
    }

    for key in field_dictionary.keys():
        value = field_dictionary[key]
        attrs_dict = {"class": value["class"]}
        strainer = SoupStrainer(value["tag"], attrs_dict)
        soup = BeautifulSoup(article_content, "html.parser", parse_only=strainer)
        element = soup.find(value["inner_tag"])
        if element:
            output[key] = element.get_text()
    return output


def split_document(article_text, metadata):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,  # Set the desired chunk size (number of characters)
        chunk_overlap=50,  # Set the desired overlap between chunks (number of characters)
        length_function=len,  # Specify the function to measure the length of the text
        is_separator_regex=False  # Specify whether the separators are regular expressions
    )
    text = [article_text]
    return text_splitter.create_documents(text, [metadata])


def ingest_documents(documents):
    vector_db.add_documents(documents)
    vector_db.persist()


if __name__ == "__main__":
    documents = []
    for url in ARTICLE_URLS:
        print(f"Processing {url}")
        content = retrieve_article_content(url)
        article_text = extract_article_text(content)
        article_metadata = get_metadata(url, content)
        splits = split_document(article_text, article_metadata)
        for split in splits:
            documents.append(split)

    ingest_documents(documents)
