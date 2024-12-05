from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunk:
    id: int
    text: str

class ChunkedDocument:
    doc_id: int
    chunks: list[Chunk]


def split_text_documents_recursive_character(documents: list, chunk_size: int=2000, chunk_overlap: int=500):
    """
    Split a list of documents into chunks using a recursive character-based text splitter.

    :param documents:
    :param chunk_size: number of characters in each chunk
    :param chunk_overlap: number of characters to overlap between chunks
    :return:
        list of dictionaries, each containing the document id and a list of chunks
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunked_docs = []
    for idx, document in enumerate(documents):
        chunks = text_splitter.split_text(text=document)
        chunked_docs.append({
            "doc_id": idx,
            "chunks": chunks
        })

    return chunked_docs

if __name__ == "__main__":

    test_txt = open("test_data/test_ita.txt", "r").read()
    test_txt_2 = open("test_data/test_eng.txt", "r").read()

    chunked_docs = split_text_documents_recursive_character([test_txt, test_txt_2], chunk_size=2000, chunk_overlap=500)
    print(f"{len(chunked_docs)} chunks")