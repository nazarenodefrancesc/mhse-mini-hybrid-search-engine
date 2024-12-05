from setuptools import setup, find_packages

setup(
    name='hybrid_search',
    version='0.1.0',
    description='A hybrid search engine combining BM25 and FAISS with reranking capabilities.',
    author='Nazareno De Francesco',
    author_email='nazarenodefrancesco@gmail.com',
    # url='https://github.com/yourusername/hybrid_search',
    packages=find_packages(),
    install_requires=[
        'faiss-cpu==1.8.0',
        'sentence_transformers==3.2.1',
        'pandas==2.0.3',
        'numpy<2',
        'bm25s==0.2.3',
        'PyStemmer==2.2.0.2',
        'langchain-text-splitters==0.2.4',
        'langsmith==0.1.137',
        'langchain-core==0.2.41',
        'einops==0.8.0',
        'cohere==5.11.1',
        'lingua-language-detector==2.0.2',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)