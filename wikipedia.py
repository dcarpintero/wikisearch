import logging
import os

from dotenv import load_dotenv
import cohere
import weaviate


class SearchEngine:
    """
    A Search Engine utility that performs keyword and semantic searches using Weaviate, and 
    reranks responses using Cohere.
    """
    WIKIPEDIA_PROPERTIES = ["text", "title", "url", "views", "lang", "_additional { distance score }"]

    def __init__(self):
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
        self.vars = self.__load_environment_vars()
        self.cohere = self.__cohere_client(self.vars["COHERE_API_KEY"])
        self.weaviate = self.__weaviate_client(self.vars["WEAVIATE_API_KEY"], 
                                               self.vars["COHERE_API_KEY"], 
                                               self.vars["WEAVIATE_URL"])
        logging.info("Initialized SearchEngine with Cohere and Weaviate clients")

    def with_bm25(self, query, lang='en', top_n=10) -> list:
        """
        Performs a keyword search (sparse retrieval) on Wikipedia Articles using embeddings stored in Weaviate.

        Parameters:
        - query (str): The search query.
        - lang (str, optional): The language of the articles. Default is 'en'.
        - top_n (int, optional): The number of top results to return. Default is 10.

        Returns:
        - list: List of top articles based on BM25F scoring.
        """
        logging.info("with_bm25()")
        where_filter = {
            "path": ["lang"],
            "operator": "Equal",
            "valueString": lang
        }
        response = (
            self.weaviate.query.get("Articles", self.WIKIPEDIA_PROPERTIES)
            .with_bm25(query=query)
            .with_where(where_filter)
            .with_limit(top_n)
            .do()
        )
        return response["data"]["Get"]["Articles"]
        
    def with_neartext(self, query, lang='en', top_n=10) -> list:
        """
        Performs a semantic search (dense retrieval) on Wikipedia Articles using embeddings stored in Weaviate.

        Parameters:
        - query (str): The search query.
        - lang (str, optional): The language of the articles. Default is 'en'.
        - top_n (int, optional): The number of top results to return. Default is 10.

        Returns:
        - list: List of top articles based on semantic similarity.
        """
        logging.info("with_neartext()")
        nearText = {
            "concepts": [query]
        }
        where_filter = {
            "path": ["lang"],
            "operator": "Equal",
            "valueString": lang
        }
        response = (
            self.weaviate.query.get("Articles", self.WIKIPEDIA_PROPERTIES)
            .with_near_text(nearText)
            .with_where(where_filter)
            .with_limit(top_n)
            .do()
        )
        return response['data']['Get']['Articles']
    
    def with_hybrid(self, query, lang='en', top_n=10) -> list:
        """
        Performs a hybrid search on Wikipedia Articles using embeddings stored in Weaviate.

        Parameters:
        - query (str): The search query.
        - lang (str, optional): The language of the articles. Default is 'en'.
        - top_n (int, optional): The number of top results to return. Default is 10.

        Returns:
        - list: List of top articles based on hybrid scoring.
        """	
        where_filter = {
            "path": ["lang"],
            "operator": "Equal",
            "valueString": lang
        }
        response = (
            self.weaviate.query.get("Articles", self.WIKIPEDIA_PROPERTIES)
            .with_hybrid(query=query)
            .with_where(where_filter)
            .with_limit(top_n)
            .do()
        )
        return response["data"]["Get"]["Articles"]
    
    def with_llm(self, context, query):
        prompt = f"""
            Use the information provided below to answer the questions at the end. /
            Include in the answer some curious or relevant fact extracted from the context, and related to the question. /
            Generate the answer in the language of the context. /
            If the answer to the question is not contained in the provided information, say "The answer is not in the context".
            ---
            Context information:
            {context}
            ---
            Question: 
            {query}
            """
        return self.cohere.generate(
            prompt=prompt,
            num_generations=1,
            max_tokens=1000,
            )
        
    def rerank(self, query, documents, top_n=10) -> dict:
        """
        Reranks a list of responses using Cohere's reranking API.

        Parameters:
        - query (str): The search query.
        - documents (list): List of documents to be reranked.
        - top_n (int, optional): The number of top reranked results to return. Default is 10.

        Returns:
        - dict: Reranked documents from Cohere's API.
        """
        return self.cohere.rerank(query=query, documents=documents, top_n=top_n, model='rerank-english-v2.0')
    
    def __load_environment_vars(self):
        """
        Load environment variables from .env file
        """
        logging.info("Loading environment variables...")

        load_dotenv()
        required_vars = ["COHERE_API_KEY", "WEAVIATE_API_KEY", "WEAVIATE_URL"]
        env_vars = {var: os.getenv(var) for var in required_vars}
        for var, value in env_vars.items():
            if not value:
                raise EnvironmentError(f"{var} environment variable not set.")
        
        logging.info("Environment variables loaded")
        return env_vars

    def __cohere_client(self, cohere_api_key):
        """
        Initialize Cohere client
        """
        return cohere.Client(cohere_api_key)

    def __weaviate_client(self, weaviate_api_key, cohere_api_key, cohere_url):
        """
        Initialize Weaviate client
        """
        auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_api_key)
        return weaviate.Client(
            url=cohere_url,
            auth_client_secret=auth_config,
            additional_headers={
                "X-Cohere-Api-Key": cohere_api_key,
            }
        )
