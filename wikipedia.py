from dotenv import load_dotenv
import cohere
import logging
import os
import weaviate


class SearchEngine:
    """
    Class to perform keyword search, semantic search, and reranking
    """
    def __init__(self):
        logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
        self.vars = self.__load_environment_vars()

        self.cohere = self.__cohere_client(self.vars["COHERE_API_KEY"])
        self.weaviate = self.__weaviate_client(
            self.vars["WEAVIATE_API_KEY"],
            self.vars["COHERE_API_KEY"],
            self.vars["WEAVIATE_URL"],
        )
        
    def by_keyword(self, query, lang='en', top_n=10):
        """
        Query the Weaviate database and return the top results.
        """	
        properties = ["text", "title", "url", "views", "lang", "_additional {distance}"]

        where_filter = {
            "path": ["lang"],
            "operator": "Equal",
            "valueString": lang
        }

        response = (
            self.weaviate.query.get("Articles", properties)
                .with_bm25(
                    query=query
                )
                .with_where(where_filter)
                .with_limit(top_n)
                .do()
        )
        result = response['data']['Get']['Articles']
        return result

    def dense_retrieval(self, query, results_lang='en', top_n=10):
        """
        Query the vectors database and return the top results.


        Parameters
        ----------
            query: str
                The search query

            results_lang: str (optional)
                Retrieve results only in the specified language.
                The demo dataset has those languages:
                en, de, fr, es, it, ja, ar, zh, ko, hi

        """

        nearText = {"concepts": [query]}
        properties = ["text", "title", "url", "views", "lang", "_additional {distance}"]
        # To filter by language
        where_filter = {
        "path": ["lang"],
        "operator": "Equal",
        "valueString": results_lang
        }
        response = (
            self.weaviate.query
                .get("Articles", properties)
                .with_near_text(nearText)
                .with_where(where_filter)
                .with_limit(top_n)
                .do()
        )

        result = response['data']['Get']['Articles']

        return result
    
    def rerank_responses(self, query, responses, top_n=10):
        reranked_responses = self.cohere.rerank(
            model = 'rerank-english-v2.0',
            query = query,
            documents = responses,
            top_n = top_n,
        )
        return reranked_responses
    
    def __load_environment_vars(self):
        """
        Load environment variables from .env file
        """
        logging.info("Loading environment variables...")

        load_dotenv()

        cohere_api_key = os.getenv("COHERE_API_KEY")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        weaviate_url = os.getenv("WEAVIATE_URL")

        if not cohere_api_key:
            raise EnvironmentError("COHERE_API_KEY environment variable not set.")
        
        if not weaviate_api_key:
            raise EnvironmentError("WEAVIATE_API_KEY environment variable not set.")
        
        if not weaviate_url:
            raise EnvironmentError("WEAVIATE_URL environment variable not set.")
        
        logging.info("Environment variables loaded.")
        return {"COHERE_API_KEY": cohere_api_key, "WEAVIATE_API_KEY": weaviate_api_key, "WEAVIATE_URL": weaviate_url}
        
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