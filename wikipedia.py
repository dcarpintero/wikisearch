class SearchEngine:
    """
    Class to perform keyword search, semantic search, and reranking
    """
    def __init__(self, cohere, weaviate):
        self.cohere = cohere
        self.weaviate = weaviate
        
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