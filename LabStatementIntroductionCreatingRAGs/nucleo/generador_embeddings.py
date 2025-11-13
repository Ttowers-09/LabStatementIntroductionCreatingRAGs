from langchain_openai import OpenAIEmbeddings


class GeneradorEmbeddings:
    def __init__(self, modelo: str = "text-embedding-ada-002"):
        self.embeddings = OpenAIEmbeddings(model=modelo)
    
    def obtener_instancia(self) -> OpenAIEmbeddings:
        return self.embeddings
