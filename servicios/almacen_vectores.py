from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


class AlmacenVectores:
    def __init__(self, embeddings, directorio_persistencia: str = "./chroma_db"):
        self.vectorstore = Chroma(persist_directory=directorio_persistencia, embedding_function=embeddings)
    
    def agregar_documentos(self, documentos: List[Document]) -> List[str]:
        return self.vectorstore.add_documents(documentos)
    
    def buscar_similitud(self, consulta: str, k: int = 4) -> List[Document]:
        return self.vectorstore.similarity_search(consulta, k=k)
    
    def obtener_retriever(self, k: int = 4):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
