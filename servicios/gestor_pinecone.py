from typing import List
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document


class GestorPinecone:
    def __init__(self, api_key: str, nombre_indice: str = "rag-index"):
        self.pc = Pinecone(api_key=api_key)
        self.nombre_indice = nombre_indice
        self.vectorstore = None
    
    def crear_indice(self, dimension: int = 1536, metrica: str = "cosine"):
        if self.nombre_indice not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.nombre_indice,
                dimension=dimension,
                metric=metrica,
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
    
    def cargar_documentos(self, documentos: List[Document], embeddings) -> List[str]:
        self.vectorstore = PineconeVectorStore.from_documents(
            documentos, embeddings, index_name=self.nombre_indice
        )
        return [str(i) for i in range(len(documentos))]
    
    def conectar_indice(self, embeddings):
        self.vectorstore = PineconeVectorStore.from_existing_index(
            index_name=self.nombre_indice, embedding=embeddings
        )
    
    def buscar_similitud(self, consulta: str, k: int = 4) -> List[Document]:
        return self.vectorstore.similarity_search(consulta, k=k)
    
    def obtener_retriever(self, k: int = 4):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
