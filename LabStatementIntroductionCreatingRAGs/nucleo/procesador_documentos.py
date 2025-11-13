from typing import List, Optional
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import bs4


class ProcesadorDocumentos:
    def __init__(self, tamano_chunk: int = 1000, solapamiento: int = 200):
        self.divisor_texto = RecursiveCharacterTextSplitter(
            chunk_size=tamano_chunk,
            chunk_overlap=solapamiento
        )
    
    def cargar_desde_archivo(self, ruta_archivo: str) -> List[Document]:
        return TextLoader(ruta_archivo, encoding='utf-8').load()
    
    def cargar_desde_web(self, urls: List[str], clases_parsear: Optional[tuple] = None) -> List[Document]:
        if clases_parsear is None:
            clases_parsear = ("post-content", "post-title", "post-header")
        return WebBaseLoader(
            web_paths=urls,
            bs_kwargs={"parse_only": bs4.SoupStrainer(class_=clases_parsear)}
        ).load()
    
    def dividir_documentos(self, documentos: List[Document]) -> List[Document]:
        return self.divisor_texto.split_documents(documentos)
    
    def procesar_texto_completo(self, ruta_archivo: str) -> List[Document]:
        return self.dividir_documentos(self.cargar_desde_archivo(ruta_archivo))
    
    def procesar_web_completo(self, urls: List[str], clases_parsear: Optional[tuple] = None) -> List[Document]:
        return self.dividir_documentos(self.cargar_desde_web(urls, clases_parsear))
