from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class CadenaRAG:
    def __init__(self, llm, retriever, plantilla_prompt: str = None):
        self.llm = llm
        self.retriever = retriever
        
        if plantilla_prompt is None:
            plantilla_prompt = """Usa el siguiente contexto para responder la pregunta. Si no sabes la respuesta, di que no lo sabes.

Contexto: {context}

Pregunta: {question}

Respuesta:"""
        
        self.prompt = ChatPromptTemplate.from_template(plantilla_prompt)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.cadena = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | llm
            | StrOutputParser()
        )
    
    def consultar(self, pregunta: str) -> dict:
        docs = self.retriever.get_relevant_documents(pregunta)
        resultado = self.cadena.invoke(pregunta)
        return {"result": resultado, "source_documents": docs}
    
    def modo_interactivo(self):
        print("\nModo Interactivo RAG")
        print("Escribe 'salir' para terminar\n")
        
        while True:
            pregunta = input("\nPregunta: ").strip()
            if pregunta.lower() in ['salir', 'exit', 'quit']:
                break
            if not pregunta:
                continue
            
            resultado = self.consultar(pregunta)
            print(f"\nRespuesta: {resultado['result']}")
            print(f"\nFuentes: {len(resultado.get('source_documents', []))} documentos")
