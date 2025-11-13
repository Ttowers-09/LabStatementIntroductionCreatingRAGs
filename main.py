import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from configuracion import ConfiguracionAmbiente
from nucleo import ProcesadorDocumentos, GeneradorEmbeddings, MotorLenguaje
from servicios import GestorPinecone, CadenaRAG


def main():
    config = ConfiguracionAmbiente.cargar_desde_ambiente()
    config.configurar_ambiente()
    
    procesador = ProcesadorDocumentos(config.chunk_size, config.chunk_overlap)
    embeddings = GeneradorEmbeddings(config.openai_embedding_model)
    llm = MotorLenguaje(config.openai_model, config.temperature, config.max_tokens)
    
    gestor = GestorPinecone(config.pinecone_api_key, config.pinecone_index_name)
    gestor.crear_indice()
    
    while True:
        print("\n=== SISTEMA RAG ===")
        print("1. Procesar documento (archivo)")
        print("2. Procesar web (URL)")
        print("3. Consultar")
        print("4. Salir")
        
        opcion = input("\nOpción: ")
        
        if opcion == "1":
            ruta = input("Ruta del archivo: ")
            fragmentos = procesador.procesar_texto_completo(ruta)
            gestor.cargar_documentos(fragmentos, embeddings.obtener_instancia())
            print(f" {len(fragmentos)} fragmentos cargados")
            
        elif opcion == "2":
            url = input("URL: ")
            fragmentos = procesador.procesar_web_completo([url])
            gestor.cargar_documentos(fragmentos, embeddings.obtener_instancia())
            print(f" {len(fragmentos)} fragmentos cargados")
            
        elif opcion == "3":
            gestor.conectar_indice(embeddings.obtener_instancia())
            retriever = gestor.obtener_retriever(k=4)
            cadena = CadenaRAG(llm.obtener_instancia(), retriever)
            cadena.modo_interactivo()
            
        elif opcion == "4":
            break


if __name__ == "__main__":
    main()
