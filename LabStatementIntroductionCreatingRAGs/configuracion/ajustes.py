import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class ConfiguracionAmbiente(BaseModel):
    openai_api_key: str = Field(...)
    openai_model: str = Field(default="gpt-3.5-turbo")
    openai_embedding_model: str = Field(default="text-embedding-ada-002")
    pinecone_api_key: str = Field(...)
    pinecone_environment: str = Field(default="gcp-starter")
    pinecone_index_name: str = Field(default="rag-knowledge-base")
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=2000)
    
    class Config:
        env_file = ".env"
    
    @classmethod
    def cargar_desde_ambiente(cls) -> 'ConfiguracionAmbiente':
        load_dotenv()
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            pinecone_environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter"),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "rag-knowledge-base"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            temperature=float(os.getenv("TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "2000"))
        )
    
    def configurar_ambiente(self) -> None:
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        os.environ["PINECONE_API_KEY"] = self.pinecone_api_key
        os.environ["PINECONE_ENVIRONMENT"] = self.pinecone_environment
