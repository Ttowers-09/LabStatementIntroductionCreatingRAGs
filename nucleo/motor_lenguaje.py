from langchain_openai import ChatOpenAI, OpenAI


class MotorLenguaje:
    def __init__(
        self, 
        modelo: str = "gpt-3.5-turbo",
        temperatura: float = 0.0,
        max_tokens: int = 2000,
        tipo_motor: str = "chat"
    ):
        if tipo_motor == "chat":
            self.llm = ChatOpenAI(model=modelo, temperature=temperatura, max_tokens=max_tokens)
        else:
            self.llm = OpenAI(model=modelo, temperature=temperatura, max_tokens=max_tokens)
    
    def obtener_instancia(self):
        return self.llm

