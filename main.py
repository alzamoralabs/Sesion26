import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

# Inicializar modelo de lenguaje
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.8,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Función para generar poema
@tool
def compose_poem(keywords: str) -> str:
    """
    Genera un poema basado en 3 palabras clave.
    
    Args:
        keywords: Cadena con 3 palabras clave separadas por comas
        
    Returns:
        Un poema inspirado en las palabras clave
    """
    words = [w.strip() for w in keywords.split(",")]
    
    if len(words) != 3:
        return "Error: Se requieren exactamente 3 palabras clave separadas por comas"
    
    prompt = f"""Crea un poema hermoso e inspirador basado en estas 3 palabras clave:
    1. {words[0]}
    2. {words[1]}
    3. {words[2]}
    
    El poema debe:
    - Tener entre 12 y 16 versos
    - Incorporar las 3 palabras de manera natural
    - Tener una métrica y rima coherente
    - Transmitir una emoción o mensaje profundo
    
    Solo proporciona el poema, sin explicaciones adicionales."""
    
    response = llm.invoke(prompt)
    return response.content

# Definir herramientas
tools = [compose_poem]

# Crear agente
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="Eres un agente compositor de poemas. Tu misión es crear poemas hermosos basados en 3 palabras clave que el usuario proporciona. Usa la herramienta compose_poem para generar los poemas."
)

# Función principal
def main():
    """Ejecuta el compositor de poemas."""
    print("=" * 70)
    print("🖋️  COMPOSITOR DE POEMAS")
    print("=" * 70)
    print("Ingresa 3 palabras clave para que te genere un poema.\n")
    print("Escribe 'salir' para terminar\n")
    
    while True:
        keywords_input = input("Ingresa 3 palabras clave (separadas por comas): ").strip()
        
        if keywords_input.lower() in ["salir", "exit", "quit"]:
            print("\n¡Gracias por usar el compositor de poemas! 📝")
            break
        
        if not keywords_input:
            print("Por favor, ingresa palabras válidas.\n")
            continue
        
        try:
            print("\n⏳ Generando poema...\n")
            
            user_input = f"Crea un poema basado en estas palabras clave: {keywords_input}"
            
            result = agent.invoke({
                "messages": [{"role": "user", "content": user_input}]
            })
            
            # Obtener el último mensaje del resultado
            final_message = result["messages"][-1]
            print(f"{final_message.content}\n")
            print("-" * 70 + "\n")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")

if __name__ == "__main__":
    main()