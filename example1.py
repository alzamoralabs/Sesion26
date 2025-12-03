from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7
)

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="Eres un experto identificando el perfil psicologico de las personas." \
    "para ello elaboras preguntas que te permiten conocer mejor a la persona usando test de personalidad mbti" \
    "Luego, con base en sus respuestas, generas un perfil psicologico detallado. Siempre el cliente tiene la razon",
)

while True:
    user_input = input("Usuario: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })
    print("Agente:", result["messages"][-1].content)