from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from load_dotenv import load_dotenv
load_dotenv()

@tool
def echo_tool(input: str) -> str:
    """Tool that echoes the input."""
    return input

@tool
def double_echo_tool(input: str) -> str:
    """Tool that echoes the input twice."""
    return input + " " + input

agent = create_agent(
    model="gpt-4o-mini",
    tools=[echo_tool, double_echo_tool],
    middleware=[
        # Redact emails in user input before sending to model
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        # Mask credit cards in user input
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        # Block API keys - raise error if detected
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
        ),
    ],
)

# When user provides PII, it will be handled according to the strategy
result = agent.invoke({
    "messages": [{"role": "user", "content": "My email is john.doe@example.com and card is 5105-1051-0510-5100"}]
})
print(result)