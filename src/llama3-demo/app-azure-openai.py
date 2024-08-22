import openai
import json
import os
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain import LLMChain

# Load config values
with open(r'config.json') as config_file:
    config_details = json.load(config_file)

# The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
openai_api_base = config_details['OPENAI_API_BASE']

# API version e.g. "2023-07-01-preview"
openai_api_version = config_details['OPENAI_API_VERSION']

# The name of your Azure OpenAI deployment chat model. e.g. "gpt-35-turbo-0613"
deployment_name = config_details['DEPLOYMENT_NAME']

# The API key for your Azure OpenAI resource.
openai_api_key = os.getenv("OPENAI_API_KEY")

# This is set to `azure`
openai_api_type = "azure"

# Create an instance of chat llm
llm = AzureChatOpenAI(
    openai_api_base=openai_api_base,
    openai_api_version=openai_api_version,
    deployment_name=deployment_name,
    openai_api_key=openai_api_key,
    openai_api_type=openai_api_type,
)

llm([HumanMessage(content="Write me a poem")])

from langchain import PromptTemplate

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


# First Example
template = """
You are a skin care consulant that recommends products based on customer
needs and preferences.

What is a good {product_type} to help with {customer_request}?
"""

prompt = PromptTemplate(
input_variables=["product_type", "customer_request"],
template=template,
)

print("Example #1:")
print(llm([HumanMessage(content=prompt.format(
        product_type="face wash",
        customer_request = "acne prone skin"
    ))]
))
print("\n")

# Second Example
system_message = "You are an AI assistant travel assistant that provides vacation recommendations."

system_message_prompt = SystemMessagePromptTemplate.from_template(system_message)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chain = LLMChain(llm=llm, prompt=chat_prompt)
result = chain.run(f"Where should I go on vaction in Decemember for warm weather and beaches?")
print("Example #2:")
print(result)

