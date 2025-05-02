from typing import Tuple, Dict
import dotenv
import os
from dotenv import load_dotenv
import requests
import json
import streamlit as st
import os
from openai import OpenAI

from langsmith import wrappers, traceable


token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4o-mini"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)


load_dotenv()
EXCHANGERATE_API_KEY = os.getenv('EXCHANGE_RATE_API_KEY')

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "moneyconvertor"


tools = [
    {
        "type": "function",
        "function": {
            "name" : "get_currency_exchange",
            "description" : "Convert a given amount of money from one currency to another. Each currency will have a three letter code",
            "parameters": {
                "type": "object",
                "properties": {
                  "base": {
                      "type": "string",
                      "description": "The base or original currency"
                  },
                "target": {
                    "type": "string",
                    "description": "The target or converted currency"
                },
                "amount": {
                    "type": "string",
                    "description": "The amount of money to convert from the base currency"
                }
                }
            },
            "required": ["base", "target", "amount"]
        }
    }
]



@traceable
def get_exchange_rate(base: str, target: str, amount: str) -> Tuple:
    """Return a tuple of (base, target, amount, conversion_result (2 decimal places))"""
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGERATE_API_KEY}/pair/{base}/{target}/{amount}"
    response = json.loads(requests.get(url).text)

    return (base, target, amount, f'{response["conversion_result"]:.2f}')

print(get_exchange_rate ("GBP", "USD", "500"))


@traceable
def call_llm(textbox_input) -> Dict:
    """Make a call to the LLM with the textbox_input as the prompt.
       The output from the LLM should be a JSON (dict) with the base, amount and target"""
    
    
    
    try:
        response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": textbox_input,
            }
        ],
        temperature=1.0,
        top_p=1.0,
        max_tokens=1000,
        model=model_name,
        tools=tools,
)

    except Exception as e:
        print(f"Exception {e} for {text}")
    else:
        return response#.choices[0].message.content

@traceable
def run_pipeline(user_input):
    """Based on textbox_input, determine if you need to use the tools (function calling) for the LLM.
    Call get_exchange_rate(...) if necessary"""

    response = call_llm(user_input)
    #st.write(response)

    if response.choices[0].finish_reason == "tool_calls":
        # Update this
       
        response_arguments = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        base = response_arguments["base"]
        target = response_arguments["target"]
        amount = response_arguments["amount"]
        _,_,_, conversion_result = get_exchange_rate(base,target,amount)

        st.write(f'{base} {amount} is {target} {conversion_result}')

    elif response.choices[0].finish_reason == "stop":
        # Update this
        st.write(f"(Function calling not used) and {response.choices[0].message.content}")
    else:
        st.write("NotImplemented")

        #Title of the App
st.title("Multi Lingual Money Changer")

#Text Box for User Input

user_input = st.text_input("Enter your request here")

#Submit Button

if st.button("Submit"):
    #Display the input submitted bwlo the text box
    #st.write(call_llm(user_input))
    run_pipeline(user_input)
    
    # st.write (response_arguments)
