# HuggingFace Updates for LangChain

## Packages

We need to include several new packages & updates to our existing versions for our updates to utilizing the clients hosted HuggingFace llm. We need these new packages for the ChatHuggingFace chat model frm langchain to work, and in order to get this chatmodel we must update our langchain to version 0.0.352 Below is the command for all the essential component updates:

```bash
# New packages to install
pip install huggingface_hub transformers jinja2

# Packages to update
pip install langchain==0.0.352

# After all updates & additions, don't forget to update the requirements.txt
pip freeze > requirements.txt
```

## Environment Variables

We should be utilizing environment variables for the purposes of this lab as it can help prevent any leaking of tokens to the associates. NOTE: It'll still be possible if they find out the names of the environment variables, but it's a step in the right direction. Along with this `HF_ENDPOINT` will result in a invalid token error with ChatHuggingFace, please replace with `LLM_ENDPOINT`.

```bash
# For HuggingFaceHub to use the ChatHuggingFace chat model from LangChain
HF_TOKEN=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

#For HuggingFaceEndpoint directly generating responses from the llm
HUGGINGFACEHUB_API_TOKEN=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

# New Standard for Obfuscation of URL
LLM_ENDPOINT=https://z8dvl7fzhxxcybd8.eu-west-1.aws.endpoints.huggingface.cloud
```

To add these environment variables to the labs for your associates, follow these steps:

1. Log onto RevPro & view your training.
2. Once you're on your training that requires these environment variables, go to the settings on the side-panel.
   1. [Image: Locate Settings](https://revature0-my.sharepoint.com/:i:/g/personal/charles_jester_revature_com/Ed27Ga5li_9PsjJPEUoQlu4BTT-4GSLM_ez46EzulViogQ?e=p1UKPG)
3. Once in settings, scroll down to Cloud Lab Environment Variables. See image below
   1. [Image: Locate Cloud Environment in Settings](https://revature0-my.sharepoint.com/:i:/g/personal/charles_jester_revature_com/EeGqzXprmuBAjtmDALBK0lYBQLQURTieSaG9ecBS_pU7VQ?e=VDYQLQ)
4. Here we can add the above variables,
   1. List for easy copy & paste into the text field: `HF_TOKEN=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,HUGGINGFACEHUB_API_TOKEN=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`
   2. Once entered into the text field, press `enter` to add the environment variables, it should look like the image below.
      1. [Image: After environment variables added](https://revature0-my.sharepoint.com/:i:/g/personal/charles_jester_revature_com/EZKjQotzBsBOnhZKi3crYuUBgBJQGjaxM72fs7roAFzTrQ?e=AQKXcC)
   3. Finally, click save.
      1. The update to the environment on gitpod seems to take awhile, though this could have been due to the timing of when I added these. Please do ahead

## Models: LLM & Chat Model

Due to the update to utilizing of HuggingFaceEndpoint & ChatHuggingFace I've provided the swaps necessary. Note: HuggingFaceEndpoint is prone to a lot of hallucinations so it would be ideal to use ChatHuggingFace whenever we can. We need to perform the following:

1. Remove all old "OpenAI" instances, such as
   1. Environment variables
   2. AzureOpenAI instances. llm model
   3. AzureChatOpenAI instances. chat model
2. Replace with the updated code blocks
   1. Environment variable updates will automatically be read by the classes. No need to directly insert into constructors.
   2. `HuggingFaceEndpoint` is now our llm model. See code block below.
   3. `ChatHuggingFace` is now our chat_model. See Code Block below.

### Example of code that needs to be removed & replaced.

```python
# Remove the following occurences, they may not match exactly to what everyones written in their labs.
api_key = os.environ['OPENAI_API_KEY']
base_url = os.environ['OPENAI_API_BASE']
version = os.environ['OPENAI_API_VERSION']

llm = AzureOpenAI(model_name="gpt-35-turbo")
chat_model = AzureChatOpenAI(model_name="gpt-35-turbo")

# Add the following where necessary.
llm = HuggingFaceEndpoint(
        endpoint_url=os.environ['LLM_ENDPOINT'],
        task="text2text-generation",
        model_kwargs={
            "max_new_tokens": 200
        }
    )
chat_model = ChatHuggingFace(llm=llm)

```

## Making a standard utility for our lab_test

Below is the new standard utility file we should use for our tests. This is to help ensure that the test_lab.py includes the minimum requirements for the associates to pass. Any other utility function will be provided by the `llm_testing_util.py`.

- Please place this utility within `src/utilities/llm_testing_util.py`
- Includes the following functions:
  - `llm_wakeup`: A wake up function for the llm via an API request to the HuggingFace endpoint.
  - `llm_connection_check`: A connection check of our llm from the HuggingFaceEndpoint class via langchain
  - `classify_relevancy`: Some have been utilizing for checking if the response from the model makes sense.

### Update in `src/test/test_lab.py`:

```python
from src.utilities.llm_testing_util import llm_connection_check, llm_wakeup, classify_relevancy


class TestLLMResponses(unittest.TestCase):

    """
    This function is a sanity check for the Language Learning Model (LLM) connection.
    It attempts to generate a response from the LLM. If a 'Bad Gateway' error is encountered,
    it initiates the LLM wake-up process. This function is critical for ensuring the LLM is
    operational before running tests and should not be modified without understanding the
    implications.
    Raises:
        Exception: If any error other than 'Bad Gateway' is encountered, it is raised to the caller.
    """
    def test_llm_sanity_check(self):
        try:
            response = llm_connection_check()
            self.assertIsInstance(response, LLMResult)
        except Exception as e:
            if 'Bad Gateway' in str(e):
                llm_wakeup()
                self.fail("LLM is not awake. Please try again in 3-5 minutes.")
```

### Add [llm_testing_util.py](https://revature0-my.sharepoint.com/:u:/g/personal/charles_jester_revature_com/EZHysoDuvAVEg6E2kVAyYBQBcR81fbBNOc51bEE2hq8Qng?e=6pMrqb) to `src/utilities` directory

```python
import os
import requests
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

token = os.environ['HF_TOKEN'] # Ideally, we have this token set. Otherwise, replace with hardcoded HF token.
API_URL = os.environ['LLM_ENDPOINT']
headers = {"Authorization": "Bearer " + token}
textInput = """
<|system|>
You are a pirate chatbot who always responds with Arr!</s>
<|user|>
{userInput}</s>
<|assistant|>
"""

llm = HuggingFaceEndpoint(
        endpoint_url=API_URL,
        task="text2text-generation",
        model_kwargs={
            "max_new_tokens": 200
        }
    )


"""
This function is used to wake up the Language Learning Model (LLM) by sending a POST request to the API endpoint.
It uses a predefined text input to initiate the wake-up process. The response from the API is printed to the console.
"""
def llm_wakeup():
    response = requests.post(API_URL, json={
        "inputs": textInput.format(userInput="Hello, how are you?")
    }, headers=headers)
    print(response.json())


"""
This function checks the connection to the Language Learning Model (LLM) by generating a response to a predefined
greeting. This utilizes the HuggingFaceEndpoint LLM, as opposed to the llm_wakeup() function, which utilizes a direct
request to the HuggingFace API.

Returns: Response generated by the LLM.
"""
def llm_connection_check():
    return llm.generate(["Hello, how are you?"])


"""
This function classifies the relevancy of a given message to a question.
It uses a chatbot model to determine if the message properly answers the question.
The chatbot responds with "yes" if the message is relevant, and "no" otherwise. STRICTLY used
within the context of test_lab.py testing for these labs.

Parameters:
message (str): The message to be classified.
question (str): The question to which the message is supposed to respond.

Returns:
bool: True if the message is relevant (i.e., it answers the question), False otherwise.
"""
def classify_relevancy(message, question):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are a chatbot who determines if a given message properly answers a question by "
                              "replying 'yes' or 'no''."),
        HumanMessagePromptTemplate.from_template("Does the following message answer the question: {question}? message: {message}"
        ),
    ])

    chat_model = ChatHuggingFace(llm=llm)
    chain = prompt | chat_model | StrOutputParser()
    result = chain.invoke({"message": message, "question": question})
    print("Result: " + result)
    print(message)
    print(question)
    if "yes" in result.lower():
        if "i will respond with 'yes' or 'no'" in result.lower():
            print("Message is not relevant")
            return False
        else:
            return True
    else:
        print(message)
        return False

```
