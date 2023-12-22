import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
import autogen
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from backend.tools.searchtool import Search

load_dotenv()

openai_model: str = os.environ.get("OPENAI_MODEL")
openai.api_key = os.environ.get("OPENAI_API_KEY")
# create client for OpenAI
client = OpenAI(api_key=openai.api_key)
search: Search = Search()  # get instance of search to query corpus

config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4-1106-preview", "gpt-4-32k-0613"],
    },
)

gpt4_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list_gpt4,
    "timeout": 120,
}

# Function to perform a Shadow Search
def shadow_search(message):
    print("calling search")
    search_result = search.search_hybrid(message)
    return search_result

if __name__ == '__main__':
        
        # Retrieve an existing assistant already setup as an OpenAI Assistant
        # this is OpenAI Assistant stuff
        shadow_retriever_assistant = client.beta.assistants.retrieve(
                        assistant_id="asst_CDgesnP9G5fWP15UVBeQQfUX",
                        ) 

        # define the config including the tools that the assistant has access to
        # this will be used by the GPTAssistant Agent that is Shadow Retriever
        shadow_retriever_config = {
            "assistant_id": shadow_retriever_assistant.id,
            "tools": [
                {
                    "type": "function",
                    "function": shadow_search,
                }
                    ]
        }

        # this is autogen stuff defining the agent that is going to be in the group
        shadow_retriever_agent = GPTAssistantAgent(
            name="Retriever",
            instructions="""
            You are a data researcher that uses a tool to retrieve data.
            """,
            llm_config=shadow_retriever_config,
        )

        shadow_retriever_agent.register_function(
            function_map={
                "shadow_search": shadow_search,
            }
        )

        # this is autogen stuff defining the agent that is going to be in the group
        shadow_planner = autogen.AssistantAgent(
            name="Planner",
            system_message='''Planner. Suggest a plan. Revise the plan based on feedback from admin, until admin approval.
                The plan may involve retriever who can retrieve data.
                Explain the plan first. Be clear which step is performed by a retriever.
                ''',
            llm_config=gpt4_config,
        )

        user_proxy = autogen.UserProxyAgent(
            name="Admin",
            code_execution_config={
                "work_dir" : "coding",
            },
            system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin."
        )

        groupchat = autogen.GroupChat(agents=[user_proxy, shadow_retriever_agent, shadow_planner], messages=[], max_round=10)
        manager = autogen.GroupChatManager(groupchat=groupchat)

        print("initiating chat")

        user_proxy.initiate_chat(
            manager,
            message="""
            I have a first meeting with a prospect - what do I need to find out and what are the most important things I need to relate to them?
            """,
            silent=False
        )