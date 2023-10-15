# Import necessary modules
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load GPT4All model
llm = GPT4All(
    model="D:\LLM_models\TheBloke\Llama-2-13B-chat-GGML\llama-2-13b-chat.ggmlv3.q4_1.bin",
    max_tokens=2048,
    n_batch=512,
    n_predict=-1,
    temp=0.8,
    seed=-1,
    repeat_penalty=1.1,
    top_p=0.95,
    top_k=40,
    repeat_last_n = 64,
    use_mlock=True,
    f16_kv = True,    
)

# Define prompt template
prompt = PromptTemplate.from_template(
    """###Instruction:{chat}
    ###Response: """
)

# Create LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Run LLMChain
chat = "Tell me about Langchain"
result = llm_chain(chat)

# Output result
print(result["text"])
