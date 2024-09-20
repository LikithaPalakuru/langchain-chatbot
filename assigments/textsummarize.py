from langchain_openai import OpenAI
import os
from langchain_core.language_models import LLM
from langchain.chains.summarize import load_summarize_chain
api_key=os.getenv("OPENAI_API_KEY")
llm=OpenAI(openai_api_key=api_key)
from langchain.schema import(
    AIMessage,
    HumanMessage,SystemMessage
)
python="""
Creating a module with a set of functions to perform useful tasks is certainly
an effective way to approach many problems. If you find that this approach
can solve all your problems, you may be content to use it and not explore
other possible ways of getting things done. However, Python provides a
simple method for implementing object oriented programming through the
use of classes. We’ve already seen situations where we’ve taken advantage of
these ideas through the methods available for manipulating lists, dictionaries
and file objects. File objects in particular show the power of these techniques.
Once you create a file object, it doesn’t matter where it came from - when you
want to, say, read a line from such an object, you simply invoke the readline
method. The ability to do what you want without having to worry about
the internal details is a hallmark of object oriented programming. The main
tool in Python for creating objects is the class statement. When you create
a class, it can contain variables (often called attributes in this context) and
methods. To actually create an object that you can use in your programs,
you invoke the name of its class; in object oriented lingo we say that the
class name serves as a constructor for the object. Methods are defined in
a similar way to functions with one small difference
"""
print(python)
chat_message=[
    SystemMessage(content="you are expert in the summarizing of the text"),
    HumanMessage(content=f"please provide a short and good summary of the text:\n Text:{python}")
]
llm.get_num_tokens(python)
print(llm.get_num_tokens(python))
#Prompt template text summarization
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
generictemplate="""
write a summary of a text:
text:{python}
Translate the precise summary to {language}
"""
prompt=PromptTemplate(
    input_variables=['python','language'],
    template=generictemplate
)
print(prompt)
complete_prompt=prompt.format(python=python,language="French")
print(complete_prompt)
llm.get_num_tokens(complete_prompt)
print(llm.get_num_tokens(complete_prompt))
llm_chain=LLMChain(llm=llm,prompt=prompt)
summary=llm_chain.run({'python':'python','language':'French'})
print(summary)
#stuff document chain text summarization
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader("data/sample-pdf-file.pdf")
docs=loader.load_and_split()
print(docs)
final_documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
print(final_documents)
len(final_documents)
template="""write a concise and short summary about text 
python:{text}
"""
prompt=PromptTemplate(input_variables=['text'],template=template)
from langchain.chains.summarize import load_summarize_chain
chain=load_summarize_chain(llm,chain_type='stuff',prompt=prompt,verbose=True)
output_summary=chain.run(docs)
print(output_summary)
chunks_prompts="""
please summarize the below speech:
speech:'{text}'
summary:
"""
map_prompt_template=PromptTemplate(input_variables=['text'],template=chunks_prompts)
final_prompt='''
provide the final speech for the entire summary with an important points 
Add a title and note it in the points.
python:{text}
'''
final_prompt_template=PromptTemplate(input_variables=['text'],template=final_prompt)
print(final_prompt_template)
summary_chain=load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    map_prompt=map_prompt_template,
    combine_prompt=final_prompt_template,
    verbose=True
    

)
output=summary_chain.run(final_documents)
print(output)
#refine text summarization
chain=load_summarize_chain(
    llm=llm,
    chain_type="refine",
    verbose=True

)
output_summary=chain.run(final_documents)
print(output_summary)