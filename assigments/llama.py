import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
#function to get response from the llamma response
def getLLammaresponse(input_text,no_words,blog_style) :
    llm=CTransformers(model='llama-2-7b-chat.ggmlv3.q2_K.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    template="""
        write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
        """
    prompt=PromptTemplate(input_variables=["blog_style","input_text","no_words"],
                          template=template)
    #we are generating a response from the llamma2 model
    response=llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    print(response)
    return response
st.set_page_config(page_title="Generate Blogs",
                   layout='centered',
                   initial_sidebar_state='collapsed')
st.header("Generative Blogs")
input_text=st.text_input("Enter the Blog Topic")
col1,col2=st.columns([5,5])
with col1:
    no_words=st.text_input("No of words")
with col2:
    blog_style=st.selectbox('writing the block for',('Researches','Data Scientist','common people'),index=0)
submit=st.button("Generate")
#by doing these we are going to get the final response
if submit:
    st.write(getLLammaresponse(input_text,no_words,blog_style))
