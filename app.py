import streamlit as st
import spacy
import nltk
import pandas as pd
from textblob import TextBlob
from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from streamlit_option_menu import option_menu

nltk.download('punkt')


st.set_page_config(
    page_title="NLP App",  
    layout="centered",
    initial_sidebar_state="expanded",   
)

hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>

                """
st.markdown(hide_st_style, unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Home", "Contact"],
        icons=["house", "envelope"],
        menu_icon="cast",
        default_index=0,

    )
if selected == "Home":
    def sumy_summarizer(doc):
        parser = PlaintextParser.from_string(doc, Tokenizer("english"))
        lex_summarizer = LexRankSummarizer()
        summary = lex_summarizer(parser.document, 3)
        summary_list = [str(sentence) for sentence in summary]
        result = ' '.join(summary_list)
        return result


    def text_analyzer(my_text):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(my_text)
        
        tokens = [token.text for token in doc]
        allData = [('"Tokens":{},\t"Lemma":{}'.format(token.text,token.lemma_)) for token in doc]
        return allData



    def entity_analyzer(my_text):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(my_text)
        entities = [(entity.text, entity.label_) for entity in doc.ents]
        return entities



    def main():

        st.title("Natural Language Processing Application with Streamlit and spaCy")
        st.write("This application built with Streamlit and spaCy provides users with an interactive and informative experience, allowing them to explore tokenization, lemmas, named entities, sentiment analysis, and text summarization capabilities.")


        # Tokenization
        if st.checkbox("Show Tokens and Lemma"):
            st.subheader("Tokenize Your Text")
            message = st.text_area("Enter Your Text", "Type Here")
            if st.button("Analyze"):
                nlp_result = text_analyzer(message)
                st.json(nlp_result)
                df = pd.DataFrame(nlp_result)
                df_result = df.to_json(path_or_buf=None, orient=None, date_format=None, double_precision=10, force_ascii=True, date_unit='ms', default_handler=None, lines=False, compression='infer', index=True)
                st.download_button(label="Download", data=df_result, mime="text/plain", file_name="Tokens_Lemmas")


        # Named Entity
        if st.checkbox("Show Named Entities"):
            st.subheader("Extract Entities From Your Text")
            message = st.text_area("Enter Your Text", "Type Here")
            if st.button("Extract"):
                nlp_result = entity_analyzer(message)
                st.json(nlp_result)
                df = pd.DataFrame(nlp_result)
                df_result = df.to_string()
                st.download_button(label="Download", data=df_result, mime="text/plain", file_name="Named_Entities")


        # Sentiment Analysis
        if st.checkbox("Show Sentiment Analysis"):
            st.subheader("Sentiment of Your Text")
            message = st.text_area("Enter Your Text", "Type Here")
            if st.button("Analyze"):
                blob = TextBlob(message)
                result_sentiment = blob.sentiment
                st.success(result_sentiment)


        # Text Summarization
        if st.checkbox("Show Text Summarization"):
            st.subheader("Summarize Your Text")
            message = st.text_area("Enter Your Text", "Type Here")
            summary_options = st.selectbox("Choice Your Summarizer", ("Gensim", "Sumy"))
            if st.button("Summarize"):
                if summary_options == 'Gensim':
                    st.text("Using Gensim...")
                    summary_result = summarize(message)
                elif summary_options == 'Sumy':
                    st.text("Using Sumy...")
                    summary_result = sumy_summarizer(message)
                else:
                    st.warning("Using Default Summarizer")
                    st.text("Using Gensim")
                    summary_result = summarize(message)
                st.success(summary_result)
                st.download_button(label="Download", data=summary_result, file_name="Summary")


    if __name__ == '__main__':
        main()

if selected == "Contact":
    st.title("Contact us")
    st.write("If you have any questions, concerns or need help using our application to its full potential, our dedicated team is ready to assist you. Please feel free to contact us at any time via this form. We're here to answer your questions, resolve technical issues and guide you towards an exceptional user experience.")

    contact_form = """ 
    <form action="https://formsubmit.co/faysalyameogo1@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Details of your problem"></textarea>
        <button type="submit">Send</button>
    </form>

    """
    st.markdown(contact_form, unsafe_allow_html=True)


    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("./Style/file.css")