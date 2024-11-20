import streamlit as st 




def main():

    # Design page layout 
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    st.header("Chat with Multiple PDFs :books:")
    st.text_input("Ask a question about your documents: ")

    # Side bar 
    with st.sidebar:
        st.subheader("Your Documents")
        st.file_uploader("Upload your PDFs and click 'Process' ")
        st.button("Process")
        
if __name__ == "__main__":
    main()