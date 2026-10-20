import streamlit as st            # streamlit run book_recommendation_system_web.py
import pickle 
import numpy as np

# Load Pickled data
books = pickle.load(open("books.pkl",'rb'))
pt=pickle.load(open("pt.pkl",'rb'))
similarity_scores=pickle.load(open("similarity.pkl",'rb'))

st.title("Book Recommendation System")
st.write("Collaborative Filtering model powered by Book-Crossing Dataset & Streamlit.")


# Recommendation function
def recommend(book_name):
    try:
        # Find index of the book
        index=np.where(pt.index==book_name)[0][0]

        # Get similar books sorted by similarity score
        similar_items=sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:6]

        recommended_books=[]
        for i in similar_items:
            item_title=pt.index[i[0]]
            recommended_books.append(item_title)

        return recommended_books
    except IndexError:
        return[]

# UI Layout: Dropdown for selecting a book
selected_book=st.selectbox("Select a book from the list:",pt.index.values)

# Button to trigger recommendation 
if st.button("Generate Recommendation "):
    recommendations=recommend(selected_book)

    if recommendations:
        st.subheader("Top Recommended Books:")
        for idx,rec in enumerate(recommendations,1):
            st.write(f"{idx}, **{rec}**")

        else:
            st.warning("No recommendation found.")