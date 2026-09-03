import streamlit as st

def home():
    st.Page('home.py', title='Home')
    st.header('Home Page')

pages = {
    'home':{
        st.Page(home)
    },
    'Models': {
        st.Page('app/linear.py', title='Linear'),
        st.Page('app/logistic.py', title='Logistic'),
        st.Page('app/svm.py', title='SVM_Classifier'),
        st.Page('app/k_means.py', title='Kmeans_clusters')
    }
}

pg = st.navigation(pages, position = 'top')
pg.run()