from setuptools import find_packages,setup

setup(
    name='medicalchatbot',
    version='0.0.1',
    author='Shivam Kumar',
    author_email='shivamkumarsingh1064@gmail.com',
    install_requires=["langchain","langchain-community","sentence-transformers","streamlit","pinecone-client","python-dotenv","PyPDF2"],
    packages=find_packages()
)