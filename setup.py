from setuptools import setup, find_packages

setup(
    name="rag-chatbot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.32.0,<1.33.0",
        "langgraph>=0.0.15",
        "ollama>=0.1.6",
        "sentence-transformers>=2.5.1",
        "python-dotenv>=1.0.0",
        "PyPDF2>=3.0.0",
        "torch>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
        ],
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="A RAG-based PDF Question Answering Chatbot",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rag-chatbot",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 