# 🦜📄 ChatPDF - Interactive Chatbot for Your PDFs

Welcome to **ChatPDF**, an interactive chatbot application that allows you to have conversational interactions with your PDF documents. Upload any PDF and start asking questions to extract information quickly and efficiently.

## 🚀 Demo

*Include a screenshot or GIF demonstrating the app here.*

## 📖 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## ✨ Features

- **Interactive Chat:** Engage in a conversational manner with your PDF documents.
- **Multiple PDF Support:** Upload and interact with multiple PDFs seamlessly.
- **Efficient Text Processing:** Leverages advanced NLP models for accurate and speedy responses.
- **Streamlit Interface:** User-friendly and responsive web interface powered by Streamlit.
- **Embeddings and Vector Storage:** Uses Sentence Transformer embeddings and Chroma for efficient text retrieval.
- **Caching Mechanism:** Efficiently caches resources and data to optimize performance.

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Natural Language Processing:** [Hugging Face Transformers](https://huggingface.co/transformers/), [LangChain](https://github.com/hwchase17/langchain)
- **Embeddings:** [Sentence Transformers](https://www.sbert.net/)
- **Vector Store:** [Chroma](https://www.trychroma.com/)
- **Language Model:** [LaMini-T5-738M](https://huggingface.co/MBZUAI/LaMini-T5-738M)
- **Programming Language:** Python 3.8+
- **Environment Management:** [virtualenv](https://virtualenv.pypa.io/en/latest/)

## ✅ Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.8 or higher installed on your system.
- `git` installed for cloning the repository.
- `virtualenv` installed for creating an isolated Python environment.
- [CUDA](https://developer.nvidia.com/cuda-downloads) installed if you plan to run the models on GPU for better performance.

## ⚙️ Installation

Follow these steps to set up the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ChatPDF.git
cd ChatPDF
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

- **On Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **On Unix or Linux/MacOS:**
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*Ensure that the `requirements.txt` file includes all necessary dependencies.*

### 5. Download Language Models and Set Up

The necessary models will be automatically downloaded when you run the application for the first time. Ensure you have a stable internet connection.

## ▶️ Usage

### 1. Run the Application

```bash
streamlit run chatbot_app.py
```

### 2. Upload PDF Documents

- Once the application is running, navigate to the local URL provided (usually `http://localhost:8501`).
- Use the file uploader to select and upload your PDF documents.

### 3. Start Chatting

- After uploading, the application will process the documents (this may take a few moments depending on the size and number of PDFs).
- Enter your queries in the chatbox, and the chatbot will provide responses based on the content of your PDFs.

### 4. Example Queries

- "Summarize the main points of the document."
- "What are the conclusions drawn in the third chapter?"
- "Provide statistics mentioned in section 2."

## 📁 Project Structure

```bash
ChatPDF/
│
├── docs/
│   └── # Folder where uploaded PDFs are stored.
│
├── db/
│   └── # Directory where Chroma stores embeddings.
│
├── chatbot_app.py
│   └── # Main Streamlit application file.
│
├── data_processor.py
│   └── # Handles document loading and processing.
│
├── requirements.txt
│   └── # Python dependencies.
│
├── constants.py
│   └── # Configuration constants for the project.
│
└── README.md
    └── # Project documentation.
```

## 🤝 Contributing

Contributions are welcome! Follow these steps to contribute:

1. **Fork the Repository**
   - Click the "Fork" button on the top right corner.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/iclal07/ChatPDF.git
   ```

3. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

4. **Commit Your Changes**

   ```bash
   git commit -m "Add your message"
   ```

5. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

6. **Open a Pull Request**
   - Navigate to the original repository and click on "New Pull Request".

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [OpenAI](https://openai.com/) for inspiration and foundational models.
- [Hugging Face](https://huggingface.co/) for providing accessible NLP models and tools.
- [Streamlit](https://streamlit.io/) for simplifying the creation of web apps.
- [LangChain](https://github.com/hwchase17/langchain) for powerful language model integrations.
- All open-source contributors and developers whose work made this project possible.

## 📧 Contact

For any inquiries or feedback, please reach out to:

- **İclal Çetin**
- **Email:** iclal.cetin.s.e@gmail.com
- **LinkedIn:** [İclal Çetin](https://www.linkedin.com/in/iclalcetin)

