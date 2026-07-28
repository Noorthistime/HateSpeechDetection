# Hate Speech Detection Web Application
A full-stack NLP and machine learning web application for detecting, classifying, and analyzing hate speech and offensive language using Python, Scikit-learn, NLTK, and Streamlit.

## Project Structure
```text
HateSpeech/
├── .streamlit/             # Streamlit theme and UI configurations
├── app.py                  # Full-stack Streamlit application (Frontend UI & Backend NLP logic)
├── Doc1Hate.py             # Data processing, cleaning & ML pipeline script
├── labeled_data.csv        # Dataset containing labeled hate speech & offensive language tweets
├── requirements.txt        # Python dependency specifications
└── README.md               # Project documentation
```

## Features
1. **Interactive NLP & ML Explorer**
   - Full-stack web application with responsive desktop and mobile layouts
   - Step-by-step interactive walkthrough of data loading, cleaning, and model evaluation
2. **Real-time Live Prediction Test**
   - Custom text input for real-time hate speech classification
   - Confidence breakdown across all three classification categories
   - Interactive text preprocessing breakdown (punctuation removal, stopword filtering, stemming)
3. **Data Processing & Cleaning Pipeline**
   - Automated URL, punctuation, and number stripping from tweet text
   - NLTK English stopword removal and Snowball stemming
4. **Model Evaluation & Visualizations**
   - Decision Tree Classifier training and evaluation
   - Confusion matrix and accuracy metrics visualization with Seaborn heatmaps
5. **Full Code & Dataset Explorer**
   - Interactive inspection of raw Python source code and dataset rows
   - Code explanation tooltips and interactive learning modules

## Technologies Used
- **Backend & ML:** Python 3.8+, Scikit-learn, NLTK, NumPy, Pandas
- **Frontend & Web Framework:** Streamlit (with custom CSS3, glassmorphism, and responsive media queries)
- **Data Visualization:** Seaborn, Matplotlib
- **Dataset:** Kaggle Hate Speech and Offensive Language Dataset

## Dataset Setup
### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git

### Dataset Details
- File name: `labeled_data.csv` (included in repository root)
- Source: Kaggle Hate Speech and Offensive Language Dataset

## Installation Steps
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd HateSpeech
   ```
2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/macOS
   source venv/bin/activate
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Building the Project
### Using pip / Python
```bash
# Verify environment and dependencies
pip check

# Download required NLTK corpus (handled automatically by app.py)
python -c "import nltk; nltk.download('stopwords')"
```

## Running the Application
1. Ensure your Python virtual environment is activated.
2. Start the Streamlit application server:
   ```bash
   streamlit run app.py
   ```
3. Access the application in your browser:
   ```
   http://localhost:8501
   ```

## Application Navigation
The application requires no authentication and provides an interactive sidebar menu with the following modules:
- **Project Overview:** High-level architectural and technical summary
- **1. Data Loading & Cleaning:** Inspect dataset rows, class distributions, and NLP cleaning steps
- **2. Model Training & Evaluation:** Check Decision Tree performance and confusion matrix
- **3. Live Prediction Test:** Test custom sentences against the trained model in real time
- **4. Full Code Explorer & 5. View Raw Source Code:** Deep dive into implementation details

## Application Modules
- `GET /` - Main Streamlit Web Application Interface
- `Sidebar -> Live Prediction Test` - Interactive NLP prediction pipeline
- `Sidebar -> Data Loading & Cleaning` - Interactive dataset exploration
- `Sidebar -> Model Training & Evaluation` - Model evaluation metrics & visualizations

## Dataset Schema (`labeled_data.csv`)
- `count` - Number of CrowdFlower users who coded each tweet
- `hate_speech` - Number of users who judged the tweet to be hate speech
- `offensive_language` - Number of users who judged the tweet to be offensive
- `neither` - Number of users who judged the tweet to be neither
- `class` - Overall classification label (0: Hate Speech, 1: Offensive Language, 2: Neither)
- `tweet` - Raw tweet text

## Classification Categories Supported
- **Class 0:** Hate Speech
- **Class 1:** Offensive Language
- **Class 2:** Neither (No hate or offensive language)

## Security & Pipeline Features
- Sanitized text input for live prediction
- Robust regex filtering for URL and special character removal
- Safe execution of Streamlit cached resources (`@st.cache_resource`)
- Responsive mobile layout preserving desktop UI integrity

## Future Enhancements
- Implementation of Transformer-based models (BERT / RoBERTa) for improved classification accuracy
- RESTful API endpoints using FastAPI for backend integration
- Multi-language hate speech detection support
- User authentication and prediction history logging
- Real-time Twitter/X API feed integration

## Troubleshooting
### NLTK Stopwords Error
- If `LookupError: Resource stopwords not found` occurs, run:
  ```bash
  python -m nltk.downloader stopwords
  ```
### Port Already in Use
- If default port 8501 is occupied, launch on an alternative port:
  ```bash
  streamlit run app.py --server.port 8502
  ```
### Dataset Not Found Error
- Ensure `labeled_data.csv` is located in the project root directory alongside `app.py`.

## Performance Optimization
- Utilizes Streamlit's `@st.cache_resource` decorator to cache dataset loading and model training
- Optimized vectorization pipeline using `CountVectorizer`
- Efficient Pandas DataFrame operations for text preprocessing

## License
This project is open source and available under the MIT License.

## Author
Developed by: Noor Mohammad

## Support
For issues and questions, please refer to the documentation or contact support.

## Deployment Checklist
- [x] Dataset `labeled_data.csv` present in repository root
- [x] `requirements.txt` dependencies verified
- [x] Streamlit page configuration and responsive CSS verified
- [x] Model training and evaluation pipeline tested
- [x] Live prediction module tested with custom input
- [x] Application successfully runs on local Streamlit server (`http://localhost:8501`)
