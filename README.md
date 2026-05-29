# CineAssist

**CineAssist** is an AI-powered movie recommendation chatbot that helps users find movies through natural language preferences instead of manually browsing long lists or rigid filters.

Users can describe what they want to watch using details such as genre, mood, language, age rating, favorite movies, year range, or themes. The system processes those preferences, compares them with a prepared movie dataset, and returns ranked movie recommendations with short explanations.

---

## Project Information

| Field | Details |
|---|---|
| Course | AML-2403 AI and ML Lab |
| Semester | Spring 2026 |
| Section | OTT01 |
| Group | Group 1 |
| Project Title | CineAssist: An AI Movie Recommendation Chatbot with Optional Multilingual Support |
| Faculty Supervisor | William Pourmajidi |
| Initial Team Lead | Alejandro Parparcen Grillet |

---

## Team Members

| Name | Student ID | Main Contribution Area |
|---|---:|---|
| Alejandro Parparcen Grillet | C0960408 | Project leadership, architecture, integration, GitHub coordination, backend/UI support |
| Carlos Antonio Graniel Manrique | C0966684 | Movie database, recommendation logic, feature schema, UI support |
| Lili Marcela Perez Clavijo | C0964898 | Chatbot conversation flow and NLP preference extraction |
| Brayan Yesid Roncancio Suarez | C0966032 | Multilingual/translation feasibility and model research |
| David Aponte Monroy | C0967956 | Testing, evaluation, metrics, model comparison, optional computer vision research |
| Motunrayo Aduloju | C0968107 | Dataset preparation, user-flow validation, feature engineering, prototype support |

---

## Problem Statement

Choosing a movie can be time-consuming because streaming platforms often provide too many options, limited filters, and recommendations that are not always aligned with a user's current mood or context. A user may want something specific, such as:

- A highly rated romantic comedy from the 2000s
- A family-friendly adventure movie
- A suspense movie similar to one they already enjoyed
- A movie in a specific language or age rating

CineAssist addresses this problem by allowing users to describe their preferences naturally and receive personalized, explainable movie recommendations.

---

## Project Goal

The goal is to build a working AI/ML prototype where a user can enter movie preferences, receive ranked recommendations, and understand why each movie was suggested.

The project combines:

- Dataset preparation
- Feature engineering
- Natural language processing
- Similarity-based recommendation
- Chatbot interaction design
- Model evaluation
- UI/API integration
- GitHub-based collaboration and documentation

---

## MVP Scope

The minimum viable product focuses on a task-specific movie recommendation chatbot.

### Included in the MVP

- Public or static movie dataset
- Data cleaning and preprocessing
- Movie feature vectors using metadata such as genres, descriptions, keywords, ratings, year, and language
- Baseline recommender using simple filters
- ML/NLP recommender using TF-IDF and cosine similarity
- Basic preference extraction from user input
- Chatbot-style interaction flow
- Ranked movie recommendations
- Short explanation for each recommendation
- Evaluation using predefined user scenarios and metrics
- GitHub repository with code, issues, progress evidence, and documentation

### Out of Scope for the MVP

- General-purpose conversational AI
- Live streaming data integration
- Full production deployment requirements
- Recommendation for product categories beyond movies
- Paid APIs unless a free academic option is confirmed

---

## Stretch Goals

These features are optional and should only be considered after the core recommender chatbot is stable.

- **Multilingual support:** Allow users to enter preferences or receive responses in more than one language.
- **Face-recognition login:** Explore a simple face-recognition login module using OpenCV or MediaPipe.
- **Deployment:** Deploy the prototype using a free or accessible platform such as Streamlit Community Cloud, Render, or Railway.

---

## Planned Technology Stack

| Category | Tools / Methods | Purpose |
|---|---|---|
| Programming Language | Python | Data processing, machine learning, NLP, evaluation, backend prototype |
| ML / Data Libraries | pandas, NumPy, scikit-learn | Cleaning, feature extraction, similarity scoring, model comparison |
| NLP Methods | TF-IDF, CountVectorizer, cosine similarity, keyword extraction | Represent movie descriptions and user preferences for recommendation |
| Data Storage | CSV or SQLite for MVP | Store movie metadata, cleaned features, and optional feedback |
| Interface | Streamlit first; React/Next.js + FastAPI if time allows | User-facing chatbot and recommendation screen |
| Computer Vision | OpenCV / MediaPipe | Optional face-recognition login exploration |
| Translation | Translation API or open-source library to be confirmed | Optional bilingual input/output |
| Development Tools | Jupyter Notebook, VS Code | Experimentation and collaboration |
| Version Control | GitHub and GitHub Projects/Issues | Code management, sprint tasks, and weekly evidence |
| Deployment | Streamlit Community Cloud, Render, Railway, or similar | Optional working demo deployment |

---

## Methodology

The project follows an Agile parallel sprint approach. Several workstreams can run at the same time, including dataset preparation, NLP preference extraction, chatbot flow, UI/API skeleton, and evaluation planning.

### Main Steps

1. **Data Preparation**
   - Collect a movie metadata dataset.
   - Clean missing or unusable records.
   - Standardize genres, language, ratings, descriptions, and other relevant fields.

2. **Baseline Recommender**
   - Build a simple filtering recommender using fields such as genre, rating, and year.
   - Use this as the comparison point for the ML/NLP recommender.

3. **ML/NLP Recommender**
   - Create movie feature vectors using metadata and text descriptions.
   - Apply TF-IDF or CountVectorizer.
   - Use cosine similarity to match user preferences with movie records.

4. **Preference Extraction**
   - Extract user preferences from natural language input.
   - Identify possible genres, moods, language preferences, rating constraints, year ranges, or similar movie titles.

5. **Chatbot Integration**
   - Build a simple chatbot-style flow.
   - Ask clarifying questions when needed.
   - Return ranked movie recommendations.

6. **Evaluation**
   - Test with predefined user scenarios.
   - Compare baseline and ML/NLP results.
   - Use manual relevance scoring and precision-style metrics where appropriate.

7. **Documentation and Delivery**
   - Organize the project in GitHub.
   - Track weekly progress through issues and reports.
   - Prepare the final report, presentation, demo, and submission package.

---

## Repository Structure

The final structure may change as the project develops, but the planned organization is:

```text
cineassist/
├── data/
│   ├── raw/                  # Original dataset files
│   ├── processed/            # Cleaned dataset files
│   └── README.md             # Dataset notes and field descriptions
│
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── baseline_recommender.ipynb
│   ├── feature_engineering.ipynb
│   └── evaluation.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── recommender.py
│   ├── preference_extraction.py
│   ├── chatbot_flow.py
│   └── evaluation.py
│
├── app/
│   ├── streamlit_app.py
│   └── components/
│
├── reports/
│   ├── weekly_progress/
│   └── final_report/
│
├── tests/
│   └── test_recommender.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd cineassist
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` has not been finalized yet, install the expected core dependencies manually:

```bash
pip install pandas numpy scikit-learn streamlit
```

### 4. Add dataset files

Place the selected movie dataset inside:

```text
data/raw/
```

Then run the preprocessing script or notebook to generate cleaned data inside:

```text
data/processed/
```

---

## Running the Prototype

If using Streamlit:

```bash
streamlit run app/streamlit_app.py
```

The application should allow the user to enter movie preferences and receive ranked recommendations with explanations.

---

## Example User Inputs

```text
I want a funny family movie from the 2000s.
```

```text
Recommend me a suspense movie similar to Inception.
```

```text
I want a romantic comedy with a good rating, preferably in English.
```

```text
Find me an adventure movie that is appropriate for a younger audience.
```

---

## Expected Output

For each user query, the system should return a ranked list of movies similar to:

```text
1. Movie Title
   Match reason: Similar genre, matching mood, high rating, and close year range.

2. Movie Title
   Match reason: Shares keywords from the user's request and fits the selected genre.

3. Movie Title
   Match reason: Recommended because it matches the requested theme and language.
```

---

## Evaluation Plan

The project will evaluate recommendation quality through:

- Predefined test queries
- Manual relevance scoring
- Comparison between simple baseline filtering and the ML/NLP recommender
- Precision-style metrics where practical
- Review of whether the recommendation explanation matches the user's request

Example evaluation questions:

- Did the system return movies that match the requested genre or mood?
- Did the system correctly use language, rating, or year constraints?
- Are the recommendation explanations understandable?
- Does the ML/NLP recommender improve over the baseline filter?

---

## Project Roadmap

| Week | Planned Work |
|---:|---|
| 1 | Finalize scope, proposal, Weekly Progress Report 1, and initial backlog |
| 2 | Create GitHub repo/project board, select dataset, define MVP, start cleaning |
| 3 | Build dataset pipeline, baseline recommender, NLP prompts, chatbot flow, UI/API skeleton, and evaluation cases |
| 4 | Feature engineering, TF-IDF/cosine recommender, NLP extraction, UI mock responses |
| 5 | Compare recommender results, build preference extraction, implement chatbot flow, run bilingual feasibility spike |
| 6 | Integrate recommender, NLP, and UI/API; complete evaluation plan |
| 7 | Improve UI, recommendation explanations, feedback collection, and stability |
| 8 | Run evaluation, refine recommender and preference extraction, fix major bugs |
| 9 | Review scope and decide whether multilingual or face-login stretch goals are realistic |
| 10 | Complete final functionality, bug fixes, deployment preparation, report outline |
| 11 | Final test pass, technical report draft, presentation visuals |
| 12 | Final report, presentation, code cleanup, README, and submission package |

---

## Current Status

As of Weekly Progress Report 1, the project is in the early planning stage. The team has selected the movie recommendation chatbot as the main project direction, defined multilingual support and face-recognition login as stretch goals, and identified the next priorities:

- Create the GitHub repository and project board
- Select and document the movie dataset
- Define MVP acceptance criteria
- Assign Week 2 sprint issues
- Begin dataset cleaning, NLP planning, evaluation planning, and prototype design

Estimated completion at the end of Week 1: **5%**.

---

## Collaboration Workflow

The team will use GitHub to track development and weekly progress.

Recommended workflow:

1. Create a GitHub issue for each task.
2. Assign an owner and sprint week.
3. Use branches for feature work.
4. Open pull requests for review.
5. Link commits and pull requests to issues.
6. Keep weekly evidence in the repository.
7. Update progress reports and documentation after each sprint.

Suggested issue labels:

- `data`
- `nlp`
- `recommender`
- `ui`
- `evaluation`
- `documentation`
- `stretch-goal`
- `bug`
- `week-2`, `week-3`, etc.

---

## License

This project is being developed for academic purposes as part of AML-2403 AI and ML Lab. A formal license can be added later if the team decides to make the repository public.

---

## Acknowledgements

This project is part of the AML-2403 AI and ML Lab capstone work for the Spring 2026 semester under the supervision of William Pourmajidi.
