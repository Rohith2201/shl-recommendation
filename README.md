# SHL Assessment Recommendation System

An intelligent recommendation system that helps hiring managers find the right SHL assessments based on job descriptions or natural language queries.

## Features

- Natural language query processing
- Job description text/URL input support
- Intelligent assessment recommendations
- RESTful API endpoints
- Modern React frontend
- Semantic search capabilities
- Evaluation metrics (Mean Recall@K and MAP@K)

## Tech Stack

### Backend
- FastAPI
- Sentence Transformers
- Google Gemini API
- scikit-learn
- pandas

### Frontend
- React
- Material-UI
- Axios

## Setup

1. Clone the repository
2. Install backend dependencies:
```bash
pip install -r requirements.txt
```
3. Install frontend dependencies:
```bash
cd frontend
npm install
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```
GOOGLE_API_KEY=your_gemini_api_key
```

5. Run the backend:
```bash
uvicorn app.main:app --reload
```

6. Run the frontend:
```bash
cd frontend
npm start
```

## API Endpoints

- `GET /api/recommend?query={query}` - Get assessment recommendations based on query
- `POST /api/recommend/text` - Get recommendations based on job description text
- `GET /api/recommend/url?url={url}` - Get recommendations based on job description URL

## Evaluation Metrics

The system is evaluated using:
- Mean Recall@K
- Mean Average Precision@K (MAP@K)

## Architecture

The system uses a combination of semantic search and LLM-based processing:
1. Query/text preprocessing
2. Semantic embedding generation
3. Similarity-based retrieval
4. Result ranking and filtering
5. Metadata enrichment 