# Create a .env file in the project root with your API key

OPENAI_API_KEY=your_openai_api_key_here

# Run the server:

uvicorn server:app --reload

# API Endpoint
GET /query/

Parameters:
    station_id: The station ID (int).
    query: The query to send to the language model (string).

Response:
    station_id: The station ID (int).
    response: response of the query by LLM

# Sample API Call

http://127.0.0.1:8000/query?station_id=your_station_id&query=your_query
