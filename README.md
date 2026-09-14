# ✈️ TripMate AI — Multi-Agent Travel Planner

### AI-Powered Travel Planning with LangGraph, MCP, Groq & FastAPI

> **TripMate AI** is an end-to-end Generative AI application that transforms a natural-language travel request into a structured travel plan covering **flights, hotels, weather, day-by-day itinerary, and estimated budget**.

Built using **LangGraph + Model Context Protocol (MCP) + LangChain + Groq + FastAPI + PostgreSQL**, the system demonstrates how multiple specialized AI agents can be orchestrated into a production-oriented workflow.

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-Framework-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange.svg)](https://www.langchain.com/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-GenAI-green.svg)](https://www.langchain.com/)
[![MCP](https://img.shields.io/badge/MCP-Tool%20Integration-purple.svg)](https://modelcontextprotocol.io/)
[![Groq](https://img.shields.io/badge/Groq-LLM-black.svg)](https://groq.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Checkpointing-336791.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</p>

---

## 🚀 Live Demo

🔗 **GitHub Repository:**
https://github.com/Suraj-Kumar09/TripMate-AI-Planner-Using-MCP

> **Note:** API keys are required for external services such as Groq, Tavily, AviationStack and OpenWeather.

---

# 🎯 Why TripMate AI?

Traditional travel planning requires manually searching multiple platforms for:

* ✈️ Flights
* 🏨 Hotels
* 🌤️ Weather
* 🗺️ Places to visit
* 💰 Budget planning
* 📅 Daily itinerary

TripMate AI combines these tasks into a single AI-driven workflow.

### Example User Request

```text
Plan a 7-day trip to the USA from India under ₹2 lakh.

Include:
- Flights
- Hotels
- Weather
- Sightseeing
- Daily itinerary
- Estimated budget
```

TripMate processes the request through specialized agents and returns a structured travel plan.

---

# 🧠 Core Architecture

```text
                         ┌──────────────────────┐
                         │        USER          │
                         │  Natural Language    │
                         │   Travel Request      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       FastAPI        │
                         │      REST API        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      LangGraph       │
                         │     StateGraph       │
                         │  Workflow Orchestration│
                         └──────────┬───────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
             ▼                      ▼                      ▼
    ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
    │  Flight Agent  │     │   Hotel Agent  │     │ Weather Agent  │
    │                │     │                │     │                │
    │ AviationStack  │     │   Tavily MCP   │     │  Custom MCP    │
    │      MCP       │     │   Web Search   │     │  OpenWeather   │
    └───────┬────────┘     └───────┬────────┘     └───────┬────────┘
            │                      │                      │
            └──────────────────────┼──────────────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │   Itinerary Agent    │
                         │      Groq LLM        │
                         │  Context Integration │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Final Agent      │
                         │      Groq LLM        │
                         │ Structured Response  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Final Travel Plan  │
                         │                      │
                         │ Flights              │
                         │ Hotels               │
                         │ Weather              │
                         │ Itinerary            │
                         │ Budget               │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     PostgreSQL       │
                         │ LangGraph Checkpoint │
                         │    Thread State      │
                         └──────────────────────┘
```

---

# 🤖 Multi-Agent Workflow

TripMate uses a sequential LangGraph workflow where each specialized agent performs a specific responsibility.

```text
START
  │
  ▼
┌─────────────────┐
│  Flight Agent   │
│                 │
│ AviationStack   │
│ Airport/Airline │
│ Information     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Hotel Agent   │
│                 │
│    Tavily MCP   │
│   Web Search    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Weather Agent   │
│                 │
│   Custom MCP    │
│  OpenWeather    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Itinerary Agent │
│                 │
│    Groq LLM     │
│ Context + Plan  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Agent    │
│                 │
│    Groq LLM     │
│ Final Formatting│
└────────┬────────┘
         │
         ▼
        END
```

### Agent Responsibilities

| Agent               | Responsibility                       | Technology               |
| ------------------- | ------------------------------------ | ------------------------ |
| ✈️ Flight Agent     | Airport, airline and flight guidance | AviationStack MCP + Groq |
| 🏨 Hotel Agent      | Hotel discovery and recommendations  | Tavily MCP               |
| 🌤️ Weather Agent   | Current weather and forecast         | Custom MCP + OpenWeather |
| 🗓️ Itinerary Agent | Generates day-by-day plan            | Groq LLM                 |
| 🎯 Final Agent      | Combines and formats final response  | Groq LLM                 |

---

# 🔌 Model Context Protocol (MCP)

One of the key engineering aspects of TripMate is its use of **Model Context Protocol (MCP)** for external tool integration.

Instead of tightly coupling every external API directly with the application, MCP provides a standardized tool interface.

```text
                         TripMate AI
                              │
                              ▼
                    MultiServerMCPClient
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
        Tavily MCP      AviationStack MCP   Weather MCP
             │                │                │
             ▼                ▼                ▼
        Web Search        Flight Tools      Custom Tools
                                              │
                                              ▼
                                         OpenWeather
```

### MCP Servers

#### 🔎 Tavily MCP

Used for web-based hotel discovery.

```text
User Query
    ↓
Hotel Agent
    ↓
Tavily MCP
    ↓
tavily_search
    ↓
Hotel Information
```

#### ✈️ AviationStack MCP

Provides aviation-related information through MCP tools such as:

```text
list_airports
list_airlines
```

The Flight Agent consumes these tools and uses the retrieved information to generate flight guidance.

#### 🌤️ Custom Weather MCP

TripMate includes a custom MCP server built with FastMCP.

Tools exposed:

```text
get_current_weather(city)
get_forecast(city)
```

Architecture:

```text
Weather Agent
      │
      ▼
Weather MCP Server
      │
      ├── get_current_weather()
      │
      └── get_forecast()
              │
              ▼
        OpenWeather API
```

---

# 🧩 LangGraph State Management

TripMate uses a typed shared state to pass information between agents.

```python
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    weather_results: str
```

This allows every stage of the workflow to access the outputs generated by previous stages.

### Example

```text
User Query
    │
    ▼
Flight Results
    │
    ▼
Hotel Results
    │
    ▼
Weather Results
    │
    ▼
Itinerary
    │
    ▼
Final Response
```

The itinerary agent receives information from all previous agents rather than generating a plan independently.

---

# 🧠 LLM Architecture

TripMate uses **Groq-hosted LLM inference** through LangChain's `ChatGroq` integration.

```text
                    User Request
                         │
                         ▼
                ┌─────────────────┐
                │  Flight Agent   │
                │     Groq LLM    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Itinerary Agent │
                │     Groq LLM    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Final Agent    │
                │     Groq LLM    │
                └────────┬────────┘
                         │
                         ▼
                  Final Response
```

Configured model:

```text
openai/gpt-oss-20b
```

---

# 🌦️ Weather Intelligence

TripMate separates **current weather** from **forecast information**.

### Current Weather

The custom MCP server retrieves:

* City
* Temperature
* Feels-like temperature
* Humidity
* Weather condition
* Wind speed

### Forecast

The forecast tool retrieves upcoming forecast entries including:

* Date/time
* Temperature
* Weather condition

The weather information is then passed to the itinerary agent so the generated plan can include weather-aware recommendations.

---

# 💾 PostgreSQL + LangGraph Checkpointing

TripMate uses PostgreSQL as the persistence layer for LangGraph.

```text
FastAPI
   │
   ▼
LangGraph
   │
   ▼
PostgresSaver
   │
   ▼
PostgreSQL
   │
   └── thread_id
       workflow state
       conversation checkpoints
```

Each request can contain a `thread_id`.

```json
{
    "message": "Plan a 7-day trip to Japan",
    "thread_id": "user_123"
}
```

If a thread ID is not provided, TripMate generates one automatically.

```python
thread_id = f"user_{uuid.uuid4().hex}"
```

This enables conversation/workflow state to be associated with a persistent thread.

---

# ⚡ FastAPI Backend

TripMate exposes a REST API using FastAPI.

### Main Endpoint

```http
POST /api/travel
```

### Request

```json
{
    "message": "Plan a 7-day trip to Japan from India under ₹1.5 lakh",
    "thread_id": "user_123"
}
```

### Response

```json
{
    "success": true,
    "thread_id": "user_123",
    "answer": "...",
    "flight_results": "...",
    "hotel_results": "...",
    "weather_results": "...",
    "itinerary": "...",
    "llm_calls": 5
}
```

### Health Check

```http
GET /health
```

Response:

```json
{
    "status": "ok",
    "message": "AI Travel Planner API is running"
}
```

---

# 🔄 End-to-End Request Flow

```text
User
 │
 │ Natural-language travel request
 ▼
FastAPI
 │
 ▼
run_travel_agent()
 │
 ▼
LangGraph StateGraph
 │
 ├──────────────► Flight Agent
 │                     │
 │                     ▼
 │               AviationStack MCP
 │
 ├──────────────► Hotel Agent
 │                     │
 │                     ▼
 │                  Tavily MCP
 │
 ├──────────────► Weather Agent
 │                     │
 │                     ▼
 │                Weather MCP
 │                     │
 │                     ▼
 │                OpenWeather
 │
 ▼
Itinerary Agent
 │
 ▼
Groq LLM
 │
 ▼
Final Agent
 │
 ▼
Structured Travel Plan
 │
 ▼
PostgreSQL Checkpoint
 │
 ▼
FastAPI JSON Response
```

---

# 🛠️ Technology Stack

## Generative AI

* Python
* Groq LLM
* LangChain
* LangGraph
* Prompt Engineering
* Multi-Agent AI
* Tool Calling
* Model Context Protocol (MCP)

## MCP & External Tools

* Tavily MCP
* AviationStack MCP
* Custom Weather MCP
* OpenWeather API
* `MultiServerMCPClient`
* FastMCP

## Backend

* FastAPI
* Pydantic
* Uvicorn
* Jinja2

## Database

* PostgreSQL
* Psycopg
* LangGraph `PostgresSaver`

## Engineering

* Async MCP tool invocation
* Environment-based configuration
* SSL certificate handling with Certifi
* Error handling
* Health endpoint
* Thread-based state management
* Modular agent architecture

---

# 📁 Project Structure

```text
TripMate-AI-Planner-Using-MCP/
│
├── app.py
│   └── FastAPI application and REST endpoints
│
├── backend.py
│   └── LangGraph workflow and travel agents
│
├── mcp_client.py
│   └── MCP client configuration and tool invocation
│
├── custom_weather_mcp_server.py
│   └── Custom Weather MCP server
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   └── js/
│
├── requirements.txt
│
├── .env.example
│
└── README.md
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/Suraj-Kumar09/TripMate-AI-Planner-Using-MCP.git

cd TripMate-AI-Planner-Using-MCP
```

## 2. Create Environment

Using Conda:

```bash
conda create -n travel python=3.10
conda activate travel
```

Or using Python:

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key

TAVILY_API_KEY=your_tavily_api_key

AVIATIONSTACK_API_KEY=your_aviationstack_api_key

OPENWEATHER_API_KEY=your_openweather_api_key

DATABASE_URL=your_postgresql_connection_string
```

### Required Services

| Service       | Purpose                          |
| ------------- | -------------------------------- |
| Groq          | LLM inference                    |
| Tavily        | Web search                       |
| AviationStack | Aviation information             |
| OpenWeather   | Weather and forecast             |
| PostgreSQL    | LangGraph checkpoint persistence |

> Never commit `.env` or API keys to GitHub.

---

# ▶️ Run the Application

Start the FastAPI server:

```bash
python app.py
```

Or:

```bash
uvicorn app:app --reload
```

Application:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

# 🧪 Example Queries

Try requests such as:

```text
Plan a 7-day trip to Japan from India under ₹1.5 lakh.
Include flights, hotels, weather and sightseeing.
```

```text
Plan a 5-day trip to Dubai from Delhi under ₹80,000.
```

```text
Create a 10-day Europe trip from India covering France,
Switzerland and Italy with a budget of ₹2.5 lakh.
```

```text
Plan a 4-day trip to Singapore for two people.
Include hotels, weather, sightseeing and estimated budget.
```

---

# 📊 Example Output

TripMate generates a structured response containing:

```text
1. Trip Summary
   └── Destination + duration + summary table

2. Flight Information
   └── Airports + airlines + duration + airfare guidance

3. Hotel Suggestions
   └── Hotel recommendations + location + estimated price

4. Weather Information
   └── Current conditions + forecast + travel advice

5. Day-by-Day Itinerary
   └── Morning + Afternoon + Evening

6. Estimated Budget
   └── Flights + Hotels + Food + Transport + Activities

7. Final Recommendations
   └── Booking + transportation + weather + saving tips
```

---

# 🏗️ Deployment Architecture

```text
                         INTERNET
                            │
                            ▼
                     ┌─────────────┐
                     │   FastAPI   │
                     │   Backend   │
                     └──────┬──────┘
                            │
                            ▼
                     ┌─────────────┐
                     │  LangGraph  │
                     │ StateGraph  │
                     └──────┬──────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
      Tavily MCP       Aviation MCP      Weather MCP
          │                 │                 │
          ▼                 ▼                 ▼
      Web Search        Flight Tools      OpenWeather
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
                            ▼
                         Groq LLM
                            │
                            ▼
                    Final Travel Plan
                            │
                            ▼
                       PostgreSQL
                    LangGraph State
```

---

# 🔐 Error Handling & Reliability

The application includes defensive handling around external MCP integrations.

For example, the Flight Agent catches external tool failures:

```python
try:
    airports = asyncio.run(
        aviation_mcp_call("list_airports")
    )
except Exception as e:
    flight_data = f"Flight information unavailable: {str(e)}"
```

MCP servers are also initialized independently.

```text
Tavily MCP
     │
     ├── Independent initialization
     │
AviationStack MCP
     │
     ├── Independent initialization
     │
Weather MCP
     │
     └── Independent initialization
```

This prevents an initialization problem with one MCP server from unnecessarily blocking the others.

---

# 💡 Key Engineering Decisions

### 1. Specialized Agents

Instead of using one large prompt for the complete task, responsibilities are separated:

```text
Flight → Hotel → Weather → Itinerary → Final
```

This makes the workflow easier to understand, debug and extend.

### 2. Shared State

LangGraph `TravelState` provides a structured communication layer between agents.

### 3. MCP-Based Tool Integration

External capabilities are exposed through MCP rather than embedding tool logic directly into every agent.

### 4. Persistent Workflow State

PostgreSQL + `PostgresSaver` provides checkpoint persistence using `thread_id`.

### 5. Structured LLM Output

The Final Agent uses explicit output instructions to consistently generate seven sections and Markdown tables.

### 6. Separation of Concerns

```text
app.py
   ↓
API Layer

backend.py
   ↓
Agent / Workflow Layer

mcp_client.py
   ↓
Tool Integration Layer

custom_weather_mcp_server.py
   ↓
Custom MCP Tool Server

PostgreSQL
   ↓
Persistence Layer
```

---

# 📈 What This Project Demonstrates

```text
                         TripMate AI
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
        Generative AI    Multi-Agent AI       MCP
             │                │                │
             ▼                ▼                ▼
         LangChain        LangGraph       Tool Calling
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
                           FastAPI
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
            External APIs             PostgreSQL
                 │                         │
                 └────────────┬────────────┘
                              ▼
                     End-to-End AI System
```

### Recruiter-Relevant Skills Demonstrated

* ✅ Generative AI application development
* ✅ Multi-agent system design
* ✅ LangGraph workflow orchestration
* ✅ LangChain
* ✅ Model Context Protocol (MCP)
* ✅ Custom MCP server development
* ✅ External API/tool integration
* ✅ LLM-based planning and reasoning
* ✅ Prompt engineering
* ✅ FastAPI REST API development
* ✅ PostgreSQL persistence
* ✅ LangGraph checkpointing
* ✅ Async tool invocation
* ✅ Error handling and modular architecture
* ✅ Environment and SSL configuration
* ✅ Production-oriented project structure

---

# 🔮 Future Improvements

Potential next steps include:

* [ ] Parallel execution of independent agents
* [ ] Human-in-the-loop approval before final booking
* [ ] Flight price comparison across providers
* [ ] Hotel booking API integration
* [ ] Redis caching for repeated searches
* [ ] LangSmith / Langfuse observability
* [ ] LLM evaluation pipeline
* [ ] Token and latency monitoring
* [ ] Docker containerization
* [ ] CI/CD pipeline
* [ ] Authentication and user accounts
* [ ] Conversation history UI
* [ ] Cost optimization and model routing
* [ ] Guardrails for unsafe or invalid travel requests

---

# 👨‍💻 Author

### Suraj Kumar

**B.Tech — Artificial Intelligence & Data Science**

Interested in:

```text
Generative AI
LLMs
AI Agents
MLOps
LLMOps
Machine Learning
Cloud & Deployment
```

### Connect

🔗 GitHub:
https://github.com/Suraj-Kumar09

---

# ⭐ If You Find This Project Useful

If you find **TripMate AI** interesting, consider giving the repository a ⭐ on GitHub.

---

## 📌 Project Summary

> **TripMate AI is a production-oriented multi-agent Generative AI travel planner built with LangGraph, MCP, LangChain, Groq, FastAPI and PostgreSQL. It demonstrates the complete flow from natural-language user requirements to external tool integration, agent orchestration, contextual itinerary generation and persistent workflow state.**

---
