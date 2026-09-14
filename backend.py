import os
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid
import asyncio
import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq

from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)


# =========================================================
# Database Configuration
# =========================================================

def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your "
            "Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


# =========================================================
# Groq LLM
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Please add it to your .env file."
    )

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY
)


# =========================================================
# Travel State
# =========================================================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    weather_results: str


# =========================================================
# Flight Agent
# =========================================================

FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")

    query = state["user_query"]

    try:
        airports = asyncio.run(
            aviation_mcp_call("list_airports")
        )

        airlines = asyncio.run(
            aviation_mcp_call("list_airlines")
        )

        print("\nAIRPORTS:", airports)
        print("\nAIRLINES:", airlines)

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000]
        )

        response = llm.invoke([
            SystemMessage(
                content="You are an expert travel flight planner."
            ),
            HumanMessage(content=prompt)
        ])

        flight_data = response.content

    except Exception as e:
        flight_data = (
            f"Flight information unavailable: {str(e)}"
        )

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendations generated"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================================================
# Hotel Agent
# =========================================================

def hotel_agent(state: TravelState):
    query = (
        f"Best hotels for {state['user_query']}"
    )

    hotel_results = asyncio.run(
        tavily_mcp_search(query)
    )

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information fetched."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================================================
# Weather Agent
# =========================================================

def weather_agent(state: TravelState):
    city = extract_destination(
        state["user_query"]
    )

    weather_data = asyncio.run(
        weather_mcp_search(city)
    )

    forecast_data = asyncio.run(
        forecast_mcp_search(city)
    )

    return {
        "weather_results": f"""
Current Weather:
{weather_data}

Forecast:
{forecast_data}
""",
        "messages": [
            AIMessage(
                content="Weather information fetched"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================================================
# Itinerary Agent
# =========================================================

def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Weather Results:
{state['weather_results']}

Make the itinerary:
- Practical
- Budget-aware
- Easy to follow
- Suitable for the requested number of days
- Based on available flight, hotel and weather information
"""

    response = llm.invoke([
        SystemMessage(
            content="You are an expert travel planner."
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================================================
# Final Response Agent
# =========================================================

def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Weather:
{state['weather_results']}

Itinerary:
{state['itinerary']}


IMPORTANT OUTPUT FORMAT

Create the final answer using EXACTLY these 7 sections:

## 1. Trip Summary

Start with a short 2-3 sentence overview.

Mention:
- Destination
- Number of days
- Starting location
- Main highlights
- Approximate budget if available

After the paragraph, create a Markdown table:

| Day | Highlight | Notes |
|-----|-----------|-------|
| Day 1 | ... | ... |
| Day 2 | ... | ... |
| Day 3 | ... | ... |

Continue the rows according to the number of
days in the user's trip.

Keep this table short and concise.
Detailed activities belong in Section 5.


## 2. Flight Information

Provide:
- Departure airport
- Arrival airport
- Airlines
- Typical flight duration
- Estimated airfare
- Booking advice

If live ticket prices are unavailable,
clearly state that airfare is approximate.

Never present an estimated price as a
live booking price.


## 3. Hotel Suggestions

Provide 3-5 suitable hotel suggestions.

For each hotel include:
- Hotel name
- Location
- Approximate price
- Why it is suitable

Keep recommendations relevant to the user's budget.


## 4. Weather Information

Provide:
- Current weather
- Temperature
- Weather condition
- Forecast
- Weather-based travel advice

Clearly distinguish current weather from forecast.


## 5. Day-by-Day Itinerary

Provide a detailed itinerary for every day.

Use:

### Day 1

**Morning**
- Activity

**Afternoon**
- Activity

**Evening**
- Activity

Continue for every day.


## 6. Estimated Budget

Use this Markdown table:

| Category | Estimated Cost |
|----------|----------------|
| Flights | ... |
| Hotels | ... |
| Food | ... |
| Local Transport | ... |
| Sightseeing | ... |
| Miscellaneous | ... |
| Total | ... |

Keep the total consistent with the user's budget.

If exact prices are unavailable, label them
as approximate estimates.


## 7. Final Recommendations

Provide concise recommendations about:
- Best time to visit
- Booking advice
- Transportation
- Weather preparation
- Important travel tips
- Budget-saving suggestions


GENERAL RULES

1. Do not omit any of the 7 sections.
2. Do not rename the sections.
3. Use Markdown headings.
4. Use Markdown tables where requested.
5. Keep the response professional and easy to read.
6. Do not invent live flight ticket prices.
7. Distinguish approximate prices from live prices.
8. Keep the itinerary consistent with trip duration.
9. Keep the budget realistic.
10. Use supplied flight, hotel and weather information.
11. Avoid unnecessary repetition.
12. Trip Summary must contain both:
    - Short introductory paragraph
    - Day-by-day summary table
13. Detailed activities belong in Section 5.
14. Make the response useful for real travel planning.
"""

    response = llm.invoke([
        SystemMessage(
            content=(
                "You are a professional AI "
                "travel booking assistant."
            )
        ),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================================================
# Build LangGraph
# =========================================================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# =========================================================
# PostgreSQL Checkpointer
# =========================================================

DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(
    checkpointer=checkpointer
)


# =========================================================
# FastAPI Travel Agent Function
# =========================================================

def run_travel_agent(
    user_input: str,
    thread_id: str | None = None
):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=user_input
                )
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get(
            "flight_results", ""
        ),
        "hotel_results": result.get(
            "hotel_results", ""
        ),
        "weather_results": result.get(
            "weather_results", ""
        ),
        "itinerary": result.get(
            "itinerary", ""
        ),
        "llm_calls": result.get(
            "llm_calls", 0
        ),
    }