import os
from openai import AzureOpenAI
import json
from pydantic import BaseModel
import inspect
from typing import Optional
import pandas as pd
import pandasql as ps
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


client = AzureOpenAI(
  azure_endpoint = "https://rbro-openai-hackatlon.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview", # insert the provided endpoint here 
  api_key=OPENAI_API_KEY, # insert the provided api key here  
  api_version="2024-08-01-preview"
)

# # Customer Service Routine

# system_message = (
#     "You are a financial analysis manager and consultant for Raiffeisen Bank, specializing in stock insights and company performance."
#     "When assisting users, follow this routine:"
#     "1. Begin by verifying that the user's question pertains to a specific company. If not, request clarification.\n"
#     "2. Confirm you understand their query. If unclear, ask follow-up questions."
#     "3. Provide a thorough analysis, drawing on recent stock price trends, market shifts, and relevant metrics.\n"
#     " - If you believe a colleague can provide additional insight, feel free to forward the query to them.\n"
#     "4. If providing a report, ask if the user would like to download it or focus on specific metrics."
#     "5. Engage in continued dialogue if it seems beneficial, or conclude the conversation if the user is satisfied.\n"
#     "6. Handle any misunderstandings by requesting clarification.\n"
#     "As a consultant, address all potential scenarios in a clear and insightful manner, offering well-rounded financial guidance."
# )

# global df
df = pd.read_csv('updated.csv')
columns = df.columns


def function_to_schema(func) -> dict:
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }

class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: str = "You are a helpful Agent"
    tools: list = []

def execute_tool_call(tool_call, tools, agent_name):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"{agent_name}:", f"{name}({args})")

    return tools[name](**args)  # call corresponding function with provided arguments

def execute_refund(item_name):
    return "success"

def run_sql_querry(querry):
    # global df
    df = pd.read_csv('updated.csv')
    print(df)
	# run the querry on the dataframe
    result = ps.sqldf(querry, locals())
    print(result)
    return result.to_string()


class Response(BaseModel):
    agent: Optional[Agent]
    messages: list
    
def run_full_turn(agent, messages):

    current_agent = agent
    num_init_messages = len(messages)
    messages = messages.copy()

    while True:

        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in current_agent.tools]
        tools = {tool.__name__: tool for tool in current_agent.tools}

        # === 1. get openai completion ===
        for message in messages:
            print(message)
        print("************************")
        response = client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": current_agent.instructions}]
            + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)

        if message.content:  # print agent response
            print(f"{current_agent.name}:", message.content)

        if not message.tool_calls:  # if finished handling tool calls, break
            break

        # === 2. handle tool calls ===

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools, current_agent.name)

            if type(result) is Agent:  # if agent transfer, update current agent
                current_agent = result
                result = (
                    f"Transfered to {current_agent.name}. Adopt persona immediately."
                )

            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)

    # ==== 3. return last agent used and new messages =====
    return Response(agent=current_agent, messages=messages[num_init_messages:])

def transfer_to_sql_agent(querry):
	return sql_agent

def transfer_back_to_triage():
	return triage_agent

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        """You are a financial analysis manager and consultant for Raiffeisen Bank, specializing in stock insights and company performance.
        You have at your disposition a SQL AGENT, a database with the following columns: address1, city,state,zip,country,phone,website,industry,industryKey,industryDisp,sector,sectorKey,sectorDisp,longBusinessSummary,fullTimeEmployees,companyOfficers,auditRisk,boardRisk,compensationRisk,shareHolderRightsRisk,overallRisk,governanceEpochDate,compensationAsOfEpochDate,irWebsite,maxAge,priceHint,previousClose,open,dayLow,dayHigh,regularMarketPreviousClose,regularMarketOpen,regularMarketDayLow,regularMarketDayHigh,dividendRate,dividendYield,exDividendDate,payoutRatio,fiveYearAvgDividendYield,beta,trailingPE,forwardPE,volume.
        When assisting users, follow this routine:
        1. If the user's question is not related to finance, tell him it is not your task and wait for another task.
        2. If the user's question want to retrieve data from the database, use the SQL AGENT to run the querry then return the result.
        3. If the user asks for your opinion or a report, provide a thorough analysis, drawing on recent stock price trends, market shifts, and relevant metrics and use the SQL AGENT to retrieve data for your analysis
        4. After both steps, take the data and generate a beautiful report for the user,including a detailed description for each company about which you where asked."""
    ),
    tools=[transfer_to_sql_agent]
)

def select_most_relevant_columns(relevant_columns):
    sql_params["relevant_columns"] = relevant_columns
    return relevant_columns 

sql_params = {
    "intial_columns": "longName, symbol, sentiment, industry, address1,address2,city,zip,twitter_nr_positive_posts,twitter_nr_negative_posts,twitter_nr_neutral_posts,yfinance_positive_news,yfinance_negative_news,yfinance_neutral_news,reddit_nr_positive_posts,reddit_nr_negative_posts,reddit_nr_neutral_posts,country,phone,fax,website,industryKey,industryDisp,sector,sectorKey,sectorDisp,longBusinessSummary,fullTimeEmployees,companyOfficers,compensationAsOfEpochDate,maxAge,priceHint,previousClose,open,dayLow,dayHigh,regularMarketPreviousClose,regularMarketOpen,regularMarketDayLow,regularMarketDayHigh,dividendRate,dividendYield,exDividendDate,payoutRatio,fiveYearAvgDividendYield,beta,trailingPE,forwardPE,volume,regularMarketVolume,averageVolume,averageVolume10days,averageDailyVolume10Day,bid,ask,marketCap,fiftyTwoWeekLow,fiftyTwoWeekHigh,priceToSalesTrailing12Months,fiftyDayAverage,twoHundredDayAverage,trailingAnnualDividendRate,trailingAnnualDividendYield,currency,enterpriseValue,profitMargins,floatShares,sharesOutstanding,heldPercentInsiders,heldPercentInstitutions,impliedSharesOutstanding,bookValue,priceToBook,lastFiscalYearEnd,nextFiscalYearEnd,mostRecentQuarter,earningsQuarterlyGrowth,netIncomeToCommon,trailingEps,forwardEps,enterpriseToRevenue,enterpriseToEbitda,52WeekChange,SandP52WeekChange,lastDividendValue,lastDividendDate,exchange,quoteType,underlyingSymbol,shortName,firstTradeDateEpochUtc,timeZoneFullName,timeZoneShortName,uuid,messageBoardId,gmtOffSetMilliseconds,currentPrice,targetHighPrice,targetLowPrice,targetMeanPrice,targetMedianPrice,recommendationMean,recommendationKey,numberOfAnalystOpinions,totalCash,totalCashPerShare,ebitda,totalDebt,quickRatio,currentRatio,totalRevenue,debtToEquity,revenuePerShare,returnOnAssets,returnOnEquity,freeCashflow,operatingCashflow,earningsGrowth,revenueGrowth,grossMargins,ebitdaMargins,operatingMargins,financialCurrency,trailingPegRatio,irWebsite,pegRatio,lastSplitFactor,lastSplitDate,auditRisk,boardRisk,compensationRisk,shareHolderRightsRisk,overallRisk,governanceEpochDate,bidSize,askSize,state,sharesShort,sharesShortPriorMonth,sharesShortPreviousMonthDate,dateShortInterest,sharesPercentSharesOut,shortRatio,shortPercentOfFloat,ticker,twitter_nr_positive_posts,twitter_nr_negative_posts,twitter_nr_neutral_posts,yfinance_positive_news,yfinance_negative_news,yfinance_neutral_news,reddit_nr_positive_posts,reddit_nr_negative_posts,reddit_nr_neutral_posts"
}



sql_agent = Agent(
    name="SQL Master",
    instructions=(
        f"You are a SQL MasterMind. \
        When getting a prompt try the following: \
        1. If you don't understand the prompt, ask for clarification. \
        2. If it is not something finance related, tell him it is not your task. \
        3. If you do understand the prompt and it can be processed later do the following: \
        I have a csv file with the following columns: \
        {columns} \
        based on the prompt select the most relevant columns. \
        Make sure to only use the columns name given above, don't make up other column names. \
        After that use the natural language querry and the most relevant columns got earlier to make a SQL querry than use it as an argument for run_sql_querry function and make the selection from variable df'. \
		After that, transfer back to triage agent. Make sure the name of the database interrogated is 'df'."
	),
	tools=[select_most_relevant_columns, run_sql_querry, transfer_back_to_triage],				
)


# agent = triage_agent
# messages = []

# print(f"{agent.name}:", "Hello! My name is William how can I help you today?")

# while True:
#     user = input("User: ")
#     messages.append({"role": "user", "content": user})

#     response = run_full_turn(agent, messages)
#     agent = response.agent
#     messages.extend(response.messages)