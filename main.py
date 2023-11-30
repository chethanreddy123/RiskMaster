from langchain.llms import GooglePalm
from langchain import PromptTemplate, LLMChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_experimental.sql import SQLDatabaseChain
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain import PromptTemplate
from langchain.utilities import SQLDatabase
from langchain import PromptTemplate, LLMChain
from fastapi import  HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder 


def predict_with_models(sample_input):

    model = joblib.load('XGBClassifier_model.joblib')
    
    data = pd.DataFrame([sample_input])
    print(sample_input)
    
    le = LabelEncoder()
    obj_col = ['HasCoSigner','LoanPurpose','HasDependents', 'HasMortgage','MaritalStatus', 'EmploymentType', 'Education']
    for col in obj_col:
        data[col] = le.fit_transform(data[col])

    data = data.drop(['LoanID'], axis=1)
    
    trained_models = joblib.load('XGBClassifier_model.joblib')
    y_pred = model.predict(data)
    return y_pred

llm = GooglePalm(
    model='models/text-bison-001',
    temperature=0,
    max_output_tokens=1024,
    google_api_key='AIzaSyA1fu-ob27CzsJozdr6pHd96t5ziaD87wM'
)


db = SQLDatabase.from_uri("sqlite:///./Loan_default.db")
toolkit = SQLDatabaseToolkit(db=db , llm=llm)



PROMPT = '''You are an agent designed to interact with a SQL database.

The dataset comprises information related to loan applicants, encompassing various demographic and financial attributes. Here is a detailed description of each field in the dataset:

1. LoanID: A unique identifier assigned to each loan application.

2. Age: The age of the loan applicant, indicating the number of years since birth.

3. Income: The total income of the applicant, representing their financial earnings.

4. LoanAmount: The amount of money requested by the applicant as a loan.

5. CreditScore: A numerical representation of the creditworthiness of the applicant, often used by lenders to assess the risk of default.

6. MonthsEmployed: The duration, in months, for which the applicant has been employed.

7. NumCreditLines: The number of credit lines the applicant currently holds.

8. InterestRate: The rate at which interest is charged on the loan amount.

9. LoanTerm: The duration of the loan in months, indicating the period within which the loan must be repaid.

10. DTIRatio (Debt-to-Income Ratio): The ratio of the applicant's total debt to their total income, providing insight into their financial obligations.

11. Education: The educational qualification of the applicant (e.g., Bachelor's, Master's, High School).

12. EmploymentType: The type of employment status of the applicant (e.g., Full-time, Unemployed).

13. MaritalStatus: The marital status of the applicant (e.g., Married, Divorced, Single).

14. HasMortgage: Indicates whether the applicant has an existing mortgage (Yes/No).

15. HasDependents: Indicates whether the applicant has dependents (Yes/No).

16. LoanPurpose: The purpose for which the loan is requested, providing context to the intended use of funds.

17. HasCoSigner: Indicates whether there is a co-signer for the loan (Yes/No).

18. Default: A binary indicator (0 or 1) representing whether the applicant has defaulted on a loan (1) or not (0).

This dataset is valuable for building predictive models to assess the risk of loan default based on various applicant characteristics. The goal is to leverage this information to create a technology solution, like RiskMaster, that utilizes Large Language Models for accurate default prediction and prevention in the Indian banking sector.


Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
If you get a "no such table" error, rewrite your query by using the table in quotes.
DO NOT use a column name that does not exist in the table.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite a different query and try again.
DO NOT try to execute the query more than three times.
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
If the question does not seem related to the database, just return "I don't know" as the answer.
If you cannot find a way to answer the question, just return the best answer you can find after trying at least three times.
'''


Suffix = '''Begin!
Question: {input}
Thought: I should look at the tables in the database to see what I can query.
{agent_scratchpad}'''




agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    prefix = PROMPT,
    suffix = Suffix
    
)

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query_loans/")
def query_sql(query: dict):
    if "query" not in query:
        raise HTTPException(status_code=400, detail="Query missing in request")
    result = agent_executor.run(query['query'])
    return  result

@app.post("/predict_loan/")
def predict_loan(user_input: dict):
    prediction = predict_with_models(user_input)
    if prediction[0] == 0:
        return "Loan will not default"
    else:
        return "Loan will default"
    
@app.post("/get_insigths_user/")
def get_insigths_user(user_input: dict):
    pass
    
    




