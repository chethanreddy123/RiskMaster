from langchain_google_genai import GoogleGenerativeAI
from langchain import PromptTemplate
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import SQLDatabase
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
from pymongo import MongoClient
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import PIL.Image
import io
import google.generativeai as genai
from helper import *
from prompts import *



df = pd.read_csv("Loan_default.csv")
GOOGLE_API_KEY = 'AIzaSyA1fu-ob27CzsJozdr6pHd96t5ziaD87wM'
genai.configure(api_key=GOOGLE_API_KEY)
llm = GoogleGenerativeAI(model="models/gemini-pro",  temperature= 0 , google_api_key=GOOGLE_API_KEY)
model_image = genai.GenerativeModel('gemini-pro-vision')
MongoCli = MongoClient("mongodb+srv://chethan1234:1234@cluster0.uou14qn.mongodb.net/?retryWrites=true&w=majority")
LoanData = MongoCli["RiskMaster"]["LoanData"]
db = SQLDatabase.from_uri("sqlite:///./Loan_default.db")
toolkit = SQLDatabaseToolkit(db=db , llm=llm)

prompt_template = PromptTemplate(
    input_variables=["query"],
    template=template
)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    prefix = PROMPT,
    suffix = Suffix
    
)
agent_pandas = create_pandas_dataframe_agent(
    llm, 
    df, 
    verbose=True,
    return_intermediate_steps=True)



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
        return (llm(
        prompt_template.format(
            query=user_input
        )
    ))
    
@app.post("/store_user_data/")
async def store_user_data(user_data: dict):
    try:
        # Insert user data into MongoDB
        result = LoanData.insert_one(user_data)

        # Return success message
        return {"message": "User data stored successfully!", "inserted_id": str(result.inserted_id)}

    except Exception as e:
        # Return error message if an exception occurs
        raise HTTPException(status_code=500, detail=f"Error storing user data: {str(e)}")
    

@app.get("/top_ten_new_loans")
async def get_top_ten_new_loans():
    try:
        # Retrieve the top ten new loan data from MongoDB in reverse order
        top_ten_loans = list(LoanData.find().sort([("_id", -1)]).limit(10))

        for i in top_ten_loans:
            # Remove the MongoDB object ID from each loan data
            i.pop("_id")
        # Return the top ten new loan data
        return top_ten_loans

    except Exception as e:
        # Return error message if an exception occurs
        raise HTTPException(status_code=500, detail=f"Error retrieving top ten new loans: {str(e)}")


@app.post("/get_user_data/")
async def get_user_data(user_data: dict):
    try:
        # Retrieve the user data from MongoDB
        user_data = LoanData.find_one(user_data)

        # Remove the MongoDB object ID from the user data
        user_data.pop("_id")

        # Return the user data
        return user_data

    except Exception as e:
        # Return error message if an exception occurs
        raise HTTPException(status_code=500, detail=f"Error retrieving user data: {str(e)}")

@app.post("/get_pandas_agent_response/")
async def get_pandas_agent_response(query: dict):

    try:
        # Retrieve the response from the pandas agent
        response = handle_query(agent_pandas, query['query'])

        # Return the response
        return response

    except Exception as e:
        # Return error message if an exception occurs
        raise HTTPException(status_code=500, detail=f"Error retrieving pandas agent response: {str(e)}")
    


@app.post("/image_analysis/")
async def image_analysis(file: UploadFile = File(...), text: str = Form(...)):

    print("Image analysis request received")
    # Read image file
    image_content = await file.read()
    image = PIL.Image.open(io.BytesIO(image_content))

    # Process with Google Gemini Vision Pro

    response = model_image.generate_content([text, image], stream=True)
    response.resolve()

    return {
        "response" : str(response.text)
    }


@app.post("/pdf_loan_analysis/")
async def pdf_loan_analysis(file: UploadFile = File(...)):
    # Read PDF file
    contents = await file.read()

    # Extract text from PDF
    extracted_text = extract_text_from_first_page(contents)

    # Process with Google Gemini Pro
    response = llm(
        "Extract the important information from the give below document in very crisp way\n"+extracted_text
    )

    
    return JSONResponse(content={"important_info": response})


