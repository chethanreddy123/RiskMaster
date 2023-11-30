# RiskMaster Project

## Project Overview

RiskMaster is an analytics-based technology solution stack designed to address the challenges faced by the Indian banking sector in predicting and preventing defaults. The project aims to mitigate financial instability in major banks by leveraging Large Language Models (LLMs) for comprehensive analysis and reporting.

## Problem Statement

Over the past 15 months, several prominent Indian banks have encountered substantial financial losses due to a surge in bad debt. The inability to accurately predict and prevent defaults has emerged as a significant challenge, leading to severe financial repercussions for major banks.

## Objectives

1. **Default Prediction & Prevention:** Develop a solution that accurately predicts the likelihood of loan default based on a comprehensive analysis of user data.

2. **Responsible Lending Practices:** Implement a system that ensures responsible lending practices by providing critical insights into potential default risks.

3. **Enhanced Financial Stability:** Improve the overall financial stability of the banking sector by preventing and minimizing losses due to bad debt.

## Methodology

RiskMaster utilizes the formidable capabilities of Large Language Models, specifically Google's PALM LLM, fine-tuned with diverse datasets encompassing various loan policies and regulations. When a loan applicant approaches, the system performs a deep dive analysis using attributes such as default history, income, repayment behavior, and more. The generated reports aid in swift and accurate loan decision-making.

## Solution Scope

The solution provided by RiskMaster covers the following key aspects:

1. **Predictive Analytics:** Employ advanced analytics to predict the likelihood of loan default based on user data.

2. **Comprehensive Analysis:** Conduct a comprehensive analysis considering attributes such as credit score, income, loan amount, and more to discern potential default risks.

3. **User-Friendly Reports:** Utilize the ReportLab module to generate clear, user-friendly reports that facilitate quick and accurate loan decision-making.

## Additional Details

### Dataset

The project utilizes a dataset with the following attributes:

LoanID, Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner, Default

## FastAPI Routes

1. **Query Loans**
   - Method: POST
   - Endpoint: `/query_loans/`
   - Description: Executes a SQL-like query on the loan data using an external agent and returns the result.

2. **Predict Loan Default**
   - Method: POST
   - Endpoint: `/predict_loan/`
   - Description: Predicts whether a given user's loan will default based on user input data.

3. **Get User Insights**
   - Method: POST
   - Endpoint: `/get_insigths_user/`
   - Description: Retrieves insights about a user using Large Language Models (LLMs) based on user input data.

4. **Store User Data**
   - Method: POST
   - Endpoint: `/store_user_data/`
   - Description: Stores user data in the MongoDB database for future analysis.

5. **Top Ten New Loans**
   - Method: GET
   - Endpoint: `/top_ten_new_loans`
   - Description: Retrieves the top ten new loan data from the MongoDB database in reverse order of insertion.

## Conclusion

RiskMaster stands as a powerful solution to the pressing issue of loan defaults in the Indian banking sector. By combining advanced analytics, responsible lending practices, and user-friendly reporting, the project aims to contribute to the enhanced financial stability of major banks and the overall economic landscape.
