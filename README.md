
# RiskMaster Project

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

## Chain of Thoughts Integration

RiskMaster incorporates Chain of Thought (CoT) prompting to guide the Large Language Models (LLMs) in thinking step by step. This strategy involves providing the model with a few-shot exemplar outlining the reasoning process, allowing the model to follow a similar chain of thought when responding to prompts. CoT prompting is particularly effective for complex tasks requiring a series of reasoning steps.

### CoT Prompt Example:

**Q:** Assess the risk of loan default for a 56-year-old applicant with an income of ₹85,994, seeking a loan amount of ₹50,587, a credit score of 520, and 80 months of employment history.

**A:** Similar to our approach in assessing Joe's egg count, we start with the given attributes. Considering age, income, loan amount, credit score, and employment history, we can delve into the nuances of potential default risks. Utilizing these parameters, the LLM performs a step-by-step analysis, providing a reasoned prediction.

RiskMaster employs CoT prompting in tasks such as predicting loan default, allowing for more structured and reasoned responses.

### Automatic Chain-of-Thought (Auto-CoT)

RiskMaster leverages Automatic Chain-of-Thought (Auto-CoT) to automate the CoT prompting process. This approach involves question clustering and demonstration sampling stages:

1. **Question Clustering:** Partition questions of a given dataset into clusters.
2. **Demonstration Sampling:** Select a representative question from each cluster and generate its reasoning chain using Zero-Shot-CoT with simple heuristics.

Auto-CoT automates the generation of reasoning chains, eliminating the need for manual crafting of examples and enhancing the efficiency of the system.

## Conclusion

RiskMaster, enriched with CoT prompting, represents a significant advancement in enhancing the reasoning capabilities of Large Language Models. By enabling models to think step by step and explain their reasoning, the project aims to improve performance on complex tasks related to loan default prediction.


