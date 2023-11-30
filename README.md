# RiskMaster

RiskMaster is an analytics-based technology solution stack for Default Prediction & Prevention in the Indian Banking Sector. The system employs Large Language Models (LLMs), specifically Google's PALM LLM, fine-tuned with diverse datasets encompassing various loan policies and regulations.

## Problem Statement

In the past 15 months, several prominent Indian banks have experienced substantial financial losses due to a surge in bad debt. The inability to accurately predict and prevent defaults has emerged as a significant challenge, resulting in substantial financial repercussions for major banks.

## Solution Overview

RiskMaster utilizes the formidable capabilities of Large Language Models (LLMs) for comprehensive default risk analysis. When a loan applicant approaches, agents input customer details. Instead of manual scrutiny, the LLMs perform a deep dive analysis using attributes like default history, income, repayment behavior, and more. The system generates clear, user-friendly reports using the ReportLab module, aiding in swift and accurate loan decision-making.

## Dataset

The dataset used for training and analysis includes the following columns:

LoanID, Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner, Default

## FastAPI Routes (main.py)

### Route 1: [Endpoint for Default Prediction]

- **Method**: POST
- **Endpoint**: `/predict_default`
- **Description**: Accepts user data and predicts whether the applicant is likely to default on the loan.

### Route 2: [Endpoint for Detailed Analysis]

- **Method**: POST
- **Endpoint**: `/detailed_analysis`
- **Description**: Provides a detailed analysis report for the given user data, including insights into the number of installments the applicant might skip.

## Getting Started

Follow these steps to set up and run RiskMaster:

1. Clone the repository: `git clone https://github.com/your_username/RiskMaster.git`
2. Navigate to the project directory: `cd RiskMaster`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the FastAPI application: `uvicorn main:app --reload`

## Usage

1. Make a POST request to the `/predict_default` endpoint with the user data to get default predictions.
2. Make a POST request to the `/detailed_analysis` endpoint with the user data to receive a detailed analysis report.

## Contributors

- [Your Name]
- [Contributor 2 Name]

## License

This project is licensed under the [Your License] License - see the [LICENSE.md](LICENSE.md) file for details.

