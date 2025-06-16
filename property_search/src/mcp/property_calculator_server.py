import random
from mcp.server.fastmcp import FastMCP
import os

mcp = FastMCP("property_calculator_server")

# --- Tool Implementations ---
@mcp.tool()
def affordability_calculator(
    annual_income,
    monthly_debts=0,
    down_payment=0,
    loan_term_years=30,
):
    """
    Use this tool when a user asks, "How much house can I afford?" or similar questions about affordability based on their finances.

    Estimate the maximum home price a buyer can afford based on their income, debts, and down payment.

    Parameters:
    - annual_income (float): The buyer's total gross annual income before taxes.
    - monthly_debts (float, optional): Total monthly debt obligations (loans, credit cards, etc.). Default is 0.
    - down_payment (float, optional): The amount of cash available for a down payment. Default is 0.
    - loan_term_years (int, optional): The desired mortgage term in years (usually 15 or 30). Default is 30.

    Returns:
    - dict: Contains the estimated maximum affordable home price, monthly budget, and key financial factors.
    """
    annual_income = float(annual_income)
    monthly_debts = float(monthly_debts)
    down_payment = float(down_payment)
    loan_term_years = int(loan_term_years)
    
    # Randomized defaults
    dti_ratio = random.uniform(0.1, 0.2)
    interest_rate = random.uniform(5.0, 7.5)
    property_tax_percent = random.uniform(0.2, 1.2)
    home_insurance_monthly = random.uniform(150, 300)
    hoa_dues_monthly = random.uniform(30, 100)

    # Income-based monthly budget using DTI
    max_monthly_payment = (annual_income / 12) * dti_ratio - monthly_debts

    def calculate_loan_amount(pmt, rate, term_years):
        monthly_rate = rate / 100 / 12
        total_payments = term_years * 12
        if monthly_rate == 0:
            return pmt * total_payments
        return pmt * ((1 + monthly_rate) ** total_payments - 1) / (monthly_rate * (1 + monthly_rate) ** total_payments)

    # Estimate tax and insurance impact
    def estimate_max_home_price():
        loan_portion = max_monthly_payment - home_insurance_monthly - hoa_dues_monthly
        # Rough estimate, assumes tax and insurance paid monthly and included in DTI
        # Home price = Loan amount + Down payment
        approx_loan_amount = calculate_loan_amount(loan_portion, interest_rate, loan_term_years)
        estimated_home_price = approx_loan_amount + down_payment
        # Property tax adjustment
        monthly_property_tax = (property_tax_percent / 100) * estimated_home_price / 12
        adjusted_payment = loan_portion - monthly_property_tax
        if adjusted_payment <= 0:
            return 0
        final_loan_amount = calculate_loan_amount(adjusted_payment, interest_rate, loan_term_years)
        return round(final_loan_amount + down_payment, 2)

    max_home_price = estimate_max_home_price()

    return {
        "max_home_price": max_home_price,
        "randomized_factors": {
            "property_tax_percent": round(property_tax_percent, 3),
            "home_insurance_monthly": round(home_insurance_monthly, 2),
            "hoa_dues_monthly": round(hoa_dues_monthly, 2)
        },
        "monthly_budget": round(max_monthly_payment, 2)
    }

@mcp.tool()
def mortgage_calculator(
    home_price,
    down_payment,
    interest_rate,
    loan_term_years=30,
):
    """
    Use this tool when a user asks, "What is my mortgage payment?" or similar questions about mortgage payments based on their finances.
    
    Estimate the monthly mortgage payment for a given home price, down payment, interest rate, and loan term.

    Parameters:
    - home_price (float): The total price of the home.
    - down_payment (float): The amount of cash available for a down payment.
    - interest_rate (float): The annual interest rate for the mortgage (e.g., 5.5 for 5.5%).
    - loan_term_years (int, optional): The desired mortgage term in years (usually 15 or 30). Default is 30.

    Returns:
    - dict: Contains the estimated monthly mortgage payment, monthly components, and key financial factors.
    """
    home_price = float(home_price)
    down_payment = float(down_payment)
    interest_rate = float(interest_rate)
    loan_term_years = int(loan_term_years)
    
    # Loan amount
    loan_amount = home_price - down_payment
    monthly_interest_rate = interest_rate / 100 / 12
    total_payments = loan_term_years * 12

    property_tax_annual = random.uniform(1500, 8000)
    home_insurance_annual = random.uniform(1800, 3600)
    hoa_dues_monthly = random.uniform(30, 100)
    pmi_rate = random.uniform(0.005, 0.015)
    include_pmi = False

    # Monthly Principal & Interest
    if monthly_interest_rate == 0:
        monthly_p_and_i = loan_amount / total_payments
    else:
        monthly_p_and_i = loan_amount * (
            monthly_interest_rate * (1 + monthly_interest_rate) ** total_payments
        ) / ((1 + monthly_interest_rate) ** total_payments - 1)

    # Monthly components
    monthly_property_tax = property_tax_annual / 12
    monthly_home_insurance = home_insurance_annual / 12
    monthly_pmi = (loan_amount * pmi_rate / 12) if include_pmi else 0

    # Total monthly payment
    total_monthly_payment = (
        monthly_p_and_i
        + monthly_property_tax
        + monthly_home_insurance
        + hoa_dues_monthly
        + monthly_pmi
    )

    return {
        "monthly_principal_interest": round(monthly_p_and_i, 2),
        "monthly_property_tax": round(monthly_property_tax, 2),
        "monthly_insurance": round(monthly_home_insurance, 2),
        "monthly_hoa_dues": round(hoa_dues_monthly, 2),
        "monthly_pmi": round(monthly_pmi, 2),
        "total_monthly_payment": round(total_monthly_payment, 2)
    }

@mcp.tool()
def buyability_calculator(
    annual_income,
    down_payment,
    location="TX",
    loan_term_years=30
):
    """
    Use this tool when a user asks, "How much house can I buy?" or similar questions about buyability based on their finances.
    
    Recommend an optimal home price range based on both lender qualification and the user's comfort level.

    Parameters:
    - annual_income (float): The buyer's total gross annual income before taxes.
    - down_payment (float): The amount of cash available for a down payment.
    - location (str, optional): The state abbreviation (e.g., 'TX', 'CA') to estimate local property taxes. Default is 'TX'.
    - loan_term_years (int, optional): The desired mortgage term in years (usually 15 or 30). Default is 30.
    - credit_score_range (str, optional): The credit score tier, e.g., 'below_620', '620_679', '680_719', '720_above'. Default is '720_above'.
    - monthly_comfort_spending (float, optional): The maximum monthly payment the user feels comfortable with. If not provided, only qualification is used.

    Returns:
    - dict: Contains lender-qualified maximum price, comfort-based target price, DTI analysis, and search recommendations.
    """
    annual_income = float(annual_income)
    down_payment = float(down_payment)
    loan_term_years = int(loan_term_years)
    
    monthly_comfort_spending = random.uniform(2000, 4500)
    monthly_debt_payments = random.uniform(500, 1500)
    credit_score_range = random.choice(["below_620", "620_679", "680_719", "720_above"])
    
    # Credit score impact on interest rates
    credit_score_rates = {
        "below_620": 7.5,
        "620_679": 6.8,
        "680_719": 6.2,
        "720_above": 5.75
    }
    
    # Location-based property tax estimates (annual %)
    location_tax_rates = {
        "TX": random.uniform(1.8, 2.3),  # Texas has higher property taxes
        "CA": random.uniform(0.7, 1.2),
        "FL": random.uniform(0.8, 1.1),
        "NY": random.uniform(1.2, 1.8)
    }
    
    interest_rate = credit_score_rates.get(credit_score_range, 6.0)
    property_tax_rate = location_tax_rates.get(location, 1.0)
    
    # Standard lending ratios
    max_dti_ratio = 0.43  # 43% max debt-to-income for qualification
    recommended_dti_ratio = 0.36  # 36% recommended for comfort
    
    # Monthly calculations
    monthly_gross_income = annual_income / 12
    
    # QUALIFICATION CALCULATION (What you can pre-qualify for)
    max_monthly_housing_payment = (monthly_gross_income * max_dti_ratio) - monthly_debt_payments
    
    # Estimate other housing costs
    estimated_insurance = random.uniform(150, 400)  # Monthly
    estimated_hoa = random.uniform(0, 200)  # Monthly
    
    # Calculate maximum loan amount for qualification
    def calculate_max_loan_amount(monthly_payment, rate, term_years):
        monthly_rate = rate / 100 / 12
        total_payments = term_years * 12
        
        if monthly_rate == 0:
            return monthly_payment * total_payments
        
        return monthly_payment * ((1 + monthly_rate) ** total_payments - 1) / (
            monthly_rate * (1 + monthly_rate) ** total_payments
        )
    
    # Iterative calculation for max home price (accounting for property tax)
    def calculate_max_qualification():
        # Start with payment minus insurance and HOA
        available_for_loan_and_tax = max_monthly_housing_payment - estimated_insurance - estimated_hoa
        
        # Estimate loan payment (roughly 85% of available, rest for taxes)
        estimated_loan_payment = available_for_loan_and_tax * 0.85
        max_loan = calculate_max_loan_amount(estimated_loan_payment, interest_rate, loan_term_years)
        
        # Calculate total home price
        estimated_home_price = max_loan + down_payment
        
        # Calculate actual property tax
        monthly_property_tax = (estimated_home_price * property_tax_rate / 100) / 12
        
        # Adjust for actual property tax
        adjusted_loan_payment = available_for_loan_and_tax - monthly_property_tax
        
        if adjusted_loan_payment <= 0:
            return 0
            
        final_max_loan = calculate_max_loan_amount(adjusted_loan_payment, interest_rate, loan_term_years)
        return final_max_loan + down_payment
    
    max_qualification_price = calculate_max_qualification()
    
    # COMFORT CALCULATION (What you're comfortable spending)
    comfortable_housing_payment = monthly_comfort_spending
    comfort_available_for_loan = comfortable_housing_payment - estimated_insurance - estimated_hoa
    
    # Calculate comfortable home price
    def calculate_comfort_price():
        estimated_loan_payment = comfort_available_for_loan * 0.85
        comfort_loan = calculate_max_loan_amount(estimated_loan_payment, interest_rate, loan_term_years)
        estimated_price = comfort_loan + down_payment
        
        monthly_tax = (estimated_price * property_tax_rate / 100) / 12
        adjusted_payment = comfort_available_for_loan - monthly_tax
        
        if adjusted_payment <= 0:
            return 0
            
        final_loan = calculate_max_loan_amount(adjusted_payment, interest_rate, loan_term_years)
        return final_loan + down_payment
    
    comfort_target_price = calculate_comfort_price()
    
    # Calculate DTI ratios
    current_dti = (monthly_debt_payments / monthly_gross_income) * 100
    max_payment_dti = (max_monthly_housing_payment + monthly_debt_payments) / monthly_gross_income * 100
    comfort_dti = (comfortable_housing_payment + monthly_debt_payments) / monthly_gross_income * 100
    
    return {
        "buyability_qualification": {
            "max_pre_qualify_amount": round(max_qualification_price, 0),
            "maximum_monthly_payment": round(max_monthly_housing_payment, 0),
            "qualification_dti": round(max_payment_dti, 1)
        },
        "comfort_recommendation": {
            "target_price": round(comfort_target_price, 0),
            "monthly_payment": round(comfortable_housing_payment, 0),
            "comfort_dti": round(comfort_dti, 1)
        },
        "recommendations": {
            "within_comfort_zone": comfort_target_price <= max_qualification_price,
            "suggested_search_range": {
                "min_price": round(comfort_target_price * 0.8, 0),
                "max_price": round(comfort_target_price * 1.1, 0)
            }
        }
    }

def main():
    #uv --port 8000 run my_mcp_server
    mcp.run(transport="sse")

if __name__ == "__main__":
    main()
