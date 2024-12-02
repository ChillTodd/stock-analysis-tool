import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

#financial modeling prep API
API_KEY = 'GRdzLuSvM2V14iAWG4DDzzNb2k9wYCM3'

# Hard-coded sector medians because of API limitations
if API_KEY == 'GRdzLuSvM2V14iAWG4DDzzNb2k9wYCM3':
    energyrevmed = -0.8
    energyPEmed = 11.8
    energyebitdamed = -2.4
    matrevmed = -0.94
    matPEmed = 17.1
    matebitdamed = 4.00
    indusrevmed = -4.02
    indusPEmed = 21.26
    indusebitdamed = 6.37
    discPEmed = 15.54
    discebitdamed = 1.22
    discrevmed = 2.02
    staplesPEmed = 17.9
    staplesebitdamed = 5.99
    staplesrevmed = 1.7
    hcPEmed = 19.09
    hcebitdamed = 7.43
    hcrevmed = 7.36
    finPEmed = 13.56
    finrevmed = 4.55
    finebitdamed = 11.26
    techPEmed = 25.27
    techrevmed = 4.48
    techebitdamed = 1.94
    commPEmed = 12.66
    commrevmed = 1.43
    commebitdamed = 0.93
    utilPEmed = 20.12
    utilrevmed = -2.52
    utilebitdamed = 8.29

ML = False

#styling Function
def highlight_cells(val):
    if val > 110:
        return "background-color: lightgreen" 
    elif val < 90:
        return "background-color: lightcoral" 
    else:
        return "background-color: lightyellow" 

# Streamlit App
st.title("Stock Analysis")

# Input for ticker
ticker = st.text_input("Enter a Stock Ticker (e.g., AAPL, MSFT):")


if st.button("Get Stock Data"):
    if ticker:

        # Fetch target company information
        ticker_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={API_KEY}"
        financials_url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=annual&limit=2&apikey={API_KEY}"
        ratios_url = f'https://financialmodelingprep.com/api/v3/ratios/{ticker}?apikey={API_KEY}'
        chart_url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={API_KEY}'
        ticker_response = requests.get(ticker_url)
        financials_response = requests.get(financials_url)
        ratios_response = requests.get(ratios_url)
        chart_response = requests.get(chart_url)


        if ticker_response.status_code == 200 and financials_response.status_code == 200:
            
            ticker_data = ticker_response.json()
            financials_data = financials_response.json()
            ratios_data = ratios_response.json()
            chart_data = chart_response.json()

            # Save relevant information
            company_name = ticker_data[0].get('companyName')
            sector = ticker_data[0].get('sector')
            pe0 = ratios_data[0].get('priceEarningsRatio')
            revenue0 = financials_data[0].get('revenue')
            revenue1 = financials_data[1].get('revenue')
            ebitda0 = financials_data[0].get('ebitda')
            ebitda1 = financials_data[1].get('ebitda')
            historical_data = chart_data["historical"]
            dates = [item["date"] for item in historical_data]
            close_prices = [item["close"] for item in historical_data]

            #caluclating growth
            revg = ((revenue0-revenue1) / revenue1) * 100
            ebitdag = ((ebitda0-ebitda1) / ebitda1) * 100

            # display information
            st.markdown(f"## {company_name}")
            st.markdown(f"**Sector:** {sector}")      

            #determine which median to use
            if sector == "Energy":
                secPEmed = energyPEmed
                secRevmed = energyrevmed
                secEBITDAmed = energy
            elif sector == "Basic Materials":
                secPEmed = matPEmed
                secRevmed = matrevmed
                secEBITDAmed = matebitdamed
            elif sector == "Industrials":
                secPEmed = indusPEmed
                secRevmed = indusrevmed
                secEBITDAmed = indusebitdamed
            elif sector == "Consumer Cyclical":
                secPEmed = discPEmed
                secRevmed = discrevmed
                secEBITDAmed = discebitdamed
            elif sector == "Consumer Defensive":
                secPEmed = staplesPEmed
                secRevmed = staplesrevmed
                secEBITDAmed = staplesebitdamed
            elif sector == "Healthcare":
                secPEmed = hcPEmed
                secRevmed = hcrevmed
                secEBITDAmed = hcebitdamed
            elif sector == "Financial Services`":
                secPEmed = finPEmed
                secRevmed = finrevmed
                secEBITDAmed = finebitdamed
            elif sector == "Technology":
                secPEmed = techPEmed
                secRevmed = techrevmed
                secEBITDAmed = techebitdamed
            elif sector == "Communication Services":
                secPEmed = commPEmed
                secRevmed = commrevmed
                secEBITDAmed = commebitdamed
            elif sector == "Utilities":
                secPEmed = utilPEmed
                secRevmed = utilrevmed
                secEBITDAmed = utilebitdamed
            else:
                secPEmed = "N/A"
                secRevmed = "N/A"
                secEBITDAmed = "N/A"

            #calculate differences
            PEDiff = (pe0 / secPEmed) * 100
            RevenueDiff = (revg / secRevmed) * 100
            EBITDADiff = (ebitdag / secEBITDAmed) * 100

            # data table
            table_data = {
                "Metric": [
                    "P/E Ratio",
                    "Revenue",
                    "EBITDA",
                ],
                "Growth %": [
                    pe0,
                    revg,
                    ebitdag,
                ],
                "Sector Median": [
                    secPEmed,
                    secRevmed,
                    secEBITDAmed,
                ],
                "%"+" of sector": [
                    PEDiff,
                    RevenueDiff,
                    EBITDADiff,
                ]
            }

            # convert to dataframe
            df = pd.DataFrame(table_data)
            styled_df = df.style.applymap(highlight_cells, subset=["%"+" of sector"])

            # dipslay table
            st.dataframe(styled_df)

            #Chart Dataframe
            df_chart = pd.DataFrame({
                "Date" : dates,
                "Close Price": close_prices
            })

            df_chart["Date"] = pd.to_datetime(df_chart["Date"])
            df_chart.sort_values(by="Date", inplace=True)
            df_chart.set_index("Date", inplace=True)

            st.line_chart(df_chart)
            ML = True
    else:
        st.error("Error")


if ML == True:
    # machine learning regresion

    if 'Close Price' in df_chart.columns:
        future_3_months = df_chart['Close Price'].shift(-60)
        df_chart['3-Month Price Change %'] = ((future_3_months - df_chart['Close Price']) / df_chart['Close Price']) * 100
        df_chart.dropna(subset=['3-Month Price Change %'], inplace=True)
    else:
        st.error("The column 'Close Price' is missing from the dataset.")

    financial_features = {
        "PEDiff": [PEDiff],
        "RevenueDiff": [RevenueDiff],
        "EBITDADiff": [EBITDADiff],
    }

    financial_broadcast = np.tile(
        np.array(list(financial_features.values())).T, 
        (len(df_chart['3-Month Price Change %']), 1)
    )   

    df_chart['Price Change %'] = df_chart['Close Price'].pct_change() * 100
    df_chart['7-Day Rolling Avg'] = df_chart['Close Price'].rolling(window=5).mean()
    df_chart['30-Day Rolling Avg'] = df_chart['Close Price'].rolling(window=20).mean()
    df_chart['Volatility (30-Day Std)'] = df_chart['Close Price'].rolling(window=20).std()
    df_chart.dropna(inplace=True)

    price_features = {
        "Price Change %": df_chart['Price Change %'].values,
        "7-Day Rolling Avg": df_chart['7-Day Rolling Avg'].values,
        "30-Day Rolling Avg": df_chart['30-Day Rolling Avg'].values,
        "Volatility": df_chart['Volatility (30-Day Std)'].values,
    }


    price_array = np.column_stack(list(price_features.values()))

    min_length = min(financial_broadcast.shape[0], price_array.shape[0])

    financial_broadcast = financial_broadcast[:min_length]
    price_array = price_array[:min_length]

    X = np.hstack([financial_broadcast, price_array])
    X = np.hstack([financial_broadcast, price_array])

    y = df_chart['3-Month Price Change %'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.subheader("Model Evaluation")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")

    current_price = df_chart['Close Price'].iloc[-1]  
    predicted_change = model.predict(X[-1].reshape(1, -1))[0]  

    predicted_price = current_price * (1 + predicted_change / 100)
    lower_bound = predicted_price * 0.95 
    upper_bound = predicted_price * 1.05  
    st.subheader("Predicted Price Range in 3 Months")
    st.write(f"Predicted Price: ${predicted_price:.2f}")
    st.write(f"Confidence Range: ${lower_bound:.2f} - ${upper_bound:.2f}")