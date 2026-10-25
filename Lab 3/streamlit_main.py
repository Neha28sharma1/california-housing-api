import requests
import streamlit as st

# URL of the FastAPI backend
API_URL = "http://127.0.0.1:8000"

# Set the browser tab title
st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠︎",
    layout="wide",
)

# Display the main application heading
st.title(":red[🏠︎] :violet[CALIFORNIA HOUSING PRICE PREDICTOR]")

# Group all input widgets into one form.
with st.form("housing_form"):
    st.info("HOUSING AND INCOME")

    # Divide the input fields into two columns.
    col1, col2 = st.columns(2)

    with col1:
        med_inc = st.number_input(
            ":violet[**Median income ($10,000 units)**]",
            min_value=0.01,
            max_value=20.0,
            value=4.445,
            step=0.1,
        )
        house_age = st.number_input(
            ":violet[**Median house age (years)**]",
            min_value=0.01,
            max_value=100.0,
            step=1.0,
            value=25.0,
        )
        ave_rooms = st.number_input(
            ":violet[**Average rooms per household**]",
            min_value=0.01,
            max_value=100.0,
            step=1.0,
            value=5.0,
        )
        longitude = st.slider(":violet[**Longitude**]", -125.0, -114.0, value=-122.43, step=0.01)

    with col2:
        population = st.number_input(
            ":violet[**District population**]",
            min_value=1.0,
            max_value=100000.0,
            value=1000.0,
        )
        ave_occup = st.number_input(
            ":violet[**Average occupants per household**]",
            min_value=0.01,
            max_value=1000.0,
            value=3.0,
        )
        ave_bedrms = st.number_input(
                    ":violet[**Average bedrooms per household**]",
                    min_value=0.01,
                    max_value=25.0,
                    value=1.0,
                    step=1.0,
        )
        
        latitude = st.slider(":violet[**Latitude**]", 32.0, 42.0, value=37.77, step=0.01)

    col_predict, col_explain = st.columns(2)
    # Submit the form to request a house price prediction only
    with col_predict:
        predict_clicked = st.form_submit_button(
            "Predict house value",
            type="primary",
            use_container_width=True,
        )
    # Submit the form to request both a prediction and an AI-generated explanation of the result
    with col_explain:
        explain_clicked = st.form_submit_button(
            "Predicting the value and explanation",
            type="primary",
            use_container_width=True,
        )

if predict_clicked or explain_clicked:
     # Collect the user-provided values into a dictionary.
    payload = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
    }
    if predict_clicked:
        endpoint = "/predict"
    else:
        endpoint = "/prediction_explanation"
    # Display a loading indicator while the backend processes the request.
    with st.spinner(
        ":green[Predicting the house value explanation...]"
    ):
        try:
            response = requests.post(
                f"{API_URL}{endpoint}",
                json=payload,
                timeout=120,
            )

         
            if response.ok:
                # Convert the JSON response into a Python dictionary.
                result = response.json()

                st.success("Prediction completed!")

                # Support the response field names used by your API.
                price = result.get(
                    "predicted_price_usd",
                    result.get("predicted_price"),
                )

                if price is None:
                    st.error("The API response did not contain a predicted price.")
                else:
                    # Display the predicted median district house value.
                    st.metric(
                        "Estimated district median value",
                        f"${price:,.2f}",
                    )

                # Show explanation only when requested
                if explain_clicked:
                    explanation = result.get("explanation")
                    st.subheader("AI-generated explanation")

                    # Show the explanation if it exists in the response.
                    if explanation:
                        st.write(explanation)
                    else:
                        st.warning(
                            "The API returned a price but no explanation."
                        )

            else:
                detail = response.json().get(
                    "detail",
                    "The prediction service returned an error.",
                )
                # Display the backend error in the Streamlit interface
                st.error(f"Could not complete the request: {detail}")

        # This exception may occur if FastAPI is stopped,the connection fails, or the request times out.
        except requests.RequestException:
            st.error(
                "Cannot connect to the prediction server. "
                "Make sure FastAPI is running at http://127.0.0.1:8000."
            )
