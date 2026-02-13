import numpy as np
import joblib
import gradio as gr

# Load trained model
model = joblib.load("burned_area_model.pkl")

# Month and Day mappings
MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

DAY_MAP = {
    'mon': 1, 'tue': 2, 'wed': 3,
    'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7
}

def predict_burned_area(grid_X, grid_Y, month_str, day_str,
                        FFMC, DMC, DC, ISI,
                        temp, RH, wind, rain):

    month_num = MONTH_MAP.get(month_str.lower())
    day_num = DAY_MAP.get(day_str.lower())

    if month_num is None or day_num is None:
        return "Invalid month or day input"

    X_input = np.array([[grid_X, grid_Y,
                         month_num, day_num,
                         FFMC, DMC, DC, ISI,
                         temp, RH, wind, rain]])

    log_pred = model.predict(X_input)[0]
    burned_area = np.expm1(log_pred)

    return round(burned_area, 2)


demo = gr.Interface(
    fn=predict_burned_area,
    inputs=[
        gr.Number(label="X Grid Coordinate (1-9)", minimum=1, maximum=9),
        gr.Number(label="Y Grid Coordinate (1-9)", minimum=1, maximum=9),
        gr.Dropdown(list(MONTH_MAP.keys()), label="Month", value='mar'),
        gr.Dropdown(list(DAY_MAP.keys()), label="Day of Week", value='fri'),
        gr.Number(label="FFMC"),
        gr.Number(label="DMC"),
        gr.Number(label="DC"),
        gr.Number(label="ISI"),
        gr.Number(label="Temperature (°C)"),
        gr.Number(label="Relative Humidity (%)"),
        gr.Number(label="Wind Speed (km/h)"),
        gr.Number(label="Rain (mm)")
    ],
    outputs=gr.Number(label="Predicted Burned Area (hectares)"),
    title="🔥 Forest Fire Burned Area Prediction",
    description="Predict burned area (hectares) using weather, FWI indices, and grid data."
)

if __name__ == "__main__":
    demo.launch()
