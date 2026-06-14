import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
from models_handler import MODEL_CONFIG
from utils import build_future_frame, prepare_dataframe

st.set_page_config(page_title="Прогноз температурного режима почвы в Салехарде", layout="wide")

def get_path(relative_path):
    return str((Path(__file__).parent.resolve().parent / relative_path).resolve())

@st.cache_data
def load_data(file_name):
    df = pd.read_csv(get_path(file_name), parse_dates=["Дата"])
    df = prepare_dataframe(df)
    return df


try:
    default_df = load_data("data/processed/train_raw, ZK 68, (48-1, 48-air), 27.11.20-31.12.24.csv")
except Exception as e:
    st.error(f"Не удалось загрузить данные: {e}")
    st.stop()

st.sidebar.header("Параметры прогноза")
model_choice = st.sidebar.selectbox(
    "Выберите модель",
    list(MODEL_CONFIG.keys()),
)

start_date = st.sidebar.date_input(
    "Дата начала прогноза",
    value=pd.to_datetime("2026-01-01"),
)

horizon = st.sidebar.slider("Горизонт прогноза (дней)", 1, 365, 30)
st.title("Прогноз температурного режима грунта в Салехарде")
if st.sidebar.button("Сформировать прогноз"):
    with st.spinner("Модель выполняет расчет..."):
        try:
            config = MODEL_CONFIG[model_choice]
            data_file = config.get(
                "data_file",
                "data/processed/train_raw, ZK 68, (48-1, 48-air), 27.11.20-31.12.24.csv",
            )
            df_raw = (
                load_data(data_file)
                if data_file
                != "data/processed/train_raw, ZK 68, (48-1, 48-air), 27.11.20-31.12.24.csv"
                else default_df
            )

            start_ts = pd.Timestamp(start_date)
            history_df = df_raw[df_raw.index < start_ts]
            if history_df.empty:
                st.error("Недостаточно исторических данных до выбранной даты.")
                st.stop()
            preprocess = config["preprocess"]
            history_processed = preprocess(history_df)
            training_end = history_df.index[-1]
            full_future_start = training_end + pd.Timedelta(days=1)
            skip_steps = max(0, (start_ts - training_end).days - 1)
            total_steps = skip_steps + horizon
            full_future_df = build_future_frame(full_future_start, total_steps)
            full_future_processed = preprocess(full_future_df)
            if "path" in config:
                forecaster = config["wrapper"](get_path(config["path"]))
            else:
                forecaster = config["wrapper"]()
            prediction = forecaster.predict(
                history_processed,
                steps=total_steps,
                future_exog=full_future_processed,
            )
            if skip_steps > 0:
                prediction = prediction.iloc[skip_steps:].copy()
            st.session_state["forecast_data"] = prediction
            st.session_state["current_model"] = model_choice
            st.session_state["depth_columns"] = forecaster.depth_columns
            st.session_state["history_df"] = history_df
        except Exception as e:
            st.error(f"Ошибка при расчете: {e}")

if "forecast_data" in st.session_state:
    prediction = st.session_state["forecast_data"]
    model_name = st.session_state["current_model"]
    history_df = st.session_state.get("history_df")
    model_depths = st.session_state.get("depth_columns", prediction.columns.tolist())
    available_cols = [col for col in model_depths if col in prediction.columns]
    st.success(f"Прогноз готов (модель: {model_name})")
    selected_depth = st.selectbox(
        "Выберите глубину для визуализации",
        available_cols,
        key=f"depth_{model_name}",
    )
    if selected_depth:
        fig = px.line(
            prediction[[selected_depth]],
            y=selected_depth,
            title=f"Прогноз: {selected_depth} (Модель: {model_name})",
            labels={"index": "Дата", selected_depth: "Температура (°C)"},
        )
        fig.update_xaxes(dtick="M1", tickformat="%b %Y", tickangle=-45)
        fig.update_traces(line_color="#FF4B4B", line_width=2)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Посмотреть числовые значения"):
            st.dataframe(prediction[[selected_depth]])
else:
    st.info("Настройте параметры в боковой панели и нажмите 'Сформировать прогноз'")
