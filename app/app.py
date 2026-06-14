import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
from models_handler import MODEL_CONFIG
from utils import build_future_frame, parse_monitoring_csv, prepare_dataframe

st.set_page_config(page_title="Прогноз температурного режима почвы в Салехарде", layout="wide")

DEFAULT_DATA_FILE = "data/processed/ZK 68, (48-1, 48-air), 27.11.20-15.12.25.csv"


def get_path(relative_path):
    return str((Path(__file__).parent.resolve().parent / relative_path).resolve())


@st.cache_data
def load_data(file_name):
    df = pd.read_csv(get_path(file_name), parse_dates=["Дата"])
    df = prepare_dataframe(df)
    return df


try:
    default_df = load_data(DEFAULT_DATA_FILE)
except Exception as e:
    st.error(f"Не удалось загрузить данные: {e}")
    st.stop()

st.sidebar.header("Параметры прогноза")

model_choice = st.sidebar.selectbox("Выберите модель", list(MODEL_CONFIG.keys()))

uploaded_file = st.sidebar.file_uploader(
    "Загрузите данные мониторинга (CSV)",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        history_df = parse_monitoring_csv(uploaded_file)
        st.sidebar.success(f"Загружено {len(history_df)} дней (до {history_df.index[-1].date()})")
        using_default = False
    except Exception as e:
        st.sidebar.error(f"Ошибка чтения файла: {e}")
        history_df = default_df
        using_default = True
else:
    history_df = default_df
    using_default = True

start_ts = history_df.index[-1] + pd.Timedelta(days=1)

if using_default:
    st.sidebar.info(f"Прогноз с: {start_ts.date()} (данные до 15.12.2025)")
else:
    st.sidebar.info(f"Прогноз с: {start_ts.date()}")

horizon = st.sidebar.slider("Горизонт прогноза (дней)", 1, 1095, 30)

st.title("Прогноз температурного режима грунта в Салехарде")

if st.sidebar.button("Сформировать прогноз"):
    with st.spinner("Модель выполняет расчет..."):
        try:
            config = MODEL_CONFIG[model_choice]

            min_history = config.get("min_history", 1)
            if len(history_df) < min_history:
                st.error(
                    f"Недостаточно данных: модель '{model_choice}' требует минимум "
                    f"{min_history} дней, загружено {len(history_df)}."
                )
                st.stop()

            preprocess = config["preprocess"]
            history_processed = preprocess(history_df)

            if "path" in config:
                forecaster = config["wrapper"](get_path(config["path"]))
            else:
                forecaster = config["wrapper"]()
            model_training_end = getattr(forecaster, "model_training_end", None)
            if model_training_end is not None:
                full_future_start = model_training_end + pd.Timedelta(days=1)
                skip_steps = max(0, (start_ts - model_training_end).days - 1)
            else:
                full_future_start = start_ts
                skip_steps = 0

            total_steps = skip_steps + horizon
            full_future_df = build_future_frame(full_future_start, total_steps)
            full_future_processed = preprocess(full_future_df)
            if hasattr(forecaster, "target_cols"):
                missing = [c for c in forecaster.target_cols if c not in history_processed.columns]
                if missing:
                    available = [c for c in forecaster.target_cols if c in history_processed.columns]
                    st.error(
                        f"Модель **{model_choice}** требует глубины: `{forecaster.target_cols}`.\n\n"
                        f"В вашем файле из них доступны только: `{available}`.\n\n"
                        f"Попробуйте модели **ARIMA**, **Exponential Smoothing** или **Linear Regression** — "
                        f"они работают с любым набором глубин."
                    )
                    st.stop()

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

        except Exception as e:
            st.error(f"Ошибка при расчете: {e}")

if "forecast_data" in st.session_state:
    prediction = st.session_state["forecast_data"]
    model_name = st.session_state["current_model"]
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