
# ============================================
# Simulación EEG para análisis del sueño
# Rol: Ingeniero Biomédico
# Diseñado para ejecutarse en Google Colab
# ============================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# Configuración general
# -----------------------------
FS = 128                    # frecuencia de muestreo [Hz]
EPOCH_SEC = 30              # duración por época [s]
N_EPOCHS = 20               # cantidad de épocas simuladas
SEED = 42                   # reproducibilidad

rng = np.random.default_rng(SEED)

# Secuencia típica simplificada de sueño
SLEEP_SEQUENCE = [
    "Awake", "N1", "N2", "N2", "N3",
    "N3", "N2", "REM", "N2", "N3",
    "N2", "REM", "N2", "N1", "N2",
    "N3", "REM", "N2", "REM", "Awake"
]

# -----------------------------
# Modelos sintéticos por etapa
# -----------------------------
def simulate_stage_signal(stage: str, duration_s: int, fs: int, rng: np.random.Generator):
    t = np.arange(0, duration_s, 1/fs)
    n = len(t)

    # ruido basal
    noise = rng.normal(0, 5, n)

    if stage == "Awake":
        # alfa + beta, amplitud moderada
        sig = (
            20*np.sin(2*np.pi*10*t) +   # alpha
            8*np.sin(2*np.pi*18*t) +    # beta
            noise
        )
    elif stage == "N1":
        # theta
        sig = (
            25*np.sin(2*np.pi*6*t) +
            6*np.sin(2*np.pi*3*t) +
            noise
        )
    elif stage == "N2":
        # theta + husos de sueño + complejos K
        sig = 20*np.sin(2*np.pi*5*t) + noise

        # husos: ráfagas 12-14 Hz
        for _ in range(3):
            center = rng.uniform(3, duration_s-3)
            spindle = np.exp(-0.5*((t-center)/0.5)**2) * 18*np.sin(2*np.pi*13*t)
            sig += spindle

        # complejos K: pulsos lentos
        for _ in range(2):
            center = rng.uniform(4, duration_s-4)
            k_complex = -40*np.exp(-((t-center)/0.15)**2) + 25*np.exp(-((t-(center+0.2))/0.25)**2)
            sig += k_complex
    elif stage == "N3":
        # delta predominante, alta amplitud
        sig = (
            60*np.sin(2*np.pi*1.5*t) +
            25*np.sin(2*np.pi*0.8*t) +
            noise
        )
    elif stage == "REM":
        # baja amplitud, frecuencia mixta similar a vigilia
        sig = (
            12*np.sin(2*np.pi*7*t) +
            10*np.sin(2*np.pi*16*t) +
            noise
        )
    else:
        sig = noise

    # deriva lenta de base
    baseline = 5*np.sin(2*np.pi*0.15*t)
    sig = sig + baseline
    return t, sig


def stage_band_powers(stage: str, rng: np.random.Generator):
    """
    Valores sintéticos relativos para bandas:
    delta, theta, alpha, beta
    """
    if stage == "Awake":
        vals = [0.10, 0.20, 0.40, 0.30]
    elif stage == "N1":
        vals = [0.15, 0.50, 0.20, 0.15]
    elif stage == "N2":
        vals = [0.25, 0.40, 0.15, 0.20]
    elif stage == "N3":
        vals = [0.70, 0.20, 0.05, 0.05]
    elif stage == "REM":
        vals = [0.10, 0.35, 0.15, 0.40]
    else:
        vals = [0.25, 0.25, 0.25, 0.25]

    vals = np.array(vals) + rng.normal(0, 0.02, 4)
    vals = np.clip(vals, 0.01, None)
    vals = vals / vals.sum()
    return vals


def build_full_signal(sequence, epoch_sec=30, fs=128, rng=None):
    t_all = []
    x_all = []
    stage_per_sample = []
    epoch_summary = []

    current_time = 0.0
    for i, stage in enumerate(sequence):
        t, x = simulate_stage_signal(stage, epoch_sec, fs, rng)
        t_shifted = t + current_time

        t_all.append(t_shifted)
        x_all.append(x)
        stage_per_sample.extend([stage] * len(t))

        band_vals = stage_band_powers(stage, rng)
        epoch_summary.append({
            "epoch": i + 1,
            "stage": stage,
            "start_s": current_time,
            "end_s": current_time + epoch_sec,
            "delta": band_vals[0],
            "theta": band_vals[1],
            "alpha": band_vals[2],
            "beta": band_vals[3],
            "rms_uV": float(np.sqrt(np.mean(x**2))),
            "peak_uV": float(np.max(np.abs(x)))
        })

        current_time += epoch_sec

    return (
        np.concatenate(t_all),
        np.concatenate(x_all),
        np.array(stage_per_sample),
        pd.DataFrame(epoch_summary)
    )


def create_sleep_dashboard(t, x, summary_df, fs=128):
    stage_numeric = {
        "Awake": 4,
        "REM": 3,
        "N1": 2,
        "N2": 1,
        "N3": 0
    }

    hypnogram_y = [stage_numeric[s] for s in summary_df["stage"]]
    stage_labels = ["N3", "N2", "N1", "REM", "Awake"]

    # ventana para mostrar EEG detalle
    detail_sec = 60
    idx = t <= detail_sec
    t_detail = t[idx]
    x_detail = x[idx]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.07,
        row_heights=[0.35, 0.20, 0.22, 0.23],
        subplot_titles=(
            "EEG simulado (primeros 60 s)",
            "Hipnograma",
            "Potencia relativa por bandas",
            "Indicadores por época"
        )
    )

    # 1) EEG
    fig.add_trace(
        go.Scatter(
            x=t_detail,
            y=x_detail,
            mode="lines",
            name="EEG"
        ),
        row=1, col=1
    )

    # 2) Hipnograma
    fig.add_trace(
        go.Scatter(
            x=summary_df["start_s"] / 60,
            y=hypnogram_y,
            mode="lines+markers",
            line_shape="hv",
            name="Etapa"
        ),
        row=2, col=1
    )

    # 3) Potencia relativa por bandas
    epochs = summary_df["epoch"]
    fig.add_trace(go.Bar(x=epochs, y=summary_df["delta"], name="Delta"), row=3, col=1)
    fig.add_trace(go.Bar(x=epochs, y=summary_df["theta"], name="Theta"), row=3, col=1)
    fig.add_trace(go.Bar(x=epochs, y=summary_df["alpha"], name="Alpha"), row=3, col=1)
    fig.add_trace(go.Bar(x=epochs, y=summary_df["beta"], name="Beta"), row=3, col=1)

    # 4) RMS y pico
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=summary_df["rms_uV"],
            mode="lines+markers",
            name="RMS (uV)"
        ),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=summary_df["peak_uV"],
            mode="lines+markers",
            name="Pico (uV)"
        ),
        row=4, col=1
    )

    fig.update_yaxes(title_text="uV", row=1, col=1)
    fig.update_xaxes(title_text="Tiempo [s]", row=1, col=1)

    fig.update_yaxes(
        title_text="Etapa",
        tickmode="array",
        tickvals=[0,1,2,3,4],
        ticktext=stage_labels,
        row=2, col=1
    )
    fig.update_xaxes(title_text="Tiempo [min]", row=2, col=1)

    fig.update_yaxes(title_text="Potencia relativa", row=3, col=1)
    fig.update_xaxes(title_text="Época", row=3, col=1)

    fig.update_yaxes(title_text="uV", row=4, col=1)
    fig.update_xaxes(title_text="Época", row=4, col=1)

    fig.update_layout(
        height=1100,
        title="Dashboard profesional - Simulación EEG para análisis del sueño",
        barmode="stack",
        template="plotly_dark",
        legend_title="Variables"
    )
    return fig


def create_summary_cards(summary_df):
    total_min = (summary_df["end_s"].max() - summary_df["start_s"].min()) / 60
    stage_counts = summary_df["stage"].value_counts()

    sleep_min = total_min - stage_counts.get("Awake", 0) * EPOCH_SEC / 60
    rem_min = stage_counts.get("REM", 0) * EPOCH_SEC / 60
    n3_min = stage_counts.get("N3", 0) * EPOCH_SEC / 60
    efficiency = 100 * sleep_min / total_min if total_min > 0 else 0

    cards = pd.DataFrame({
        "Indicador": [
            "Tiempo total registrado [min]",
            "Tiempo total de sueño [min]",
            "Sueño REM [min]",
            "Sueño profundo N3 [min]",
            "Eficiencia de sueño [%]",
            "RMS EEG promedio [uV]"
        ],
        "Valor": [
            round(total_min, 1),
            round(sleep_min, 1),
            round(rem_min, 1),
            round(n3_min, 1),
            round(efficiency, 1),
            round(summary_df["rms_uV"].mean(), 1)
        ]
    })
    return cards


# -----------------------------
# Ejecutar simulación
# -----------------------------
t, x, stage_per_sample, summary_df = build_full_signal(
    SLEEP_SEQUENCE,
    epoch_sec=EPOCH_SEC,
    fs=FS,
    rng=rng
)

cards_df = create_summary_cards(summary_df)
fig = create_sleep_dashboard(t, x, summary_df, fs=FS)

print("Resumen clínico-técnico sintético")
print(cards_df.to_string(index=False))

print("\nResumen por época")
print(summary_df[["epoch", "stage", "delta", "theta", "alpha", "beta", "rms_uV", "peak_uV"]].round(3).to_string(index=False))

fig.show()

# Exportación opcional
summary_df.to_csv("sleep_eeg_summary.csv", index=False)
cards_df.to_csv("sleep_eeg_kpis.csv", index=False)

print("\nArchivos generados:")
print("- sleep_eeg_summary.csv")
print("- sleep_eeg_kpis.csv")
