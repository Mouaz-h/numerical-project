import dash
from dash import dcc, html, Input, Output
import numpy as np
import plotly.graph_objects as go

# ==========================
# Simulation Functions
# ==========================
def forces(v, m, Cd, A, rho, Crr, g, P_max, F_max):
    F_drag = 0.5 * rho * Cd * A * v**2
    F_roll = m * g * Crr
    F_trac = min(F_max, P_max / max(v, 1e-3))
    return F_trac, F_drag, F_roll

def dvdt(v, m, Cd, A, rho, Crr, g, P_max, F_max):
    F_trac, F_drag, F_roll = forces(v, m, Cd, A, rho, Crr, g, P_max, F_max)
    return (F_trac - F_drag - F_roll) / m

def f_state(y, m, Cd, A, rho, Crr, g, P_max, F_max):
    v = y[0]
    return np.array([dvdt(v, m, Cd, A, rho, Crr, g, P_max, F_max), v])

def euler_step(y, h, *args):
    return y + h * f_state(y, *args)

def heun_step(y, h, *args):
    k1 = f_state(y, *args)
    k2 = f_state(y + h*k1, *args)
    return y + 0.5*h*(k1 + k2)

def rk4_step(y, h, *args):
    k1 = f_state(y, *args)
    k2 = f_state(y + 0.5*h*k1, *args)
    k3 = f_state(y + 0.5*h*k2, *args)
    k4 = f_state(y + h*k3, *args)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate(step_func, y0, t, *args):
    y = np.array(y0, dtype=float)
    sol = np.zeros((len(t), 2))
    sol[0] = y
    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        y = step_func(y, h, *args)
        sol[i] = y
    return sol

# ==========================
# Dash App
# ==========================
app = dash.Dash(__name__)

# ==========================
# Layout
# ==========================
app.layout = html.Div(style={'backgroundColor': '#111111', 'color': 'white', 'font-family':'Arial'}, children=[
    html.H1("🚗 Car Acceleration Simulator", style={'textAlign':'center'}),

    html.Div([
        html.Div([
            html.H3("Simulation Parameters"),
            html.Div([
                html.Label("Car mass (kg)"),
                dcc.Input(id='mass', type='number', value=1400.0, style={'width':'100%'}),
                html.Label("Drag coefficient Cd"),
                dcc.Input(id='Cd', type='number', value=0.32, style={'width':'100%'}),
                html.Label("Frontal area A (m^2)"),
                dcc.Input(id='A', type='number', value=2.2, style={'width':'100%'}),
                html.Label("Air density rho (kg/m^3)"),
                dcc.Input(id='rho', type='number', value=1.225, style={'width':'100%'}),
                html.Label("Rolling resistance Crr"),
                dcc.Input(id='Crr', type='number', value=0.015, style={'width':'100%'}),
                html.Label("Max engine power P_max (W)"),
                dcc.Input(id='P_max', type='number', value=80000.0, style={'width':'100%'}),
                html.Label("Max traction force F_max (N)"),
                dcc.Input(id='F_max', type='number', value=4000.0, style={'width':'100%'}),
                html.Label("Simulation time T (s)"),
                dcc.Input(id='T', type='number', value=20.0, style={'width':'100%'}),
                html.Label("Initial velocity v0 (m/s)"),
                dcc.Input(id='v0', type='number', value=0.0, style={'width':'100%'}),
                html.Label("Initial position x0 (m)"),
                dcc.Input(id='x0', type='number', value=0.0, style={'width':'100%'}),
                html.Label("Reference dt (for RK4)"),
                dcc.Input(id='dt_ref', type='number', value=1e-4, style={'width':'100%'}),
                html.Label("dt list (comma separated)"),
                dcc.Input(id='dt_list', type='text', value="0.5,0.2,0.1,0.05,0.02", style={'width':'100%'}),
                html.Button('Run Simulation', id='run-btn', n_clicks=0, style={'marginTop':'10px', 'width':'100%'})
            ], style={'display':'grid', 'gap':'10px'})
        ], style={'padding':'10px', 'backgroundColor':'#222222', 'borderRadius':'10px', 'width':'350px'}),

        html.Div([
            html.H3("Results"),
            html.Pre(id='text-output', style={
                'color': 'white',
                'backgroundColor':'#222222',
                'padding':'10px',
                'borderRadius':'10px',
                'font-family':'monospace'
            }),  # <-- COMMA added

            html.H3("Velocity Curves"),
            dcc.Graph(id='velocity-plot', style={'height':'400px'}),
            html.H3("Error vs dt (log-log)"),
            dcc.Graph(id='error-plot', style={'height':'400px'})
        ], style={'flex':'1', 'padding':'10px'})
    ], style={'display':'flex', 'gap':'20px'})
])

# ==========================
# Callback
# ==========================
@app.callback(
    Output('velocity-plot', 'figure'),
    Output('error-plot', 'figure'),
    Output('text-output', 'children'),
    Input('run-btn', 'n_clicks'),
    Input('mass', 'value'),
    Input('Cd', 'value'),
    Input('A', 'value'),
    Input('rho', 'value'),
    Input('Crr', 'value'),
    Input('P_max', 'value'),
    Input('F_max', 'value'),
    Input('T', 'value'),
    Input('v0', 'value'),
    Input('x0', 'value'),
    Input('dt_ref', 'value'),
    Input('dt_list', 'value')
)
def update_sim(n_clicks, m, Cd, A, rho, Crr, P_max, F_max, T, v0, x0, dt_ref, dt_list_str):
    g = 9.81
    dt_list = [float(x) for x in dt_list_str.split(",")]
    y0 = [v0, x0]

    t_ref = np.arange(0, T+dt_ref, dt_ref)
    sol_ref = integrate(rk4_step, y0, t_ref, m, Cd, A, rho, Crr, g, P_max, F_max)
    v_ref = sol_ref[:,0]

    methods = [('Euler', euler_step), ('Heun', heun_step), ('RK4', rk4_step)]
    errors = {name: [] for name, _ in methods}

    # Calculate errors
    for dt in dt_list:
        t = np.arange(0, T+dt, dt)
        for name, func in methods:
            sol = integrate(func, y0, t, m, Cd, A, rho, Crr, g, P_max, F_max)
            v_sol = sol[:,0]
            v_ref_at_t = np.interp(t, t_ref, v_ref)
            err = np.sqrt(np.mean((v_sol - v_ref_at_t)**2))
            errors[name].append(err * 1e6)  # μm/s

    # Velocity plot (last dt)
    t = np.arange(0, T+dt_list[-1], dt_list[-1])
    fig_vel = go.Figure()
    for name, func in methods:
        sol = integrate(func, y0, t, m, Cd, A, rho, Crr, g, P_max, F_max)
        fig_vel.add_trace(go.Scatter(x=t, y=sol[:,0], mode='lines', name=name))
    fig_vel.add_trace(go.Scatter(x=t_ref[::200], y=v_ref[::200], mode='lines', name='RK4 ref', line=dict(dash='dash')))
    fig_vel.update_layout(
        xaxis_title="Time (s)", yaxis_title="Velocity (m/s)",
        title=f"Car Speed Comparison (dt={dt_list[-1]})",
        plot_bgcolor='#111111', paper_bgcolor='#111111', font_color='white'
    )

    # Error plot log-log
    fig_err = go.Figure()
    for name in errors:
        fig_err.add_trace(go.Scatter(x=dt_list, y=errors[name], mode='lines+markers', name=name))
    fig_err.update_layout(
        xaxis_title="dt (s)", yaxis_title="L2 Error (μm/s)",
        title="Error vs dt (log-log)", xaxis_type="log", yaxis_type="log",
        plot_bgcolor='#111111', paper_bgcolor='#111111', font_color='white'
    )

    # Text output
    text_lines = ["dt      Euler      Heun      RK4 (log10 L2 error)"]
    for i, dt in enumerate(dt_list):
        vals = [errors[name][i] for name, _ in methods]
        vals_log = [np.log10(v) for v in vals]
        best_idx = np.argmin(vals_log)

        line = f"{dt:<7.3f}"
        for j, val in enumerate(vals_log):
            if j == best_idx:
                line += f"{val:>12.2f}*"
            else:
                line += f"{val:>12.2f}"
        text_lines.append(line)

    # Time to 100 km/h
    v_target = 27.78
    idx = np.argmax(v_ref >= v_target)
    if v_ref[idx] >= v_target:
        text_lines.append(f"\nReference time to reach 100 km/h ≈ {t_ref[idx]:.2f} s")
    else:
        text_lines.append("\nCar did not reach 100 km/h.")

    return fig_vel, fig_err, "\n".join(text_lines)

# ==========================
# Run App
# ==========================
if __name__ == '__main__':
    app.run(debug=True)


