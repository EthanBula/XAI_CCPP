import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
app.title = "Planta de Ciclo Combinado • XAI"

CARD_STYLE = {"fontSize": "1.1rem", "borderRadius": "16px", "boxShadow": "0 6px 18px rgba(0,0,0,0.08)"}
IMG_STYLE = {"width": "60%", "maxWidth": "700px", "display": "block", "margin": "0.5rem auto"}
IFRAME_STYLE = {"width": "100%", "height": "420px", "border": "0", "borderRadius": "12px", "boxShadow": "0 6px 18px rgba(0,0,0,0.08)"}
PILL_STYLE = {"fontSize": "1.1rem", "marginRight": "0.4rem"}

def metric_badges():
    return html.Div([
        dbc.Badge(f"R² train: 0.99", color="success", pill=True, style=PILL_STYLE),
        dbc.Badge(f"R² test: 0.96", color="info", pill=True, style=PILL_STYLE),
        dbc.Badge(f"n test: 1914", color="secondary", pill=True, style=PILL_STYLE),
    ], className="my-2")

app.layout = dbc.Container([
    html.Div([
        html.H1("Correlación entre las variables ambientales y la potencia eléctrica producida en una planta de ciclo combinado", className="mt-4 mb-2"),
        html.P("Esta pagina web fue elaborada por Ethan Bula y muestra los resultados de un análisis de correlación modelado con técnicas de XAI para un conjunto de datos extraidos de UCI Machine Learning Repository.")
    ], className="text-center"),

    html.Hr(),

    dbc.Tabs([
        dbc.Tab(label="Contexto", children=[
            html.Br(),
            dbc.Card([
                dbc.CardHeader("Información del conjunto de datos", className="fw-bold"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            dcc.Markdown(
                                """
                                Una planta de ciclo combinado integra dos ciclos termodinámicos:
                                el ciclo Brayton (turbina de gas) y el ciclo Rankine (turbina de vapor). El calor
                                de los gases de exhosto de la turbina de gas se aprovecha en una caldera de recuperación (HRSG)
                                para generar vapor y producir potencia adicional en la turbina de vapor.

                                El conjunto de datos contiene 9568 puntos de datos recopilados de una central eléctrica de ciclo 
                                combinado durante 6 años (2006-2011), cuando la central eléctrica estaba configurada para funcionar
                                a plena carga. La planta de ciclo combinado que proporcionó el conjunto de datos para este estudio,
                                está diseñada con una capacidad de generación nominal de 480 MW, compuesta por dos turbinas de gas ABB
                                13E2 de 160 MW, dos calderas recuperadoras de calor (HRSG) de doble presión y una turbina de vapor ABB
                                de 160 MW.
                                
                                En este proyecto, buscamos estimar la potencia eléctrica (PE) a partir de variables de proceso
                                y ambientales. Las variables de entrada consideradas son:

                                - **AT (Ambient Temperature, °C):** Temperatura del aire a la entrada del compresor.
                                En general, mayor AT reduce la densidad del aire y puede disminuir la potencia.
                                - **V (Exhaust Vacuum, cm Hg):** Vacío del condensador. Un vacío más alto favorece el
                                rendimiento del ciclo de vapor; desviaciones pueden correlacionarse con pérdidas.
                                - **AP (Ambient Pressure, mbar):** Presión atmosférica. Cambios en AP afectan la masa de aire admitida
                                y, por ende, pueden impactar la potencia.
                                - **RH (Relative Humidity, %):** Humedad relativa del aire. Afecta la razón de mezcla y la compresión,
                                y puede modular el desempeño en conjunto con AT.
                                """
                            ), width=8
                        ),
                        dbc.Col(
                        html.Img(
                            src="assets/Ciclo.jpg",  # Ruta de la imagen
                            style={"width": "100%", "border-radius": "8px"}
                        ),
                        width=4
                    )
                    ])
                ])
            ], style=CARD_STYLE),

            html.Br(),

            dbc.Card([
                dbc.CardHeader("Fuentes", className="fw-bold"),
                dbc.CardBody([
                    dcc.Markdown(
                        """
                        URL del conjunto de datos: https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant

                        Artículo introductorio: https://www.sciencedirect.com/science/article/pii/S0142061514000908?via%3Dihub 

                        Código: https://github.com/EthanBula/XAI_CCPP.git 
                        """
                    )
                ])
            ], style=CARD_STYLE)
        ]),

        dbc.Tab(label="Método", children=[
            html.Br(),

            dbc.Card([
                dbc.CardHeader("Modelo Random Forest", className="fw-bold"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            dcc.Markdown(
                                f"""
                                **Modelo:** \`RandomForestRegressor\` con 60 árboles, 
                                profundidad máxima 15 y semillas fijas para reproducibilidad.

                                **Fundamentos de Random Forest:**
                                - Cada árbol de decisión divide el espacio de variables en regiones homogéneas respecto a la variable objetivo.
                                - Usa bootstrap (muestreo con reemplazo) y selecciona subconjuntos aleatorios de variables en cada división, 
                                lo que introduce diversidad y reduce el riesgo de sobreajuste.
                                - Para regresión, se promedian las predicciones individuales de los árboles, logrando un modelo más estable y menos sensible al ruido.
                                - En este contexto, es ideal porque captura interacciones no lineales entre variables ambientales y maneja bien datos atípicos.

                                **Preprocesamiento:**
                                - División 80/20 entre entrenamiento y prueba usando `train_test_split`.
                                - Estandarización con \`StandardScaler\` para obtener media 0 y desviación estándar 1.
                                - Aunque Random Forest no requiere escalado, se aplicó para mantener coherencia metodológica.

                                **Interpretación de indicadores:**
                                - **R² train:** mide qué tan bien el modelo explica la variabilidad de los datos de entrenamiento (1.00 sería ajuste perfecto).
                                - **R² test:** indica el desempeño sobre datos no vistos, clave para evaluar generalización.
                                - **n test:** número de observaciones en el conjunto de prueba.

                                Estos son los resultados de los indicadores para el modelo:
                                """
                            ),
                            width=8
                        ),
                        dbc.Col(
                            html.Img(src="/assets/forest.png", style={**IMG_STYLE, "width": "100%", "borderRadius": "12px"}),
                            width=4
                        )
                    ]),
                    html.Br(),
                    metric_badges()
                ])
            ], style=CARD_STYLE),

            html.Br(),

            dbc.Card([
                dbc.CardHeader("XAI con SHAP", className="fw-bold"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            dcc.Markdown(
                                """
                                **SHAP (SHapley Additive exPlanations):**
                                - Técnica derivada de la teoría de juegos cooperativos, donde cada variable es un jugador que aporta a la predicción final.
                                - Calcula valores de Shapley que indican la contribución marginal media de cada variable considerando todas las combinaciones posibles.

                                **Tipos de interpretaciones utilizadas:**
                                - **Globales:** muestran la importancia promedio de cada variable.
                                - **Locales:** revelan cómo las variables de un caso específico empujan la predicción hacia arriba o abajo respecto al valor esperado.
                                - **Dependence plots:** muestran cómo cambia la contribución SHAP de una variable en función de su valor, 
                                permitiendo detectar umbrales y relaciones no lineales.

                                SHAP aporta transparencia al modelo, mostrando no solo qué tan importante es una variable, sino cómo influye en cada predicción.
                                """
                            ),
                            width=8
                        ),
                        dbc.Col(
                            html.Img(src="/assets/shap.png", style={**IMG_STYLE, "width": "100%", "borderRadius": "12px"}),
                            width=4
                        )
                    ])
                ])
            ], style=CARD_STYLE)
        ]),

        dbc.Tab(label="Resultados", children=[
            html.Br(),
            dbc.Card([
                dbc.CardHeader("Vista Global", className="fw-bold"),
                dbc.CardBody([
                    html.H5("SHAP Summary (dot)", className="mt-2"),
                    html.Img(src="/assets/summary_plot.png", style=IMG_STYLE),
                    html.P(
                        """
                        El color de cada punto indica el valor real de la variable para un punto de datos específico, siendo rojo un valor alto y azul un valor bajo.
                        Si el valor Shap es positivo indica que tiende a aumentar el valor de la potencia eléctrica, mientras que si es negativo indica que tiende a disminuirlo.
                        Su distancia con respecto a 0 indica su importancia relativa en la predicción.
                        La temperatura ambiente sería la variable mas influyente al tener el mayor rango de valores Shap, y al estar los puntos rojos (temperaturas altas) a la izquierda
                        del gráfico con valores negativos significa que temperaturas mas altas disminuyen la potencia eléctrica producida.
                        """,
                        className="text-muted text-center"
                    ),

                    html.Hr(),

                    html.H5("SHAP Summary (bar)", className="mt-2"),
                    html.Img(src="/assets/detailed_summary_plot.png", style=IMG_STYLE),
                    html.P("Este gráfico muestra la magnitud promedio del impacto de cada variable, siendo con diferencia la temperatura ambiente la más importante, seguida del vacío.",
                           className="text-muted text-center"),
                ])
            ], style=CARD_STYLE),

            html.Br(),

            dbc.Card([
                dbc.CardHeader("Relaciones de Dependencia", className="fw-bold"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("AT/RH"),
                            html.Img(src=f"/assets/dependence_plot_AT.png", style={**IMG_STYLE, "width": "55%"}),
                            html.P("Se vuelve a apreciar una correlación negativa entre temperatura ambiente y valor Shap con un comportameinto relativamente lineal, pero la baja humedad relativa para valores de temperatura altos afecta positivamente", className="text-muted text-center")
                        ], md=6),
                        dbc.Col([
                            html.H6("V/AT"),
                            html.Img(src=f"/assets/dependence_plot_V.png", style={**IMG_STYLE, "width": "55%"}),
                            html.P("Teoricamente, un vacio más alto debería favorecer la eficiencia del ciclo de vapor, sin embargo a considerar que una alta temperatura ambiente tiene un efecto contrario en el ciclo de gas, esta ultima opaca las contribuciones del vacio a la generación total de la planta.", className="text-muted text-center")
                        ], md=6),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.H6("AP/V"),
                            html.Img(src=f"/assets/dependence_plot_AP.png", style={**IMG_STYLE, "width": "55%"}),
                            html.P("Cuando AP tiene valores cercanos a 0 (que vendría siendo el promedio en datos estandarizados) es cuando el Shap es mayor, coincidiendo con los valores más altos de V.", className="text-muted text-center")
                        ], md=6),
                        dbc.Col([
                            html.H6("RH/AT"),
                            html.Img(src=f"/assets/dependence_plot_RH.png", style={**IMG_STYLE, "width": "55%"}),
                            html.P("Cuando la humedad relativa es baja, esta contribuye positivamente a la predicción de la potencia eléctrica, aún cuando coincide con los valores más altos de temperatura.", className="text-muted text-center")
                        ], md=6),
                    ]),
                ])
            ], style=CARD_STYLE),

            html.Br(),

            dbc.Card([
                dbc.CardHeader("Ejemplo de predicción para una instancia", className="fw-bold"),
                dbc.CardBody([
                    html.P("Caso de prueba: instancia 19."),
                    html.Img(src="/assets/force_plot.png", style=IMG_STYLE),
                    html.P(
                            """
                            Este gráfico muestra cómo el modelo llegó a una predicción específica para una instancia individual.
                            El valor base es la predicción promedio de todas las instancias del conjunto de datos, que es 454.3 MW.
                            Es el punto de partida de la predicción. Las variables que empujan la predicción hacia la izquierda están en azul.
                            En este caso, AT y V contribuyen negativamente. Las variables que empujan la predicción hacia la derecha están en rojo.
                            Aquí, RH es la única variable que contribuye positivamente. La predicción de la potencia final para esta iteración fue
                            de 431.84 MW debido a la fuerte influencia de AT y V, y siendo un poco amortiguada por RH.
                            """,
                            className="text-muted text-center")
                ])
            ], style=CARD_STYLE),

            html.Br(),
        ])
    ], className="mt-2")
], fluid=True)


if __name__ == "__main__":
    app.run()
