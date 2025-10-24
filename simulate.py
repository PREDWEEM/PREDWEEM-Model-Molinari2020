# ---------- RUN ----------
if run_btn:
    # ✅ CORREGIDO: usar sow_date en lugar de sow (evita NameError)
    df, sow, preR_window, postR_window, gram_window = simulate(
        nyears, seed_bank0, K, Tb, sim_seed,
        preR_days_before, preR_eff_S1S2, preR_residual,
        postR_days_after, postR_eff_S1S4, postR_residual,
        gram_days_after, gram_eff_S1S3, gram_residual,
        sow_date,  # ← CORREGIDO
        LAI_max, t_lag, t_close, LAI_hc, Cs, Ca,
        p_S1, p_S2, p_S3, p_S4
    )
    st.success(f"Listo: {len(df)} días simulados desde {sow}.")

    # ---------- Helper: bandas de control como shapes ----------
    def control_shapes_from_windows(preR_window, postR_window, gram_window):
        # compactar conjuntos de fechas en rangos contiguos
        def to_ranges(days_set):
            if not days_set:
                return []
            days = sorted(list(days_set))
            ranges = []
            start = prev = days[0]
            for d in days[1:]:
                if d == prev + dt.timedelta(days=1):
                    prev = d
                else:
                    ranges.append((start, prev))
                    start = prev = d
            ranges.append((start, prev))
            return ranges

        shapes = []
        for (x0,x1) in to_ranges(preR_window):
            shapes.append(dict(type="rect", xref="x", yref="paper",
                               x0=pd.to_datetime(x0), x1=pd.to_datetime(x1 + dt.timedelta(days=1)),
                               y0=0, y1=1, fillcolor="rgba(255,0,0,0.10)", line=dict(width=0)))
        for (x0,x1) in to_ranges(postR_window):
            shapes.append(dict(type="rect", xref="x", yref="paper",
                               x0=pd.to_datetime(x0), x1=pd.to_datetime(x1 + dt.timedelta(days=1)),
                               y0=0, y1=1, fillcolor="rgba(0,128,0,0.10)", line=dict(width=0)))
        for (x0,x1) in to_ranges(gram_window):
            shapes.append(dict(type="rect", xref="x", yref="paper",
                               x0=pd.to_datetime(x0), x1=pd.to_datetime(x1 + dt.timedelta(days=1)),
                               y0=0, y1=1, fillcolor="rgba(0,0,255,0.10)", line=dict(width=0)))
        return shapes

    shapes_controls = control_shapes_from_windows(preR_window, postR_window, gram_window)

    # ---------- TABS ----------
    t1, t2, t3, t4 = st.tabs([
        "Densidades S1–S4 + Bandas de control",
        "Supervivencia total y mortalidad",
        "Proporción por estado",
        "Datos / Descargar"
    ])

    # --- Tab 1: Densidades por estado + bandas control
    with t1:
        fig = go.Figure()
        for s in ["W1","W2","W3","W4"]:
            fig.add_trace(go.Scatter(x=df["date"], y=df[s], mode="lines", name=s))
        fig.update_layout(
            title="Densidad por estado (S1–S4) con ventanas de control",
            xaxis_title="Fecha", yaxis_title="pl·m⁻²",
            template="plotly_white",
            shapes=shapes_controls,
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bandas: 🔴 PreR, 🟢 PostR, 🔵 Gram")

    # --- Tab 2: Supervivencia total (Σ S1–S4) y mortalidad acumulada
    with t2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["date"], y=df["W_total_S1S4"], name="Vivos (Σ S1–S4)", mode="lines"))
        fig2.add_trace(go.Scatter(x=df["date"], y=df["Mort_preR_cum"], name="Mort acumulada PreR", mode="lines"))
        fig2.add_trace(go.Scatter(x=df["date"], y=df["Mort_postR_cum"], name="Mort acumulada PostR", mode="lines"))
        fig2.add_trace(go.Scatter(x=df["date"], y=df["Mort_gram_cum"], name="Mort acumulada Gram", mode="lines"))
        fig2.update_layout(
            title="Supervivencia total y mortalidad acumulada por control",
            xaxis_title="Fecha", yaxis_title="pl·m⁻²",
            template="plotly_white",
            shapes=shapes_controls,
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig2, use_container_width=True)

        fig2b = go.Figure()
        fig2b.add_trace(go.Bar(x=df["date"], y=df["Mort_preR"], name="PreR"))
        fig2b.add_trace(go.Bar(x=df["date"], y=df["Mort_postR"], name="PostR"))
        fig2b.add_trace(go.Bar(x=df["date"], y=df["Mort_gram"], name="Gram"))
        fig2b.update_layout(
            barmode="stack",
            title="Mortalidad diaria por control (apilada)",
            xaxis_title="Fecha", yaxis_title="pl·m⁻²·día⁻¹",
            template="plotly_white",
            shapes=shapes_controls,
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig2b, use_container_width=True)

    # --- Tab 3: Proporción por estado fenológico
    with t3:
        fig3 = go.Figure()
        for s in ["W1_prop","W2_prop","W3_prop","W4_prop"]:
            fig3.add_trace(go.Scatter(x=df["date"], y=df[s], mode="lines", name=s.replace("_prop","")))
        fig3.update_layout(
            title="Estructura fenológica (proporción por estado) en la población viva",
            xaxis_title="Fecha", yaxis_title="Proporción (0–1)",
            template="plotly_white",
            shapes=shapes_controls,
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig3, use_container_width=True)

    # --- Tab 4: Datos / Descargar
    with t4:
        st.dataframe(df.tail(100), use_container_width=True)
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Descargar CSV", csv, "weedcrop_controls_visual.csv", "text/csv")

else:
    st.info("Configura parámetros y presioná ▶ Ejecutar simulación.")

