from manim import *
import numpy as np
import random

class SLLHNNFlowchartAnimation(Scene):
    def construct(self):
        # =========================================================================
        # 0. INITIAL CONFIGURATION (Loop and Fade-in)
        # =========================================================================
        self.wait(0.5) 
        
        titulo = Text("Separable Latent Linear Hamiltonian Neural Networks", font_size=28, color=WHITE)
        subtitulo = Text("SLL-HNN Architecture: Physical and Mathematical Flow", font_size=20, color=GRAY).next_to(titulo, DOWN)
        header = VGroup(titulo, subtitulo).to_edge(UP)
        
        self.play(FadeIn(header, shift=DOWN), run_time=1.5)
        self.wait(0.5)
        
        def animate_flow(arrow, particle_color=WHITE, time=0.8):
            particle = Dot(radius=0.06, color=particle_color)
            self.play(MoveAlongPath(particle, arrow), run_time=time, rate_func=linear)
            self.remove(particle)

        # =========================================================================
        # 1. VECTOR INPUTS AND EXPLICIT NORMALIZATION (max(|q|))
        # =========================================================================
        q_input = MathTex(r"\vec{q}", color=BLUE_B).scale(1.5).move_to(LEFT * 6.2 + UP * 1.5)
        p_input = MathTex(r"\vec{p}", color=RED_B).scale(1.5).move_to(LEFT * 6.2 + DOWN * 2.5)
        
        self.play(FadeIn(q_input, shift=RIGHT), FadeIn(p_input, shift=RIGHT))
        self.play(
            q_input.animate.scale(1.2), p_input.animate.scale(1.2),
            rate_func=there_and_back, run_time=0.8
        )
        
        norm_q = MathTex(r"\max(|\vec{q}|)", color=TEAL_A).scale(1.0).next_to(q_input, RIGHT, buff=0.6)
        norm_p = MathTex(r"\max(|\vec{p}|)", color=MAROON_A).scale(1.0).next_to(p_input, RIGHT, buff=0.6)
        self.play(FadeIn(norm_q, shift=UP), FadeIn(norm_p, shift=DOWN))
        
        q_norm_eq = MathTex(r"\vec{\hat{q}}", "=", r"\frac{\vec{q}}{\max(|\vec{q}|)}").move_to(LEFT * 4.8 + UP * 1.5)
        q_norm_eq[0].set_color(TEAL_C)
        q_norm_eq[2].set_color(TEAL_A) 
        
        p_norm_eq = MathTex(r"\vec{\hat{p}}", "=", r"\frac{\vec{p}}{\max(|\vec{p}|)}").move_to(LEFT * 4.8 + DOWN * 2.5)
        p_norm_eq[0].set_color(RED_C)
        p_norm_eq[2].set_color(MAROON_A) 

        self.play(
            ReplacementTransform(VGroup(q_input, norm_q), q_norm_eq),
            ReplacementTransform(VGroup(p_input, norm_p), p_norm_eq),
            run_time=1.2
        )
        self.wait(0.3)

        # =========================================================================
        # 2. LINEAR TRANSFORMATION TO LATENT SPACE
        # =========================================================================
        trans_lineal_q = MathTex(r"\vec{\hat{q}}_{\text{lat}}", "=", r"\vec{\hat{q}}", r"\odot W_q + b_q").scale(0.85).move_to(LEFT * 2.5 + UP * 1.5)
        trans_lineal_q[0].set_color(BLUE_A)
        trans_lineal_q[2].set_color(TEAL_C)
        trans_lineal_q[3].set_color(YELLOW_A)
        
        trans_lineal_p = MathTex(r"\vec{\hat{p}}_{\text{lat}}", "=", r"\vec{\hat{p}}", r"\odot W_p + b_p").scale(0.85).move_to(LEFT * 3.0 + DOWN * 2.5)
        trans_lineal_p[0].set_color(RED_A)
        trans_lineal_p[2].set_color(RED_C)
        trans_lineal_p[3].set_color(YELLOW_A)

        self.play(
            ReplacementTransform(q_norm_eq, trans_lineal_q),
            ReplacementTransform(p_norm_eq, trans_lineal_p),
            run_time=1.2
        )
        self.play(Circumscribe(trans_lineal_q[0], color=BLUE_A), Circumscribe(trans_lineal_p[0], color=RED_A), run_time=1)

        # =========================================================================
        # 3. ENERGY BIFURCATION (V and T)
        # =========================================================================
        # Q BRANCH: Potential Energy (V)
        capas_config = [3, 4, 3, 1] 
        nn_nodos = VGroup()
        origen_nn = ORIGIN + UP * 1.5 + RIGHT * 0.5
        
        for i, num_nodos in enumerate(capas_config):
            capa = VGroup()
            for j in range(num_nodos):
                nodo = Circle(radius=0.1, color=BLUE_E, stroke_width=2).set_fill(BLACK, opacity=1)
                nodo.move_to(origen_nn + RIGHT * i * 0.7 + UP * (j - (num_nodos - 1) / 2) * 0.4)
                capa.add(nodo)
            nn_nodos.add(capa)
            
        conexiones_por_capa = []
        nn_conexiones = VGroup()
        for i in range(len(capas_config) - 1):
            grupo_lineas = VGroup()
            for n_act in nn_nodos[i]:
                for n_sig in nn_nodos[i+1]:
                    linea = Line(n_act.get_center(), n_sig.get_center(), stroke_width=1.5, color=DARK_GRAY)
                    grupo_lineas.add(linea)
            conexiones_por_capa.append(grupo_lineas)
            nn_conexiones.add(*grupo_lineas)
                    
        nn_vgraph = VGroup(nn_conexiones, nn_nodos)
        flecha_a_nn = Arrow(trans_lineal_q.get_right(), nn_nodos[0].get_left(), buff=0.1, color=BLUE_A)
        
        self.play(GrowArrow(flecha_a_nn), FadeIn(nn_vgraph))
        animate_flow(flecha_a_nn, BLUE_A)
        
        colores = [TEAL, GREEN, YELLOW, ORANGE]
        
        self.play(*[n.animate.set_fill(colores[0], opacity=0.8).set_color(colores[0]) for n in nn_nodos[0]], run_time=0.2)
        
        for i in range(len(capas_config) - 1):
            self.play(conexiones_por_capa[i].animate.set_color(colores[i+1]).set_stroke(width=3.5), run_time=0.2)
            self.play(*[n.animate.set_fill(colores[i+1], opacity=0.8).set_color(colores[i+1]) for n in nn_nodos[i+1]], run_time=0.2)
            
        v_output = MathTex("V", color=ORANGE).scale(1.5).next_to(nn_nodos[-1], RIGHT, buff=0.3)
        self.play(Write(v_output), Flash(v_output, color=ORANGE))
        
        # P BRANCH: Kinetic Energy (T)
        bloque_m_inv = RoundedRectangle(width=1.4, height=0.9, color=RED_E).set_fill(RED_E, opacity=0.2)
        bloque_m_inv.move_to(RIGHT * 0.2 + DOWN * 2.5) 
        
        texto_m_inv = MathTex(r"M^{-1}", color=WHITE).move_to(bloque_m_inv.get_center())
        encapsulado_m = VGroup(bloque_m_inv, texto_m_inv)
        
        nodo_suma_t = Circle(radius=0.3, color=YELLOW_E).move_to(RIGHT * 2.5 + DOWN * 2.5) 
        simbolo_suma_t = MathTex(r"\sum", color=WHITE).move_to(nodo_suma_t.get_center())
        grupo_suma_t = VGroup(nodo_suma_t, simbolo_suma_t)

        flecha_a_m = Arrow(trans_lineal_p.get_right(), bloque_m_inv.get_left(), buff=0.1, color=RED_A)
        flecha_m_suma = Arrow(bloque_m_inv.get_right(), nodo_suma_t.get_left(), buff=0.1, color=YELLOW_A)

        self.play(GrowArrow(flecha_a_m), FadeIn(encapsulado_m))
        animate_flow(flecha_a_m, RED_A)
        self.play(bloque_m_inv.animate.set_stroke(YELLOW, width=4).set_fill(YELLOW, opacity=0.4), run_time=0.6)
        
        self.play(GrowArrow(flecha_m_suma), FadeIn(grupo_suma_t))
        animate_flow(flecha_m_suma, YELLOW_A)
        self.play(Indicate(grupo_suma_t, color=YELLOW, scale_factor=1.3))

        t_output = MathTex("T", color=YELLOW).scale(1.5).move_to(RIGHT * 4.0 + DOWN * 2.5)
        flecha_suma_t = Arrow(nodo_suma_t.get_right(), t_output.get_left(), buff=0.1, color=YELLOW_C)
        self.play(GrowArrow(flecha_suma_t), Write(t_output))
        animate_flow(flecha_suma_t, YELLOW)
        self.play(Flash(t_output, color=YELLOW))
        
        v_pos = RIGHT * 3 + UP * 1.5
        t_pos = RIGHT * 3 + DOWN * 1.5
        self.play(
            FadeOut(trans_lineal_q), FadeOut(flecha_a_nn), FadeOut(nn_vgraph),
            FadeOut(trans_lineal_p), FadeOut(flecha_a_m), FadeOut(encapsulado_m),
            FadeOut(flecha_m_suma), FadeOut(grupo_suma_t), FadeOut(flecha_suma_t),
            v_output.animate.move_to(v_pos),
            t_output.animate.move_to(t_pos)
        )

        # =========================================================================
        # 4. EXPLICIT ADDITIVE FUSION (Summation Node)
        # =========================================================================
        nodo_suma = Circle(radius=0.4, color=GREEN_C).move_to(RIGHT * 5 + DOWN * 0.0)
        simbolo_suma = MathTex(r"\bigoplus", color=WHITE).scale(1.5).move_to(nodo_suma.get_center())
        grupo_suma = VGroup(nodo_suma, simbolo_suma)
        
        flecha_v_suma = Arrow(v_output.get_right(), nodo_suma.get_top(), buff=0.1, color=ORANGE)
        flecha_t_suma = Arrow(t_output.get_right(), nodo_suma.get_bottom(), buff=0.1, color=YELLOW)
        
        self.play(FadeIn(grupo_suma))
        self.play(GrowArrow(flecha_v_suma), GrowArrow(flecha_t_suma))
        
        animate_flow(flecha_v_suma, ORANGE, 0.5)
        animate_flow(flecha_t_suma, YELLOW, 0.5)
        
        self.play(
            grupo_suma.animate.scale(1.5).set_color(GREEN_C),
            FadeOut(flecha_v_suma), FadeOut(flecha_t_suma),
            FadeOut(v_output), FadeOut(t_output),
            run_time=0.5
        )
        h_output = MathTex("H", "=", "T + V", color=GREEN_C).scale(1.8).move_to(grupo_suma.get_center())
        self.play(ReplacementTransform(grupo_suma, h_output), Flash(h_output, color=GREEN_E, line_length=0.4))
        self.wait(0.5)

        self.play(h_output.animate.scale(0.6).move_to(LEFT * 5.0), FadeOut(header))

        # =========================================================================
        # 5. SYMPLECTIC AUTOGRAD
        # =========================================================================
        bloque_autograd = RoundedRectangle(width=4.0, height=2.0, color=GREEN_A).move_to(LEFT * 0.5)
        texto_ag_math = MathTex(r"\nabla_{\text{symp}} H", color=WHITE).move_to(bloque_autograd.get_center() + UP * 0.3).scale(1.2)
        texto_ag_desc = Text("Symplectic Autograd", font_size=18, color=LIGHT_GRAY).next_to(texto_ag_math, DOWN, buff=0.2)
        
        flecha_h_a_ag = Arrow(h_output.get_right(), bloque_autograd.get_left(), buff=0.2, color=GREEN_C)
        self.play(GrowArrow(flecha_h_a_ag), DrawBorderThenFill(bloque_autograd), Write(texto_ag_math), FadeIn(texto_ag_desc))
        animate_flow(flecha_h_a_ag, GREEN_C)
        
        dot_r_lat = MathTex(r"\dot{\vec{\hat{q}}}_{\text{lat}} = \frac{\partial H}{\partial \vec{\hat{p}}}", color=BLUE_A).scale(1.1).move_to(RIGHT * 3.5 + UP * 1.5)
        dot_p_lat = MathTex(r"\dot{\vec{\hat{p}}}_{\text{lat}} = -\frac{\partial H}{\partial \vec{\hat{q}}}", color=RED_A).scale(1.1).move_to(RIGHT * 3.5 + DOWN * 1.5)
        
        flecha_ag_r = Arrow(bloque_autograd.get_right(), dot_r_lat.get_left(), buff=0.2, color=BLUE_B)
        flecha_ag_p = Arrow(bloque_autograd.get_right(), dot_p_lat.get_left(), buff=0.2, color=RED_B)
        
        self.play(GrowArrow(flecha_ag_r), FadeIn(dot_r_lat, shift=RIGHT))
        self.play(GrowArrow(flecha_ag_p), FadeIn(dot_p_lat, shift=RIGHT))
        animate_flow(flecha_ag_r, BLUE_A)
        animate_flow(flecha_ag_p, RED_A)

        # =========================================================================
        # 6. LINEAR TRANSFORMATION OF DERIVATIVES (ZERO BIAS)
        # =========================================================================
        self.play(
            FadeOut(h_output), FadeOut(flecha_h_a_ag), FadeOut(bloque_autograd), FadeOut(texto_ag_math), FadeOut(texto_ag_desc),
            FadeOut(flecha_ag_r), FadeOut(flecha_ag_p),
            dot_r_lat.animate.move_to(LEFT * 4.5 + UP * 1.5).scale(0.8),
            dot_p_lat.animate.move_to(LEFT * 4.5 + DOWN * 1.5).scale(0.8)
        )
        
        r_dot_norm = MathTex(r"\dot{\vec{\hat{q}}} = \dot{\vec{\hat{q}}}_{\text{lat}} \odot W_{\dot{q}}", color=TEAL_C).scale(0.9).move_to(ORIGIN + UP * 1.5)
        p_dot_norm = MathTex(r"\dot{\vec{\hat{p}}} = \dot{\vec{\hat{p}}}_{\text{lat}} \odot W_{\dot{p}}", color=RED_C).scale(0.9).move_to(ORIGIN + DOWN * 1.5)

        flecha_lin_r = Arrow(dot_r_lat.get_right(), r_dot_norm.get_left(), buff=0.1, color=BLUE_A)
        flecha_lin_p = Arrow(dot_p_lat.get_right(), p_dot_norm.get_left(), buff=0.1, color=RED_A)
        
        self.play(GrowArrow(flecha_lin_r), FadeIn(r_dot_norm))
        animate_flow(flecha_lin_r, BLUE_A)
        self.play(GrowArrow(flecha_lin_p), FadeIn(p_dot_norm))
        animate_flow(flecha_lin_p, RED_A)
        
        self.play(Circumscribe(r_dot_norm[0][-2:], color=YELLOW), Circumscribe(p_dot_norm[0][-2:], color=YELLOW), run_time=1.5)

        # =========================================================================
        # 7. FINAL DENORMALIZATION WITH max(|dot{q}|) (Real Physical Values)
        # =========================================================================
        denorm_r_final = MathTex(r"\dot{\vec{q}}", "=", r"\dot{\vec{\hat{q}}}", r"\odot", r"\max(|\dot{\vec{q}}|)").scale(0.9).move_to(RIGHT * 4.5 + UP * 1.5)
        denorm_r_final[0].set_color(BLUE_C)  # Sin sombrerito a la izquierda
        denorm_r_final[2].set_color(TEAL_C)
        denorm_r_final[4].set_color(TEAL_A)

        denorm_p_final = MathTex(r"\dot{\vec{p}}", "=", r"\dot{\vec{\hat{p}}}", r"\odot", r"\max(|\dot{\vec{p}}|)").scale(0.9).move_to(RIGHT * 4.5 + DOWN * 1.5)
        denorm_p_final[0].set_color(RED_C)  # Sin sombrerito a la izquierda
        denorm_p_final[2].set_color(RED_C)
        denorm_p_final[4].set_color(MAROON_A)
        
        flecha_denorm_r = Arrow(r_dot_norm.get_right(), denorm_r_final.get_left(), buff=0.1, color=TEAL_C)
        flecha_denorm_p = Arrow(p_dot_norm.get_right(), denorm_p_final.get_left(), buff=0.1, color=RED_C)
        
        self.play(GrowArrow(flecha_denorm_r), FadeIn(denorm_r_final))
        animate_flow(flecha_denorm_r, TEAL_A)
        self.play(GrowArrow(flecha_denorm_p), FadeIn(denorm_p_final))
        animate_flow(flecha_denorm_p, MAROON_A)
        
        self.play(Circumscribe(denorm_r_final[0], color=TEAL_A, time_width=2), Circumscribe(denorm_p_final[0], color=RED_A, time_width=2))
        self.wait(1)
        
        self.play(*[FadeOut(m) for m in self.mobjects])

        # =========================================================================
        # 8. PHYSICAL INTEGRATION (600 YEARS ORBITAL STABILITY)
        # =========================================================================
        texto_integracion = Text("SLL-HNN Symplectic Integration (600 Years)", font_size=24, color=YELLOW_A).to_edge(UP)
        counter_label = Text("Time (Years): ", font_size=20, color=WHITE).move_to(RIGHT * 3.5 + UP * 2.5)
        counter = DecimalNumber(0, num_decimal_places=0, color=GREEN_C).next_to(counter_label, RIGHT)
        
        self.play(FadeIn(texto_integracion), FadeIn(counter_label), FadeIn(counter))
        
        sol = Dot(point=ORIGIN, radius=0.4, color=YELLOW)
        brillo_sol = Circle(radius=0.6, color=YELLOW_C, stroke_width=0).set_fill(YELLOW_C, opacity=0.3)
        self.play(FadeIn(sol), GrowFromCenter(brillo_sol))
        self.play(brillo_sol.animate.scale(1.3), rate_func=there_and_back, run_time=2, loops=2)
        
        orbita_1 = Ellipse(width=4.0, height=2.5, stroke_width=1.5, color=BLUE_E).set_opacity(0.6)
        planeta_1 = Dot(radius=0.12, color=TEAL_A)
        orbita_2 = Ellipse(width=7.5, height=4.5, stroke_width=1.5, color=RED_E).set_opacity(0.6)
        orbita_2.rotate(15 * DEGREES) 
        planeta_2 = Dot(radius=0.15, color=ORANGE)
        
        self.play(Create(orbita_1), Create(orbita_2))
        
        planeta_1.move_to(orbita_1.point_from_proportion(0))
        planeta_2.move_to(orbita_2.point_from_proportion(0))
        self.play(FadeIn(planeta_1, scale=0), FadeIn(planeta_2, scale=0))
        
        estela_1 = TracedPath(planeta_1.get_center, stroke_width=3, stroke_color=TEAL_A, dissipating_time=0.8)
        estela_2 = TracedPath(planeta_2.get_center, stroke_width=3, stroke_color=ORANGE, dissipating_time=1.2)
        self.add(estela_1, estela_2)
        
        self.play(
            MoveAlongPath(planeta_1, orbita_1, rate_func=linear),
            MoveAlongPath(planeta_2, orbita_2, rate_func=linear),
            counter.animate.set_value(600),
            run_time=5.0
        )
        
        self.wait(2.0)

        # =========================================================================
        # 9. ORGANIC GRAVITATIONAL DISINTEGRATION (Loop Closure)
        # =========================================================================
        piezas = []
        for m in self.mobjects:
            if isinstance(m, (Text, MathTex, Tex, VGroup)):
                piezas.extend(m.submobjects)
            else:
                piezas.append(m)
                
        random.shuffle(piezas)
        
        animaciones_caida = [
            p.animate(
                path_arc=random.uniform(-1.5 * PI, 1.5 * PI)
            ).move_to(ORIGIN).scale(0.01).rotate(random.uniform(-2 * PI, 2 * PI)).set_opacity(0)
            for p in piezas if p is not None
        ]
        
        self.play(AnimationGroup(*animaciones_caida, lag_ratio=0.02), run_time=5.0, rate_func=smooth)
        self.wait(1.00)