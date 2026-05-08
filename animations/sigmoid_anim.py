from manim import *
import numpy as np

inputs   = [-4.0, -2.0, 0.0, 2.0, 4.0]
sig_out  = [1 / (1 + np.exp(-x)) for x in inputs]


class SigmoidExplained(Scene):
    def construct(self):

        # ── 1. TITLE ────────────────────────────────────────────
        title    = Text("Sigmoid Function", font_size=46, color=YELLOW, weight=BOLD)
        subtitle = Text("Squashing any number into 0 → 1",
                        font_size=24, color=GREY_B)
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(subtitle))
        self.wait(1.2)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ── 2. REAL-WORLD ANALOGY ────────────────────────────────
        analogy = Text(
            "Think of sigmoid like a dimmer switch:\n"
            "Very negative input  →  nearly OFF  (≈ 0%)\n"
            "Zero input           →  halfway       (50%)\n"
            "Very positive input  →  nearly ON    (≈ 100%)",
            font_size=25, color=WHITE, line_spacing=1.5,
        ).to_edge(UP, buff=0.4)
        self.play(Write(analogy), run_time=2.2)
        self.wait(1.5)
        self.play(FadeOut(analogy))

        # ── 3. THE FORMULA ──────────────────────────────────────
        formula_title = Text("The Formula:", font_size=30, color=YELLOW).to_edge(UP, buff=0.5)
        self.play(Write(formula_title))

        formula = MathTex(
            r"\sigma(x) = \frac{1}{1 + e^{-x}}",
            font_size=52, color=WHITE,
        ).move_to(ORIGIN + UP * 0.5)
        self.play(Write(formula), run_time=1.5)

        parts = VGroup(
            Text("e  is Euler's number ≈ 2.718  (a special math constant)",
                 font_size=20, color=BLUE_B),
            Text("The output is always between 0 and 1",
                 font_size=20, color=GREEN_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        parts.next_to(formula, DOWN, buff=0.45)
        self.play(LaggedStart(*[FadeIn(p) for p in parts], lag_ratio=0.4))
        self.wait(1.5)
        self.play(FadeOut(formula_title), FadeOut(formula), FadeOut(parts))

        # ── 4. THE S-CURVE GRAPH ─────────────────────────────────
        graph_title = Text("The Famous S-Curve", font_size=30, color=YELLOW).to_edge(UP, buff=0.4)
        self.play(Write(graph_title))

        ax = Axes(
            x_range=[-5, 5, 1], y_range=[-0.1, 1.1, 0.25],
            x_length=8, y_length=4,
            axis_config={"color": GREY_B, "include_numbers": True, "font_size": 18},
        ).move_to(ORIGIN + DOWN * 0.2)
        x_lbl = ax.get_x_axis_label(MathTex("x", font_size=26))
        y_lbl = ax.get_y_axis_label(MathTex(r"\sigma(x)", font_size=24))

        self.play(Create(ax), FadeIn(x_lbl), FadeIn(y_lbl))

        curve = ax.plot(lambda x: 1 / (1 + np.exp(-x)),
                        x_range=[-5, 5], color=YELLOW, stroke_width=4)
        self.play(Create(curve), run_time=1.8)

        # key reference lines
        half_line = ax.get_horizontal_line(ax.coords_to_point(5, 0.5),
                                           color=GREEN_B, stroke_width=1.5,
                                           line_func=DashedLine)
        one_line  = ax.get_horizontal_line(ax.coords_to_point(5, 1.0),
                                           color=RED_B, stroke_width=1.5,
                                           line_func=DashedLine)
        zero_line = ax.get_horizontal_line(ax.coords_to_point(5, 0.0),
                                           color=RED_B, stroke_width=1.5,
                                           line_func=DashedLine)

        lbl_half = MathTex(r"\sigma = 0.5", font_size=20, color=GREEN_B)\
            .next_to(ax.coords_to_point(3.5, 0.5), UP, buff=0.1)
        lbl_one  = MathTex(r"\sigma \to 1", font_size=20, color=RED_B)\
            .next_to(ax.coords_to_point(3.5, 1.0), UP, buff=0.1)
        lbl_zero = MathTex(r"\sigma \to 0", font_size=20, color=RED_B)\
            .next_to(ax.coords_to_point(3.5, 0.0), DOWN, buff=0.1)

        self.play(Create(half_line), FadeIn(lbl_half))
        self.play(Create(one_line),  FadeIn(lbl_one))
        self.play(Create(zero_line), FadeIn(lbl_zero))

        origin_dot = Dot(ax.coords_to_point(0, 0.5), color=GREEN_B, radius=0.08)
        origin_note = Text("At x=0, output = 0.5", font_size=18, color=GREEN_B)\
            .next_to(ax.coords_to_point(0, 0.5), LEFT, buff=0.15)
        self.play(FadeIn(origin_dot), FadeIn(origin_note))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ── 5. STEP-BY-STEP EXAMPLE ─────────────────────────────
        ex_title = Text("Let's plug in some values!", font_size=28, color=YELLOW).to_edge(UP, buff=0.4)
        self.play(Write(ex_title))

        rows = VGroup()
        for i, (x, s) in enumerate(zip(inputs, sig_out)):
            col = BLUE_B if x < 0 else GREEN_B
            inp  = MathTex(rf"\sigma({x})", font_size=26, color=WHITE)
            eq   = MathTex(rf"= \frac{{1}}{{1 + e^{{-({x})}}}}", font_size=24, color=GREY_B)
            out  = MathTex(rf"= {s:.3f}", font_size=26, color=col)
            pct  = Text(f"≈ {s*100:.1f}%", font_size=20, color=col)
            row  = VGroup(inp, eq, out, pct).arrange(RIGHT, buff=0.3)
            row.move_to([0, 1.8 - i * 0.9, 0])
            rows.add(row)

        self.play(LaggedStart(*[Write(r) for r in rows], lag_ratio=0.35))
        self.wait(1.5)
        self.play(FadeOut(ex_title), FadeOut(rows))

        # ── 6. VISUAL BAR — BEFORE / AFTER ──────────────────────
        bar_title = Text("Input vs Sigmoid Output", font_size=30, color=YELLOW).to_edge(UP, buff=0.4)
        self.play(Write(bar_title))

        x_pos = np.linspace(-4, 4, len(inputs))
        bar_w = 0.55
        axis_y = -1.0

        axis_line = Line([-5.5, axis_y, 0], [5.5, axis_y, 0], color=GREY_B, stroke_width=1.5)
        self.play(Create(axis_line))

        sig_bars = VGroup()
        for i, (x, s) in enumerate(zip(inputs, sig_out)):
            cx  = x_pos[i]
            col = BLUE_B if x < 0 else GREEN_B
            h   = s * 2.5
            r   = Rectangle(width=bar_w, height=h,
                             fill_color=col, fill_opacity=0.8, stroke_color=col)
            r.next_to([cx, axis_y, 0], UP, buff=0)
            v_in  = MathTex(str(x), font_size=18, color=GREY_B)\
                .next_to(r, DOWN, buff=0.15)
            v_out = MathTex(f"{s:.2f}", font_size=18, color=col)\
                .next_to(r, UP, buff=0.08)
            sig_bars.add(VGroup(r, v_in, v_out))

        self.play(LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in sig_bars], lag_ratio=0.15))
        self.play(LaggedStart(
            *[FadeIn(b[1]) for b in sig_bars],
            *[FadeIn(b[2]) for b in sig_bars],
            lag_ratio=0.1,
        ))

        bound_note = Text("All outputs squeezed between 0 and 1  ✓",
                          font_size=22, color=GREEN).to_edge(DOWN, buff=0.4)
        self.play(Write(bound_note))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ── 7. SIGMOID vs SOFTMAX ────────────────────────────────
        compare_title = Text("Sigmoid  vs  Softmax", font_size=32, color=YELLOW).to_edge(UP, buff=0.4)
        self.play(Write(compare_title))

        table_data = VGroup(
            Text("Sigmoid",           font_size=22, color=BLUE,   weight=BOLD),
            Text("Softmax",           font_size=22, color=ORANGE, weight=BOLD),
            Text("One output at a time",   font_size=20, color=BLUE_B),
            Text("Multiple outputs at once", font_size=20, color=ORANGE),
            Text("Binary (Yes / No)",       font_size=20, color=BLUE_B),
            Text("Multi-class (Cat/Dog/Bird)", font_size=20, color=ORANGE),
            Text("Output range: 0 → 1",    font_size=20, color=BLUE_B),
            Text("All outputs sum to 1",   font_size=20, color=ORANGE),
        )

        col1 = VGroup(table_data[0], table_data[2], table_data[4], table_data[6])\
            .arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([-3, 0, 0])
        col2 = VGroup(table_data[1], table_data[3], table_data[5], table_data[7])\
            .arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([2, 0, 0])
        divider = Line([0, 1.8, 0], [0, -1.8, 0], color=GREY_B, stroke_width=1)

        self.play(Create(divider))
        self.play(LaggedStart(*[FadeIn(t) for t in col1], lag_ratio=0.3))
        self.play(LaggedStart(*[FadeIn(t) for t in col2], lag_ratio=0.3))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ── 8. SUMMARY ──────────────────────────────────────────
        summary = Text("The Sigmoid Formula", font_size=32, color=YELLOW).to_edge(UP, buff=0.5)
        self.play(Write(summary))

        formula2 = MathTex(
            r"\sigma(x) = \frac{1}{1 + e^{-x}}",
            font_size=48, color=WHITE,
        ).move_to(ORIGIN + UP * 0.4)
        self.play(Write(formula2))

        bullets = VGroup(
            Text("• Output is always between 0 and 1",          font_size=21, color=BLUE_B),
            Text("• Perfect for binary classification (yes/no)", font_size=21, color=GREEN_B),
            Text("• At x = 0, output = 0.5  (uncertain)",        font_size=21, color=YELLOW),
            Text("• Large positive → close to 1  |  Large negative → close to 0",
                 font_size=21, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        bullets.next_to(formula2, DOWN, buff=0.45)
        self.play(LaggedStart(*[FadeIn(b) for b in bullets], lag_ratio=0.4))
        self.wait(2.5)

        self.play(*[FadeOut(m) for m in self.mobjects])
        end = Text("That's Sigmoid!", font_size=52, color=YELLOW, weight=BOLD)
        self.play(Write(end))
        self.wait(2)
