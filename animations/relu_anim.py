from manim import *
import numpy as np

inputs      = [-3.0, -1.5, 0.0, 2.0, 4.0]
relu_out    = [max(0, x) for x in inputs]
colors_in   = [RED_B if x < 0 else GREEN_B for x in inputs]


class ReLUExplained(Scene):
    def construct(self):

        # ── 1. TITLE ────────────────────────────────────────────
        title    = Text("ReLU Activation Function", font_size=46, color=YELLOW, weight=BOLD)
        subtitle = Text("Keeping what matters, killing the rest",
                        font_size=24, color=GREY_B)
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(subtitle))
        self.wait(1.2)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ── 2. REAL-WORLD ANALOGY ────────────────────────────────
        analogy = Text(
            "Think of ReLU like a water tap:\n"
            "Negative pressure → no water flows  (output = 0)\n"
            "Positive pressure → water flows freely  (output = x)",
            font_size=26, color=WHITE, line_spacing=1.5,
        ).to_edge(UP, buff=0.5)
        self.play(Write(analogy), run_time=2)
        self.wait(1.5)
        self.play(FadeOut(analogy))

        # ── 3. THE RULE ─────────────────────────────────────────
        rule_title = Text("The Rule is Simple:", font_size=30, color=YELLOW).to_edge(UP, buff=0.5)
        self.play(Write(rule_title))

        rule = MathTex(
            r"\text{ReLU}(x) = \max(0,\, x) = \begin{cases} 0 & x < 0 \\ x & x \geq 0 \end{cases}",
            font_size=38, color=WHITE,
        ).move_to(ORIGIN + UP * 0.5)
        self.play(Write(rule), run_time=1.5)

        neg_note = Text("Negative?  → becomes 0", font_size=22, color=RED_B)
        pos_note = Text("Positive?  → stays as is", font_size=22, color=GREEN_B)
        neg_note.next_to(rule, DOWN, buff=0.4)
        pos_note.next_to(neg_note, DOWN, buff=0.2)
        self.play(FadeIn(neg_note), FadeIn(pos_note))
        self.wait(1.5)
        self.play(FadeOut(rule_title), FadeOut(rule), FadeOut(neg_note), FadeOut(pos_note))

        # ── 4. THE GRAPH ─────────────────────────────────────────
        graph_title = Text("ReLU Graph", font_size=30, color=YELLOW).to_edge(UP, buff=0.4)
        self.play(Write(graph_title))

        ax = Axes(
            x_range=[-4, 4, 1], y_range=[-1, 4, 1],
            x_length=7, y_length=4,
            axis_config={"color": GREY_B, "include_numbers": True, "font_size": 20},
        ).move_to(ORIGIN + DOWN * 0.3)
        x_label = ax.get_x_axis_label(MathTex("x", font_size=26))
        y_label = ax.get_y_axis_label(MathTex(r"\text{ReLU}(x)", font_size=22))

        self.play(Create(ax), FadeIn(x_label), FadeIn(y_label))

        relu_neg = ax.plot(lambda x: 0,        x_range=[-4, 0], color=RED_B,   stroke_width=4)
        relu_pos = ax.plot(lambda x: x,         x_range=[0,  4], color=GREEN_B, stroke_width=4)
        origin_dot = Dot(ax.coords_to_point(0, 0), color=YELLOW, radius=0.07)

        self.play(Create(relu_neg), run_time=0.8)
        self.play(Create(relu_pos), run_time=0.8)
        self.play(FadeIn(origin_dot))

        neg_lbl = Text("= 0  (flat)", font_size=18, color=RED_B).move_to(ax.coords_to_point(-2.5, 0.4))
        pos_lbl = Text("= x  (grows)", font_size=18, color=GREEN_B).move_to(ax.coords_to_point(2.5, 3.2))
        self.play(FadeIn(neg_lbl), FadeIn(pos_lbl))
        self.wait(2)
        self.play(FadeOut(graph_title), FadeOut(ax), FadeOut(x_label), FadeOut(y_label),
                  FadeOut(relu_neg), FadeOut(relu_pos), FadeOut(origin_dot),
                  FadeOut(neg_lbl), FadeOut(pos_lbl))

        # ── 5. STEP-BY-STEP EXAMPLE ─────────────────────────────
        ex_title = Text("Let's run some numbers through ReLU!",
                        font_size=28, color=YELLOW).to_edge(UP, buff=0.4)
        self.play(Write(ex_title))

        rows = VGroup()
        for i, (x, r) in enumerate(zip(inputs, relu_out)):
            col = RED_B if x < 0 else GREEN_B
            arrow = MathTex(r"\rightarrow", font_size=28, color=WHITE)
            inp   = MathTex(f"\\text{{ReLU}}({x})", font_size=28, color=WHITE)
            out   = MathTex(f"= {r:.1f}", font_size=28, color=col)
            note  = Text("(negative → 0)" if x < 0 else "(positive → kept)",
                         font_size=18, color=GREY_B)
            row = VGroup(inp, arrow, out, note).arrange(RIGHT, buff=0.3)
            row.move_to([0, 1.5 - i * 0.75, 0])
            rows.add(row)

        self.play(LaggedStart(*[Write(r) for r in rows], lag_ratio=0.35))
        self.wait(1.5)
        self.play(FadeOut(ex_title), FadeOut(rows))

        # ── 6. BEFORE / AFTER BARS ───────────────────────────────
        bar_title = Text("Before vs After ReLU", font_size=30, color=YELLOW).to_edge(UP, buff=0.4)
        self.play(Write(bar_title))

        bar_w = 0.5
        scale = 0.5
        x_positions = np.linspace(-4, 4, len(inputs))

        before_bars = VGroup()
        after_bars  = VGroup()

        for i, (x, r) in enumerate(zip(inputs, relu_out)):
            cx = x_positions[i]

            # before
            h_b  = abs(x) * scale
            col_b = RED_B if x < 0 else GREEN_B
            b_rect = Rectangle(width=bar_w, height=max(h_b, 0.05),
                                fill_color=col_b, fill_opacity=0.7, stroke_color=col_b)
            if x < 0:
                b_rect.next_to([cx, -0.3, 0], DOWN, buff=0)
            else:
                b_rect.next_to([cx, -0.3, 0], UP, buff=0)
            b_val = MathTex(str(x), font_size=18, color=col_b)
            b_val.next_to(b_rect, DOWN if x >= 0 else UP, buff=0.1)
            before_bars.add(VGroup(b_rect, b_val))

            # after
            h_a   = r * scale
            a_rect = Rectangle(width=bar_w, height=max(h_a, 0.05),
                                fill_color=GREEN_B, fill_opacity=0.7, stroke_color=GREEN_B)
            a_rect.next_to([cx, -0.3, 0], UP, buff=0)
            a_val = MathTex(str(r), font_size=18, color=GREEN_B)
            a_val.next_to(a_rect, DOWN, buff=0.1)
            after_bars.add(VGroup(a_rect, a_val))

        axis_line = Line([-5, -0.3, 0], [5, -0.3, 0], color=GREY_B, stroke_width=1.5)

        before_lbl = Text("Input", font_size=20, color=WHITE).move_to([-5.5, 0.5, 0])
        after_lbl  = Text("After\nReLU", font_size=20, color=GREEN_B).move_to([5.5, 0.5, 0])

        self.play(Create(axis_line))
        self.play(LaggedStart(*[GrowFromPoint(b[0], b[0].get_bottom())
                                for b in before_bars], lag_ratio=0.15))
        self.play(LaggedStart(*[FadeIn(b[1]) for b in before_bars], lag_ratio=0.1))
        self.play(FadeIn(before_lbl))
        self.wait(0.8)

        self.play(LaggedStart(
            *[Transform(before_bars[i][0], after_bars[i][0]) for i in range(len(inputs))],
            *[Transform(before_bars[i][1], after_bars[i][1]) for i in range(len(inputs))],
            lag_ratio=0.2,
        ))
        self.play(FadeIn(after_lbl))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ── 7. WHY WE USE RELU ───────────────────────────────────
        why_title = Text("Why Do We Use ReLU?", font_size=34, color=YELLOW, weight=BOLD)
        why_title.to_edge(UP, buff=0.5)
        self.play(Write(why_title))

        points = VGroup(
            Text("• Adds non-linearity so the network can learn complex patterns",
                 font_size=22, color=BLUE_B),
            Text("• Very fast to compute — just max(0, x)",
                 font_size=22, color=GREEN_B),
            Text("• Prevents vanishing gradients (unlike sigmoid)",
                 font_size=22, color=ORANGE),
            Text("• Used in almost every modern deep learning model",
                 font_size=22, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(ORIGIN)
        self.play(LaggedStart(*[FadeIn(p) for p in points], lag_ratio=0.45))
        self.wait(2.5)

        self.play(*[FadeOut(m) for m in self.mobjects])
        end = Text("That's ReLU!", font_size=52, color=YELLOW, weight=BOLD)
        self.play(Write(end))
        self.wait(2)
