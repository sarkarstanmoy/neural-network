from manim import *
import numpy as np

# Example: animal classifier
labels   = ["Cat", "Dog", "Bird"]
scores   = [2.0, 1.0, 0.1]
colors   = [BLUE, GREEN, ORANGE]

exp_vals = np.exp(scores)
probs    = exp_vals / exp_vals.sum()


def bar_chart(values, bar_labels, bar_colors, max_h=2.5, width=0.7, x_start=-2.0):
    group = VGroup()
    for i, (v, lbl, col) in enumerate(zip(values, bar_labels, bar_colors)):
        h = max(v / max(values) * max_h, 0.05)
        x = x_start + i * 2.2
        rect = Rectangle(width=width, height=h, fill_color=col,
                         fill_opacity=0.85, stroke_color=col)
        rect.align_to(ORIGIN, DOWN).move_to([x, 0, 0])
        rect.shift(DOWN * 0.5)

        val_tex = MathTex(f"{v:.2f}", font_size=22, color=col)
        val_tex.next_to(rect, UP, buff=0.1)

        lbl_tex = Text(lbl, font_size=22, color=WHITE)
        lbl_tex.next_to(rect, DOWN, buff=0.15)

        group.add(VGroup(rect, val_tex, lbl_tex))
    return group


class SoftmaxExplained(Scene):
    def construct(self):

        # ── 1. TITLE ────────────────────────────────────────────
        title = Text("Softmax Function", font_size=48, color=YELLOW, weight=BOLD)
        subtitle = Text("Turning scores into probabilities",
                        font_size=26, color=GREY_B)
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(subtitle))
        self.wait(1.2)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ── 2. STORY SETUP ──────────────────────────────────────
        story = Text(
            "Imagine a neural network that looks at a photo\n"
            "and tries to guess: Cat, Dog, or Bird?",
            font_size=28, color=WHITE, line_spacing=1.4,
        ).to_edge(UP, buff=0.6)
        self.play(Write(story), run_time=1.8)
        self.wait(1)

        network_out = Text(
            "The network gives raw scores (called logits):",
            font_size=24, color=GREY_B,
        ).next_to(story, DOWN, buff=0.4)
        self.play(FadeIn(network_out))
        self.wait(0.5)

        # raw score bars
        raw_bars = bar_chart(scores, labels, colors, max_h=2.2)
        raw_bars.move_to(ORIGIN + DOWN * 0.3)
        self.play(LaggedStart(
            *[GrowFromEdge(b[0], DOWN) for b in raw_bars],
            lag_ratio=0.2,
        ))
        self.play(LaggedStart(
            *[FadeIn(b[1]) for b in raw_bars],
            *[FadeIn(b[2]) for b in raw_bars],
            lag_ratio=0.15,
        ))
        self.wait(1)

        # problem statement
        problem = Text(
            "But wait — 2.0 + 1.0 + 0.1 = 3.1  (not 100%!)",
            font_size=22, color=RED_B,
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(problem))
        self.wait(1.5)
        self.play(FadeOut(story), FadeOut(network_out),
                  FadeOut(raw_bars), FadeOut(problem))

        # ── 3. STEP 1 — EXPONENTIATION ──────────────────────────
        step1_title = Text("Step 1: Apply  ", font_size=32, color=YELLOW)
        step1_math  = MathTex(r"e^{\,\text{score}}", font_size=36, color=YELLOW)
        step1_hdr   = VGroup(step1_title, step1_math).arrange(RIGHT).to_edge(UP, buff=0.4)
        self.play(Write(step1_hdr))

        why1 = Text(
            "Why? The function eˣ makes every number positive\n"
            "and stretches bigger scores much further apart.",
            font_size=22, color=GREY_B, line_spacing=1.3,
        ).next_to(step1_hdr, DOWN, buff=0.3)
        self.play(FadeIn(why1))
        self.wait(0.8)

        # show e^x calculation for each
        calc_group = VGroup()
        for i, (s, ev, col) in enumerate(zip(scores, exp_vals, colors)):
            row = MathTex(
                f"e^{{{s}}} = {ev:.3f}",
                font_size=28, color=col,
            ).move_to([- 2.5 + i * 2.5, -0.5, 0])
            calc_group.add(row)

        self.play(LaggedStart(*[Write(r) for r in calc_group], lag_ratio=0.4))
        self.wait(0.5)

        # exp bars
        exp_bars = bar_chart(exp_vals, labels, colors, max_h=2.2)
        exp_bars.move_to(ORIGIN + DOWN * 1.6)
        self.play(LaggedStart(
            *[GrowFromEdge(b[0], DOWN) for b in exp_bars], lag_ratio=0.2,
        ))
        self.play(LaggedStart(
            *[FadeIn(b[1]) for b in exp_bars],
            *[FadeIn(b[2]) for b in exp_bars],
            lag_ratio=0.15,
        ))
        self.wait(1.5)
        self.play(FadeOut(step1_hdr), FadeOut(why1),
                  FadeOut(calc_group), FadeOut(exp_bars))

        # ── 4. STEP 2 — DIVIDE BY SUM ───────────────────────────
        step2_title = Text("Step 2: Divide each value by the total sum",
                           font_size=28, color=YELLOW).to_edge(UP, buff=0.4)
        self.play(Write(step2_title))

        sum_val = exp_vals.sum()
        sum_line = MathTex(
            rf"\text{{Sum}} = {exp_vals[0]:.3f} + {exp_vals[1]:.3f}"
            rf" + {exp_vals[2]:.3f} = {sum_val:.3f}",
            font_size=24, color=WHITE,
        ).next_to(step2_title, DOWN, buff=0.35)
        self.play(Write(sum_line))
        self.wait(0.8)

        why2 = Text(
            "Dividing by the sum makes all values add up to exactly 1\n"
            "(i.e. 100%) — now they behave like probabilities!",
            font_size=22, color=GREY_B, line_spacing=1.3,
        ).next_to(sum_line, DOWN, buff=0.3)
        self.play(FadeIn(why2))
        self.wait(0.8)

        # probability calculation rows
        prob_calcs = VGroup()
        for i, (ev, p, lbl, col) in enumerate(zip(exp_vals, probs, labels, colors)):
            row = MathTex(
                rf"P(\text{{{lbl}}}) = \frac{{{ev:.3f}}}{{{sum_val:.3f}}} = {p:.3f}",
                font_size=24, color=col,
            ).move_to([0, -0.6 - i * 0.85, 0])
            prob_calcs.add(row)

        self.play(LaggedStart(*[Write(r) for r in prob_calcs], lag_ratio=0.5))
        self.wait(1.5)
        self.play(FadeOut(step2_title), FadeOut(sum_line),
                  FadeOut(why2), FadeOut(prob_calcs))

        # ── 5. FINAL PROBABILITIES ──────────────────────────────
        result_title = Text("Final Result — Probabilities!", font_size=34,
                            color=YELLOW, weight=BOLD).to_edge(UP, buff=0.4)
        self.play(Write(result_title))

        prob_bars = bar_chart(probs, labels, colors, max_h=2.5)
        prob_bars.move_to(ORIGIN + DOWN * 0.2)

        # replace values with percentages
        for i, (b, p) in enumerate(zip(prob_bars, probs)):
            b[1].become(
                MathTex(f"{p*100:.1f}\\%", font_size=22, color=colors[i])
                .next_to(b[0], UP, buff=0.1)
            )

        self.play(LaggedStart(
            *[GrowFromEdge(b[0], DOWN) for b in prob_bars], lag_ratio=0.2,
        ))
        self.play(LaggedStart(
            *[FadeIn(b[1]) for b in prob_bars],
            *[FadeIn(b[2]) for b in prob_bars],
            lag_ratio=0.15,
        ))

        sums_to = MathTex(
            r"65.9\% + 24.2\% + 9.9\% \approx 100\%\ \checkmark",
            font_size=24, color=GREEN,
        ).to_edge(DOWN, buff=0.45)
        self.play(Write(sums_to))
        self.wait(1.5)

        # winner highlight
        winner_box = SurroundingRectangle(prob_bars[0], color=YELLOW, buff=0.12)
        winner_txt = Text("Most likely: Cat!", font_size=24, color=YELLOW, weight=BOLD)
        winner_txt.next_to(winner_box, UP, buff=0.15)
        self.play(Create(winner_box), Write(winner_txt))
        self.wait(1)
        self.play(FadeOut(result_title), FadeOut(prob_bars),
                  FadeOut(sums_to), FadeOut(winner_box), FadeOut(winner_txt))

        # ── 6. FORMULA SUMMARY ──────────────────────────────────
        summary_title = Text("The Softmax Formula", font_size=32,
                             color=YELLOW).to_edge(UP, buff=0.5)
        self.play(Write(summary_title))

        formula = MathTex(
            r"\text{Softmax}(x_i) = \frac{e^{x_i}}{\displaystyle\sum_{j} e^{x_j}}",
            font_size=48, color=WHITE,
        ).move_to(ORIGIN + UP * 0.3)
        self.play(Write(formula), run_time=1.5)
        self.wait(0.5)

        bullets = VGroup(
            Text("• Converts any scores into probabilities",  font_size=21, color=BLUE_B),
            Text("• All outputs are between 0 and 1",         font_size=21, color=GREEN_B),
            Text("• All outputs always add up to 1  (100%)", font_size=21, color=ORANGE),
            Text("• The biggest score always wins the most %", font_size=21, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        bullets.next_to(formula, DOWN, buff=0.45)

        self.play(LaggedStart(*[FadeIn(b) for b in bullets], lag_ratio=0.4))
        self.wait(2.5)

        # fade everything out, end card
        self.play(*[FadeOut(m) for m in self.mobjects])
        end = Text("That's Softmax!", font_size=52, color=YELLOW, weight=BOLD)
        self.play(Write(end))
        self.wait(2)
