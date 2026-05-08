from manim import *
import numpy as np

labels      = ["Cat", "Dog", "Bird"]
colors      = [BLUE, GREEN, ORANGE]
true_label  = 0
true_onehot = [1, 0, 0]

good_pred = [0.70, 0.20, 0.10]
bad_pred  = [0.10, 0.70, 0.20]

good_loss = -np.log(good_pred[true_label])
bad_loss  = -np.log(bad_pred[true_label])


class CrossEntropyExplained(Scene):
    def construct(self):

        # ── 1. TITLE ────────────────────────────────────────────
        title    = Text("Cross-Entropy Loss", font_size=46, color=YELLOW, weight=BOLD)
        subtitle = Text("How do we measure how wrong a neural network is?",
                        font_size=22, color=GREY_B)
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(subtitle))
        self.wait(1.3)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ── 2. ANALOGY ──────────────────────────────────────────
        analogy = Text(
            "Imagine a student taking a test.\n"
            "The teacher gives a score: 0 = perfect, big number = very wrong.\n"
            "Cross-Entropy is the teacher's score for our neural network!",
            font_size=25, color=WHITE, line_spacing=1.5,
        ).to_edge(UP, buff=0.45)
        self.play(Write(analogy), run_time=2.2)
        self.wait(1.5)
        self.play(FadeOut(analogy))

        # ── 3. SETUP THE PROBLEM ────────────────────────────────
        setup_title = Text("Our Problem:", font_size=30, color=YELLOW).to_edge(UP, buff=0.45)
        self.play(Write(setup_title))

        setup_txt = Text(
            "A neural network looks at a photo and must pick:\n"
            "Cat,  Dog,  or  Bird?\n"
            "The correct answer is:  Cat  ✓",
            font_size=26, color=WHITE, line_spacing=1.5,
        ).move_to(ORIGIN + UP * 0.3)
        self.play(Write(setup_txt), run_time=1.8)

        onehot = Text(
            "True label (one-hot):  [1,  0,  0]",
            font_size=24, color=GREEN,
        ).next_to(setup_txt, DOWN, buff=0.4)
        onehot_note = Text(
            "1 = correct class,  0 = wrong class",
            font_size=18, color=GREY_B,
        ).next_to(onehot, DOWN, buff=0.15)
        self.play(FadeIn(onehot), FadeIn(onehot_note))
        self.wait(1.5)
        self.play(FadeOut(setup_title), FadeOut(setup_txt),
                  FadeOut(onehot), FadeOut(onehot_note))

        # ── 4. THE -log CURVE ───────────────────────────────────
        curve_title = Text("The Secret Weapon:  -log(p)", font_size=30, color=YELLOW)\
            .to_edge(UP, buff=0.4)
        self.play(Write(curve_title))

        ax = Axes(
            x_range=[0.01, 1, 0.1], y_range=[0, 5, 1],
            x_length=7, y_length=4,
            axis_config={"color": GREY_B, "include_numbers": True, "font_size": 18},
            x_axis_config={"numbers_to_include": [0.2, 0.4, 0.6, 0.8, 1.0]},
        ).move_to(ORIGIN + DOWN * 0.2)
        x_lbl = ax.get_x_axis_label(Text("p", font_size=24))
        y_lbl = ax.get_y_axis_label(Text("-log(p)", font_size=20))

        self.play(Create(ax), FadeIn(x_lbl), FadeIn(y_lbl))

        log_curve = ax.plot(lambda p: -np.log(p), x_range=[0.02, 1],
                            color=RED, stroke_width=4)
        self.play(Create(log_curve), run_time=1.8)

        pt_high = Dot(ax.coords_to_point(0.1, -np.log(0.1)), color=RED_B,   radius=0.07)
        pt_low  = Dot(ax.coords_to_point(0.9, -np.log(0.9)), color=GREEN_B, radius=0.07)

        lbl_high = Text("p = 0.1\nLoss = 2.3  (bad!)",  font_size=17, color=RED_B)\
            .next_to(ax.coords_to_point(0.1, -np.log(0.1)), RIGHT, buff=0.15)
        lbl_low  = Text("p = 0.9\nLoss = 0.1  (good!)", font_size=17, color=GREEN_B)\
            .next_to(ax.coords_to_point(0.9, -np.log(0.9)), UP, buff=0.15)

        self.play(FadeIn(pt_high), FadeIn(lbl_high))
        self.play(FadeIn(pt_low),  FadeIn(lbl_low))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ── 5. FORMULA ──────────────────────────────────────────
        form_title = Text("The Formula:", font_size=30, color=YELLOW).to_edge(UP, buff=0.5)
        self.play(Write(form_title))

        formula = Text(
            "L  =  - Σ  yᵢ · log( ŷᵢ )",
            font_size=46, color=WHITE,
        ).move_to(ORIGIN + UP * 0.6)
        self.play(Write(formula), run_time=1.5)

        legend = VGroup(
            VGroup(
                Text("yᵢ",  font_size=24, color=GREEN_B),
                Text("= true label  (1 for correct class, 0 for others)",
                     font_size=20, color=WHITE),
            ).arrange(RIGHT, buff=0.2),
            VGroup(
                Text("ŷᵢ",  font_size=24, color=ORANGE),
                Text("= predicted probability from softmax",
                     font_size=20, color=WHITE),
            ).arrange(RIGHT, buff=0.2),
            VGroup(
                Text("L",  font_size=24, color=RED_B),
                Text("= the loss   (lower = better)",
                     font_size=20, color=WHITE),
            ).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        legend.next_to(formula, DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(r) for r in legend], lag_ratio=0.4))
        self.wait(1.5)
        self.play(FadeOut(form_title), FadeOut(formula), FadeOut(legend))

        # ── 6. GOOD PREDICTION ──────────────────────────────────
        good_title = Text("Good Prediction  ✓", font_size=30,
                          color=GREEN, weight=BOLD).to_edge(UP, buff=0.4)
        self.play(Write(good_title))

        good_bars = self._pred_bars(good_pred, labels, colors, true_label)
        good_bars.move_to(ORIGIN + UP * 0.3)
        self.play(LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in good_bars], lag_ratio=0.15))
        self.play(LaggedStart(
            *[FadeIn(b[1]) for b in good_bars],
            *[FadeIn(b[2]) for b in good_bars],
            lag_ratio=0.1,
        ))

        good_calc = Text(
            f"L  =  -log({good_pred[0]})  =  {good_loss:.3f}",
            font_size=30, color=GREEN_B,
        ).to_edge(DOWN, buff=0.55)
        low_note = Text("Low loss = network is confident and correct!",
                        font_size=20, color=GREEN_B)
        low_note.next_to(good_calc, UP, buff=0.2)
        self.play(Write(good_calc), FadeIn(low_note))
        self.wait(1.5)
        self.play(FadeOut(good_title), FadeOut(good_bars),
                  FadeOut(good_calc), FadeOut(low_note))

        # ── 7. BAD PREDICTION ───────────────────────────────────
        bad_title = Text("Bad Prediction  ✗", font_size=30,
                         color=RED, weight=BOLD).to_edge(UP, buff=0.4)
        self.play(Write(bad_title))

        bad_bars = self._pred_bars(bad_pred, labels, colors, true_label)
        bad_bars.move_to(ORIGIN + UP * 0.3)
        self.play(LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in bad_bars], lag_ratio=0.15))
        self.play(LaggedStart(
            *[FadeIn(b[1]) for b in bad_bars],
            *[FadeIn(b[2]) for b in bad_bars],
            lag_ratio=0.1,
        ))

        bad_calc = Text(
            f"L  =  -log({bad_pred[0]})  =  {bad_loss:.3f}",
            font_size=30, color=RED_B,
        ).to_edge(DOWN, buff=0.55)
        high_note = Text("High loss = network is confident but WRONG!",
                         font_size=20, color=RED_B)
        high_note.next_to(bad_calc, UP, buff=0.2)
        self.play(Write(bad_calc), FadeIn(high_note))
        self.wait(1.5)
        self.play(FadeOut(bad_title), FadeOut(bad_bars),
                  FadeOut(bad_calc), FadeOut(high_note))

        # ── 8. SIDE-BY-SIDE COMPARISON ──────────────────────────
        comp_title = Text("Side-by-Side Comparison", font_size=30, color=YELLOW)\
            .to_edge(UP, buff=0.4)
        self.play(Write(comp_title))

        divider = Line([0, 2.5, 0], [0, -2.8, 0], color=GREY_B, stroke_width=1.5)
        self.play(Create(divider))

        good_lbl = Text("Good (confident + correct)", font_size=18, color=GREEN)\
            .move_to([-3.2, 2.2, 0])
        bad_lbl  = Text("Bad (confident + wrong)",    font_size=18, color=RED)\
            .move_to([3.2,  2.2, 0])
        self.play(FadeIn(good_lbl), FadeIn(bad_lbl))

        for i, (p, lbl, col) in enumerate(zip(good_pred, labels, colors)):
            h  = p * 2.2
            r  = Rectangle(width=0.55, height=h,
                           fill_color=col, fill_opacity=0.8, stroke_color=col)
            r.move_to([-4.5 + i * 1.5, -0.5, 0])
            vl = Text(f"{p:.0%}", font_size=17, color=col).next_to(r, UP,   buff=0.07)
            nl = Text(lbl,        font_size=17, color=WHITE).next_to(r, DOWN, buff=0.1)
            self.add(r, vl, nl)

        for i, (p, lbl, col) in enumerate(zip(bad_pred, labels, colors)):
            h  = p * 2.2
            r  = Rectangle(width=0.55, height=h,
                           fill_color=col, fill_opacity=0.8, stroke_color=col)
            r.move_to([0.7 + i * 1.5, -0.5, 0])
            vl = Text(f"{p:.0%}", font_size=17, color=col).next_to(r, UP,   buff=0.07)
            nl = Text(lbl,        font_size=17, color=WHITE).next_to(r, DOWN, buff=0.1)
            self.add(r, vl, nl)

        good_loss_lbl = Text(f"Loss = {good_loss:.3f}", font_size=26, color=GREEN_B)\
            .move_to([-3.2, -2.4, 0])
        bad_loss_lbl  = Text(f"Loss = {bad_loss:.3f}",  font_size=26, color=RED_B)\
            .move_to([3.2,  -2.4, 0])
        self.play(FadeIn(good_loss_lbl), FadeIn(bad_loss_lbl))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ── 9. SUMMARY ──────────────────────────────────────────
        summ_title = Text("Cross-Entropy Loss — Key Ideas",
                          font_size=30, color=YELLOW).to_edge(UP, buff=0.5)
        self.play(Write(summ_title))

        formula2 = Text(
            "L  =  - Σ  yᵢ · log( ŷᵢ )",
            font_size=44, color=WHITE,
        ).move_to(ORIGIN + UP * 0.6)
        self.play(Write(formula2))

        bullets = VGroup(
            Text("• Loss = 0 means perfect prediction",                font_size=21, color=GREEN_B),
            Text("• Higher loss = network is more wrong",               font_size=21, color=RED_B),
            Text("• We train the network to minimise this loss",        font_size=21, color=BLUE_B),
            Text("• Only the true class probability matters in the sum",font_size=21, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        bullets.next_to(formula2, DOWN, buff=0.4)
        self.play(LaggedStart(*[FadeIn(b) for b in bullets], lag_ratio=0.4))
        self.wait(2.5)

        self.play(*[FadeOut(m) for m in self.mobjects])
        end = Text("That's Cross-Entropy Loss!", font_size=44, color=YELLOW, weight=BOLD)
        self.play(Write(end))
        self.wait(2)

    def _pred_bars(self, preds, labels, colors, correct_idx):
        group  = VGroup()
        x_start = -2.5
        bar_w   = 0.7
        for i, (p, lbl, col) in enumerate(zip(preds, labels, colors)):
            h  = p * 2.2
            cx = x_start + i * 2.5
            r  = Rectangle(width=bar_w, height=h,
                           fill_color=col, fill_opacity=0.8, stroke_color=col)
            r.next_to([cx, -0.8, 0], UP, buff=0)
            vl = Text(f"{p:.0%}", font_size=22, color=col).next_to(r, UP,   buff=0.08)
            nl = Text(lbl,        font_size=22, color=WHITE).next_to(r, DOWN, buff=0.12)
            if i == correct_idx:
                star = Text("✓ correct", font_size=16, color=GREEN)\
                    .next_to(r, UP, buff=0.35)
                group.add(VGroup(r, vl, nl, star))
            else:
                group.add(VGroup(r, vl, nl))
        return group
