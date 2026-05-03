from manim import *
import numpy as np

inputs  = [1.0, 2.0, 3.0, 2.5]
weights = [
    [ 0.2,  0.8, -0.5,  1.0],
    [ 0.5, -0.91, 0.26, -0.5],
    [-0.26,-0.27,  0.17,  0.87],
]
biases = [2.0, 3.0, 0.5]
outputs = list(np.dot(weights, inputs) + biases)


class ThreeNeurons(Scene):
    def construct(self):
        # ── layout ──────────────────────────────────────────────
        input_x  = -4.5
        neuron_x =  0.0
        output_x =  4.5

        input_ys  = [1.5, 0.5, -0.5, -1.5]
        neuron_ys = [1.5,  0.0, -1.5]

        node_r = 0.35

        # ── helper ──────────────────────────────────────────────
        def make_node(x, y, label, color=WHITE):
            c = Circle(radius=node_r, color=color, stroke_width=2.5)
            c.move_to([x, y, 0])
            t = Text(label, font_size=20, color=color).move_to([x, y, 0])
            return VGroup(c, t)

        # ── title ───────────────────────────────────────────────
        title = Text("3-Neuron Layer Forward Pass", font_size=30, color=YELLOW)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title))
        self.wait(0.4)

        # ── input nodes ─────────────────────────────────────────
        input_nodes = VGroup(*[
            make_node(input_x, y, f"x_{i+1}", BLUE)
            for i, y in enumerate(input_ys)
        ])

        input_labels = VGroup(*[
            Text(str(v), font_size=18, color=BLUE_B).next_to(
                input_nodes[i], LEFT, buff=0.15)
            for i, v in enumerate(inputs)
        ])

        self.play(LaggedStart(*[FadeIn(n) for n in input_nodes], lag_ratio=0.2))
        self.play(LaggedStart(*[FadeIn(l) for l in input_labels], lag_ratio=0.15))
        self.wait(0.3)

        # ── neuron nodes ────────────────────────────────────────
        neuron_nodes = VGroup(*[
            make_node(neuron_x, y, f"N_{i+1}", GREEN)
            for i, y in enumerate(neuron_ys)
        ])

        self.play(LaggedStart(*[FadeIn(n) for n in neuron_nodes], lag_ratio=0.25))
        self.wait(0.3)

        # ── edges (input → neuron) with weight labels ───────────
        all_edges = VGroup()
        all_weight_labels = VGroup()

        for j, ny in enumerate(neuron_ys):
            for i, iy in enumerate(input_ys):
                w = weights[j][i]
                color = GREEN_B if w >= 0 else RED_B

                start = np.array([input_x + node_r, iy, 0])
                end   = np.array([neuron_x - node_r, ny, 0])

                edge = Arrow(start, end, buff=0,
                             stroke_width=1.2, color=color,
                             max_tip_length_to_length_ratio=0.08)
                all_edges.add(edge)

                mid = (start + end) / 2 + np.array([0, 0.15, 0])
                wl  = Text(f"{w}", font_size=13, color=color).move_to(mid)
                all_weight_labels.add(wl)

        self.play(LaggedStart(*[GrowArrow(e) for e in all_edges], lag_ratio=0.04))
        self.play(LaggedStart(*[FadeIn(l) for l in all_weight_labels], lag_ratio=0.04))
        self.wait(0.5)

        # ── bias labels inside neurons ───────────────────────────
        bias_labels = VGroup(*[
            Text(f"+{b}", font_size=16, color=YELLOW).next_to(
                neuron_nodes[i], UP, buff=0.05)
            for i, b in enumerate(biases)
        ])
        bias_title = Text("biases", font_size=16, color=YELLOW).next_to(
            bias_labels, UP, buff=0.1)

        self.play(FadeIn(bias_title),
                  LaggedStart(*[FadeIn(l) for l in bias_labels], lag_ratio=0.2))
        self.wait(0.4)

        # ── output nodes ────────────────────────────────────────
        output_nodes = VGroup(*[
            make_node(output_x, y, f"o_{i+1}", ORANGE)
            for i, y in enumerate(neuron_ys)
        ])

        out_value_labels = VGroup(*[
            Text(f"{v:.2f}", font_size=18, color=ORANGE).next_to(
                output_nodes[i], RIGHT, buff=0.15)
            for i, v in enumerate(outputs)
        ])

        out_edges = VGroup(*[
            Arrow(
                np.array([neuron_x + node_r, ny, 0]),
                np.array([output_x - node_r, ny, 0]),
                buff=0, stroke_width=2, color=ORANGE,
                max_tip_length_to_length_ratio=0.12,
            )
            for ny in neuron_ys
        ])

        self.play(LaggedStart(*[GrowArrow(e) for e in out_edges], lag_ratio=0.2))
        self.play(LaggedStart(*[FadeIn(n) for n in output_nodes], lag_ratio=0.2))
        self.play(LaggedStart(*[Write(l) for l in out_value_labels], lag_ratio=0.2))
        self.wait(0.5)

        # ── formula banner ──────────────────────────────────────
        formula = MathTex(
            r"\text{output} = W \cdot x + b",
            font_size=28, color=WHITE,
        ).to_edge(DOWN, buff=0.35)
        self.play(Write(formula))
        self.wait(2)

        # ── final highlight pulse ────────────────────────────────
        self.play(
            *[Flash(output_nodes[i], color=ORANGE, flash_radius=0.6)
              for i in range(3)],
            run_time=1.2,
        )
        self.wait(1.5)
