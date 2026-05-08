from manim import *
import numpy as np

inputs_batch = [[1, 2, 3, 2.5], [2., 5., -1., 2], [-1.5, 2.7, 3.3, -0.8]]
sample = inputs_batch[0]

weights1 = np.array([[0.2, 0.8, -0.5, 1],
                     [0.5, -0.91, 0.26, -0.5],
                     [-0.26, -0.27, 0.17, 0.87]])
biases1  = np.array([2, 3, 0.5])

weights2 = np.array([[0.1, -0.14, 0.5],
                     [-0.5, 0.12, -0.33],
                     [-0.44, 0.73, -0.13]])
biases2  = np.array([-1, 2, -0.5])

layer1_out = weights1 @ sample + biases1
layer2_out = weights2 @ layer1_out + biases2


class TwoHiddenLayers(Scene):
    def construct(self):
        input_x  = -5.0
        l1_x     = -1.5
        l2_x     =  1.5
        output_x =  5.0

        input_ys  = [2.25, 0.75, -0.75, -2.25]
        hidden_ys = [1.5, 0.0, -1.5]
        node_r    = 0.32

        def node(x, y, tex, color):
            c = Circle(radius=node_r, color=color, stroke_width=2.5).move_to([x, y, 0])
            t = MathTex(tex, font_size=22, color=color).move_to([x, y, 0])
            return VGroup(c, t)

        def edge(x1, y1, x2, y2, color=GREY, sw=1.0):
            return Line(
                [x1 + node_r, y1, 0], [x2 - node_r, y2, 0],
                stroke_width=sw, color=color,
            )

        def weight_label(x1, y1, x2, y2, w, color):
            mid = np.array([(x1+x2)/2, (y1+y2)/2 + 0.13, 0])
            return MathTex(f"{w:.2f}", font_size=12, color=color).move_to(mid)

        # title
        title = Text("Two Hidden Layers — Forward Pass", font_size=28, color=YELLOW)
        title.to_edge(UP, buff=0.2)
        self.play(Write(title))
        self.wait(0.3)

        # layer labels
        lbl_input  = MathTex(r"\text{Input}",   font_size=20, color=BLUE).move_to([input_x,  -3.2, 0])
        lbl_layer1 = MathTex(r"\text{Layer 1}", font_size=20, color=GREEN).move_to([l1_x,    -2.7, 0])
        lbl_layer2 = MathTex(r"\text{Layer 2}", font_size=20, color=PURPLE).move_to([l2_x,   -2.7, 0])
        lbl_output = MathTex(r"\text{Output}",  font_size=20, color=ORANGE).move_to([output_x,-2.7, 0])
        self.play(FadeIn(lbl_input), FadeIn(lbl_layer1), FadeIn(lbl_layer2), FadeIn(lbl_output))

        # nodes
        input_nodes = VGroup(*[node(input_x,  y, f"x_{i+1}", BLUE)   for i, y in enumerate(input_ys)])
        l1_nodes    = VGroup(*[node(l1_x,     y, f"N_{i+1}", GREEN)  for i, y in enumerate(hidden_ys)])
        l2_nodes    = VGroup(*[node(l2_x,     y, f"M_{i+1}", PURPLE) for i, y in enumerate(hidden_ys)])
        out_nodes   = VGroup(*[node(output_x, y, f"o_{i+1}", ORANGE) for i, y in enumerate(hidden_ys)])

        self.play(LaggedStart(*[FadeIn(n) for n in input_nodes],  lag_ratio=0.15))
        self.play(LaggedStart(*[FadeIn(n) for n in l1_nodes],     lag_ratio=0.15))
        self.play(LaggedStart(*[FadeIn(n) for n in l2_nodes],     lag_ratio=0.15))
        self.play(LaggedStart(*[FadeIn(n) for n in out_nodes],    lag_ratio=0.15))
        self.wait(0.3)

        # input values
        in_vals = VGroup(*[
            MathTex(str(v), font_size=18, color=BLUE_B).next_to(input_nodes[i], LEFT, buff=0.12)
            for i, v in enumerate(sample)
        ])
        self.play(LaggedStart(*[FadeIn(l) for l in in_vals], lag_ratio=0.15))
        self.wait(0.2)

        # edges: input → layer 1
        edges1   = VGroup()
        wlabels1 = VGroup()
        for j, ny in enumerate(hidden_ys):
            for i, iy in enumerate(input_ys):
                w   = weights1[j][i]
                col = GREEN_B if w >= 0 else RED_B
                edges1.add(edge(input_x, iy, l1_x, ny, col))
                wlabels1.add(weight_label(input_x, iy, l1_x, ny, w, col))

        self.play(LaggedStart(*[Create(e) for e in edges1],   lag_ratio=0.03))
        self.play(LaggedStart(*[FadeIn(l) for l in wlabels1], lag_ratio=0.03))

        b1_labels = VGroup(*[
            MathTex(f"+{b}", font_size=16, color=YELLOW).next_to(l1_nodes[i], UP, buff=0.05)
            for i, b in enumerate(biases1)
        ])
        self.play(LaggedStart(*[FadeIn(l) for l in b1_labels], lag_ratio=0.2))
        self.wait(0.4)

        # layer 1 outputs
        l1_out_labels = VGroup(*[
            MathTex(f"{v:.2f}", font_size=15, color=GREEN_B).next_to(l1_nodes[i], RIGHT, buff=0.08)
            for i, v in enumerate(layer1_out)
        ])
        self.play(LaggedStart(*[Write(l) for l in l1_out_labels], lag_ratio=0.2))
        self.wait(0.3)

        # edges: layer 1 → layer 2
        edges2   = VGroup()
        wlabels2 = VGroup()
        for j, ny in enumerate(hidden_ys):
            for i, iy in enumerate(hidden_ys):
                w   = weights2[j][i]
                col = PURPLE_B if w >= 0 else RED_B
                edges2.add(edge(l1_x, iy, l2_x, ny, col))
                wlabels2.add(weight_label(l1_x, iy, l2_x, ny, w, col))

        self.play(LaggedStart(*[Create(e) for e in edges2],   lag_ratio=0.05))
        self.play(LaggedStart(*[FadeIn(l) for l in wlabels2], lag_ratio=0.05))

        b2_labels = VGroup(*[
            MathTex(f"+{b}", font_size=16, color=YELLOW).next_to(l2_nodes[i], UP, buff=0.05)
            for i, b in enumerate(biases2)
        ])
        self.play(LaggedStart(*[FadeIn(l) for l in b2_labels], lag_ratio=0.2))
        self.wait(0.4)

        # edges: layer 2 → output
        out_edges = VGroup(*[
            Arrow(
                [l2_x + node_r, y, 0], [output_x - node_r, y, 0],
                buff=0, stroke_width=2, color=ORANGE,
                max_tip_length_to_length_ratio=0.1,
            )
            for y in hidden_ys
        ])
        self.play(LaggedStart(*[GrowArrow(e) for e in out_edges], lag_ratio=0.2))

        # final output values
        out_vals = VGroup(*[
            MathTex(f"{v:.3f}", font_size=18, color=ORANGE).next_to(out_nodes[i], RIGHT, buff=0.12)
            for i, v in enumerate(layer2_out)
        ])
        self.play(LaggedStart(*[Write(l) for l in out_vals], lag_ratio=0.2))
        self.wait(0.5)

        # formula
        formula = MathTex(
            r"\mathbf{o} = W_2 \cdot (W_1 \cdot \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2",
            font_size=26, color=WHITE,
        ).to_edge(DOWN, buff=0.35)
        self.play(Write(formula))
        self.wait(1.5)

        # sample label
        sample_tex = MathTex(
            r"\mathbf{x} = [1.0,\ 2.0,\ 3.0,\ 2.5]",
            font_size=20, color=GREY_B,
        ).next_to(formula, UP, buff=0.18)
        self.play(FadeIn(sample_tex))
        self.wait(0.5)

        # flash
        self.play(
            *[Flash(out_nodes[i], color=ORANGE, flash_radius=0.55) for i in range(3)],
            run_time=1.2,
        )
        self.wait(1.5)
