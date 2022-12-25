from manim import *
import numpy as np
from scipy.linalg import svd
import imageio

class SVDExample(LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            include_background_plane = False,
            show_coordinates = False,
            leave_ghost_vectors = True,
            show_basis_vectors = False
        )
    def construct(self):

        # SVD decomposition
        A = [[1, 1], [0, 2]]
        U, S, Vt = svd(A)
        S = np.diag(S)
        V = Vt.transpose()

        Atext = MathTex(r"\mathbf{A} = \begin{bmatrix}" + str(A[0][0]) + r"&" + str(A[0][1]) + r"\\" + str(A[1][0]) + r"&" + str(A[1][1]) + r"\end{bmatrix}",
                        color=RED).to_edge(UL).add_background_rectangle()
        Utext = MathTex(r"\mathbf{U} = \begin{bmatrix}" + str(round(U[0][0], 2)) + r"&" + str(round(U[0][1], 2)) + r"\\" + str(round(U[1][0], 2)) + r"&" + str(round(U[1][1], 2)) + r"\end{bmatrix}"
                        ).to_edge(UL).add_background_rectangle()
        Stext = MathTex(r"\mathbf{\Sigma} = \begin{bmatrix}" + str(round(S[0][0], 2)) + r"&" + str(round(S[0][1], 2)) + r"\\" + str(round(S[1][0], 2)) + r"&" + str(round(S[1][1], 2)) + r"\end{bmatrix}",
                        ).to_edge(UL).add_background_rectangle()
        Vtext = MathTex(r"\mathbf{V}^\intercal = \begin{bmatrix}" + str(round(V[0][0], 2)) + r"&" + str(round(V[0][1], 2)) + r"\\" + str(round(V[1][0], 2)) + r"&" + str(round(V[1][1], 2)) + r"\end{bmatrix}^\intercal",
                        ).to_edge(UL).add_background_rectangle()

        # Description of linear transformations
        Amaptext = MathTex(r"\mathbf{x} \mapsto \mathbf{Ax}").next_to(Atext, DOWN)
        Vmaptext = MathTex(r"\mathbf{x} \mapsto \mathbf{V^\intercal x}").next_to(Vtext, DOWN)
        Smaptext = MathTex(r"\mathbf{V^\intercal x} \mapsto \mathbf{\Sigma V^\intercal x}").next_to(Stext, DOWN)
        Umaptext = MathTex(r"\mathbf{\Sigma V^\intercal x} \mapsto \mathbf{U \Sigma V^\intercal x}").next_to(Utext, DOWN)
        
        # Ellipse
        ellipse = Ellipse(width=3, height=3, color=RED, stroke_width=5)
        ellipse.set_fill(RED, opacity=0.4)

        # Basis vectors
        basis1 = self.get_vector(V[:,0])
        basis2 = self.get_vector(V[:,1])

        basis1.color, basis2.color = RED, GREEN
        basis1.stroke_width, basis2.stroke_width = 5, 5
        
        # Transformation A
        r = 1.3
        self.add_transformable_mobject(ellipse, basis1, basis2)

        self.wait()

        self.play(Write(Atext), Write(Amaptext))
        self.moving_mobjects = []
        self.apply_matrix(A, run_time=r)

        self.wait()

        # Undo transformation A
        self.play(FadeOut(Atext), FadeOut(Amaptext))
        self.moving_mobjects = []
        self.apply_inverse(A, run_time=0.5)
        
        # Apply SVD one by one
        w = 0.3

        self.play(Write(Vtext), Write(Vmaptext))
        self.wait(w)
        self.moving_mobjects = []
        self.apply_matrix(Vt, run_time=r)

        self.play(Write(Stext), FadeOut(Vtext), ReplacementTransform(Vmaptext, Smaptext))
        self.wait(w)
        self.moving_mobjects = []
        self.apply_matrix(S, run_time=r)

        self.play(Write(Utext), FadeOut(Stext), ReplacementTransform(Smaptext, Umaptext))
        self.wait(w)
        self.moving_mobjects = []
        self.apply_matrix(U, run_time=r)

        self.wait()

        self.play(FadeOut(Utext), FadeOut(Umaptext))
        self.moving_mobjects = []
        self.apply_inverse(A, run_time=0.5)

        self.wait()

class ImageChannels(Scene):
    def construct(self):

        # Split into RGB channels
        img = imageio.imread(r"tiger.jpeg")

        A_img = np.asarray(img)

        A_red = np.zeros(A_img.shape)
        A_green = np.zeros(A_img.shape)
        A_blue = np.zeros(A_img.shape)

        A_red[:,:,0] = A_img[:,:,0]
        A_green[:,:,1] = A_img[:,:,1]
        A_blue[:,:,2] = A_img[:,:,2]

        A_img = ImageMobject(A_img)
        A_red = ImageMobject(A_red).scale(0.5).to_edge(DL)
        A_green = ImageMobject(A_green).scale(0.5).to_edge(DOWN)
        A_blue = ImageMobject(A_blue).scale(0.5).to_edge(DR)

        self.play(FadeIn(A_img), run_time=0.5)
        self.play(A_img.animate.to_edge(UP).scale(0.5))
        self.wait(0.3)
        animations = [
            FadeIn(A_red, target_position=UP),
            FadeIn(A_green, target_position=UP),
            FadeIn(A_blue, target_position=UP)
        ]

        self.play(AnimationGroup(*animations, lag_ratio=0.2))
        self.wait()

class SVDOuterProduct(Scene):
    def construct(self):

        X_img = np.asarray(imageio.imread("tiger.jpeg"))

        for r in range(490,500):
            rgb = [X_img[:,:,i] for i in range(3)]
            for i in range(3):
                U, S, Vt = np.linalg.svd(rgb[i])
                S = np.diag(S)
                rgb[i] = U[:,:r] @ S[:r,:r] @ Vt[:r,:]
            X_comp = np.dstack((rgb[0], rgb[1], rgb[2]))
            X_comp_mobj = ImageMobject(np.uint8(X_comp)).scale(1.5)
            self.add(X_comp_mobj)
            self.wait(0.5)