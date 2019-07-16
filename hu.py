# The [Hu] system
# The similar philosophy but different approach of GAN.

# A revivable self-supervised Learning system
# The theory contains two:
# 1. The Hu system consists of two agents: the first agent try to provide a good initial solution for the second
#    to optimize until the solution meets the specified requirement. Meanwhile, the second agent uses this
#    optimized solution to train the first agent to help it provide a better initial solution for the next time.
#    In such an iterative system, Agent A and B helps each other to work better, and together they reach the
#    overall target as good and faster as possible.
# 2. A revivable training manner: each time the model is trained to Nash Equilibrium, fixed those non-sparse connections
#    (for example, weighted kernels or filters), and reinitialize the sparse connections, and train again with new
#    iteration between the two agents. Thus the learnt parameters to curve the distribution of samples during each
#    equilibrium round will forward onto the next generation and keeps the learning process a spiral up, instead of
#    unordered and easily corrupted training of GANs.


class Hu(object):
    def __init__(self):
        self.