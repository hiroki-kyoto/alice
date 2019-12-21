# Alice
Target: Creating the first **REAL AI** with self consciousness.

Routine: 
1. Unsupervised Learning by free access to tons of resources in the air.
2. Supervised Learning by proactive feedback observation. 
3. Only one objective is given: survival.

Main ideas:
1. AI is more like a **CLOSE LOOP CONTROL**, with iterative adjustment 
during each inference, other than an open loop 
single shot processor.
2. AI learns generic knowledge from various observation, 
the goal for it is to figure out the **THEORIES THAT DO NOT CHANGE**
3. Effective learning requires generation iteration and fundamental 
neural form evolution. 
Like Ticket Hypothesis, **TWO TIME-SCALEs** are required in learning,
the coarse scale learns the basic graph linkage,
the fine scale learns the extract weights for 
**transferring environment**.
4. Efficient learning requires visualized and explainable analysis.
Such as, given a classifier, it tells the label of an test image.
Let $x$ be the test image, and $f(\cdot)$ be the classifier equivalent
function.
In a traditional case, we know the ground truth label is $y$.
Thus learning is often described as,
to minimize the metric $$ || f(x) - y || $$.
However, in our new case, we are dealing more human alike.
We're not only telling the truth label, but also giving hints for 
which the cost occurs.
We introduce a generator $g(\cdot)$ as a network that reproduce an
image using the given label and subsequent detail control latent
features extracted in layers closer to input $x$.
We also introduce a metric learning module $h(x)$, it tells the 
distance between two input example.
Net $g$ can be pretrained by an auto encoder style, and
net $h$ can be pretrained by inter-class and intra-class image
pairs.
In fact, $g$ and $h$ is actually a GAN model.
And our aim is to minimize the overall cost,
$$|| f(x) - y || + \lambda \cdot h(g(f(x), z), x)$$

