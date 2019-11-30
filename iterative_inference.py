# iterative_inference.py
# NN inference in an iterative manner, instead of a forward single shot.

class IINN(object):
    def __init__(self, dim_x, dim_y):
        pass
    def attention(self, x, y):
        pass
    def inference(self, x, a):
        pass
    def getInputPlaceHolder(self):
        pass
    def getFeedbackPlaceHolder(self):
        pass
    def getOutputTensor(self):
        pass


def Build_IINN(n_class):
    dim_x = [None, None, None, 3]
    dim_y = [n_class]
    return IINN(dim_x, dim_y)



if __name__ == "__main__":
    n_class = 10
    iinn_ = Build_IINN(n_class)
    y_trivial = np.ones(n_class) # start from a trivial solution
    a = iinn_.attention(x, y_trivial)
    y = iinn_.inference(x, a)
    a = iinn_.attention(x, y)
    y = iinn_.inference(x, a)
    # ... this procedure goes on and on until converged
    pass