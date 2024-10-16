from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

class MaxSAT_Validator():

    def __init__(self, file):
        self.hardconst_file = file
    

    def correct_v1(self, output, threshold=0.3):
        hardconst_formula = WCNF(self.hardconst_file)
        for i in range(len(output)):
            if output[i] > threshold:
                hardconst_formula.append([i+1], weight=output[i])
            else:
                hardconst_formula.append([-i-1], weight=1-output[i])
        with RC2(hardconst_formula) as rc2:
            model = rc2.compute()
        true_answer = []
        for i in model:
            if i > 0:
                true_answer.append(i-1)
        return true_answer