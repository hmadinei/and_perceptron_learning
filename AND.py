import numpy as np

def activation (net):
    if net >= 0:
        return 1
    elif net < 0:
        return 0

pattern = [[0,0], [0,1], [1,0], [1,1]]
p = np.asarray(pattern)
y_true = [0, 0, 0, 1]
# rendon weight 
w = np.random.rand(1,3) * 10
w1 = np.round(w[0][0], 1)
w2 = np.round(w[0][1], 1)
w3 = np.round(w[0][2], 1)
error = np.zeros(10)
learning_rate = 0.05
# modify the weights until error == 0
while True:
    for i in range(len(p)):
        y_act = w1 * p[i][0] + w2 * p[i][1] + w3 
        y_act = activation(y_act)
        error[i] = y_true[i] - y_act 
        if error[i] != 0 :
            # modifying the weights 
            w1 = w1 + learning_rate * error[i] * p[i][0]
            w2 = w2 + learning_rate * error[i] * p[i][1]
            w3 = w3 + learning_rate * error[i] 
    if np.all((error == 0)): 
        # final weights 
        print([[w1, w2, w3]])
        break