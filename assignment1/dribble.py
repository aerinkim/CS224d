import numpy as np



def softmax_grad(s):
	# input s is softmax value of the original input x. Its shape is (1,n) 
	# e.i. s = np.array([0.3,0.7]), x = np.array([0,1])

	# make the matrix whose size is n^2.
	jacobian_m = np.diag(s)

	for i in range(len(jacobian_m)):
		for j in range(len(jacobian_m)):
			if i == j:
				jacobian_m[i][j] = s[i] * (1-s[i])
			else: 
				jacobian_m[i][j] = -s[i]*s[j]

	return jacobian_m







SM = self.value.reshape((-1,1))
jac = np.diag(self.value) - np.dot(SM, SM.T)




# what does softmax gradient mean?
# It doesn't even have a graph!


    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                self.gradient[i] = self.value[i] * (1-self.input[i))
            else: 
                 self.gradient[i] = -self.value[i]*self.input[j]
