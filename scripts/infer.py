
import json, random

N = 3
random.seed(42)

def matmul(W, x):
    return [sum(W[i][j]*x[j] for j in range(N)) for i in range(N)]

W = [[random.randint(0, 10) for _ in range(N)] for _ in range(N)]
x = [random.randint(0, 10) for _ in range(N)]
y = matmul(W, x)

payload = {"w": W, "x": x, "y": y}
print(json.dumps(payload, indent=2))
