import json, random

N = 3
L = 2
random.seed(42)

def matmul(W, x):
    return [sum(W[i][j]*x[j] for j in range(N)) for i in range(N)]

def forward(weights, x, num, den, qmin, qmax):
    v = x[:]
    for l in range(L):
        t = matmul(weights[l], v)
        # quantized requant: y = floor( (t * num)/den ); assume no clipping for POC
        v = [ (ti * num) // den for ti in t ]
        # (optional clip) v = [ min(max(qmin, yi), qmax) for yi in v ]
    return v

# small ints for POC
weights = [ [[random.randint(0, 10) for _ in range(N)] for _ in range(N)] for _ in range(L) ]
x = [random.randint(0, 10) for _ in range(N)]
scale_num = 3
scale_den = 2
qmin = -128
qmax = 127
y = forward(weights, x, scale_num, scale_den, qmin, qmax)

payload = {
    "w": weights,
    "x": x,
    "y": y,
    "scale_num": scale_num,
    "scale_den": scale_den,
    "qmin": qmin,
    "qmax": qmax
}
print(json.dumps(payload, indent=2))