def gather_all(x):
    # 멀티GPU가 아니면 그대로 반환(필요시 DDP 집계로 대체)
    return x