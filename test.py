def savings(m: float, rate: float, n: int) -> float:
    """
    計算每年末存 m 元、年利率 rate、存 n 年後的本利和（遞迴版）。
    
    參數：
    m    -- 每年存入的金額
    rate -- 年利率（例如 0.05 表示 5%）
    n    -- 存款年數
    
    回傳：
    n 年後的本利和
    """
    # 基本情況：存 0 年，沒有任何本利和
    if n <= 0:
        return 0.0
    # 遞迴關係：前年本利和滾利，然後加上今年年末存入的 m 元
    return savings(m, rate, n - 1) * (1 + rate) + m


# 範例
if __name__ == "__main__":
    m = 5000      # 每年存 1000 元
    rate = 0.008333   # 年利率 5%
    n = 180        # 存 10 年
    total = savings(m, rate, n)
    print(f"每年存 {m} 元，年利率 {rate*100:.1f}%，存 {n} 年後本利和 = {total:.2f} 元")
