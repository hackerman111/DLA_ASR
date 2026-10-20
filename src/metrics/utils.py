def levi_dist(target_text, predicted_text) -> float:
    """
    Растояние Левинштейна. Переписать на более эффективную версию
    """
    n, m = len(target_text), len(predicted_text)

    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + int(target_text[i - 1] != predicted_text[j - 1]),
            )
    return dp[n][m]


def calc_cer(target_text, predicted_text) -> float:
    """
    Доктесты

    >>> calc_cer("hello", "hello")
    0.0

    >>> calc_cer("hello", "hallo")
    0.2

    >>> calc_cer("hello", "hell")
    0.2

    >>> calc_cer("hello", "helloo")
    0.2

    >>> calc_cer("hello", "helloabc")
    0.6

    >>> calc_cer("hello", "")
    1.0

    >>> calc_cer("abc", "xyz")
    1.0

    >>> calc_cer("", "")
    0.0

    >>> calc_cer("a", "abc")
    2.0

    >>> calc_cer("a b", "ab")
    0.3333333333333333
    """
    n, m = len(target_text), len(predicted_text)
    if n == 0:
        if m == 0:
            return 0.0
        else:
            return float("inf")
    return levi_dist(target_text, predicted_text) / n


def calc_wer(target_text, predicted_text) -> float:
    """
    Доктестики

    >>> calc_wer("hello world", "hello world")
    0.0

    >>> calc_wer("hello world", "hello there")
    0.5

    >>> calc_wer("hello world", "hello")
    0.5

    >>> calc_wer("hello world", "hello beautiful world")
    0.5

    >>> calc_wer("hello", "hello beautiful world")
    2.0

    >>> calc_wer("hello world", "")
    1.0

    >>> calc_wer("", "")
    0.0
    """
    split_target = target_text.split(" ")
    split_pred = predicted_text.split(" ")
    n, m = len(split_target), len(split_pred)
    if n == 0:
        if m == 0:
            return 0.0
        else:
            return float("inf")
    return levi_dist(split_target, split_pred) / n
