size_t Binom::HashPair::operator()(const Pair& p) const
{
    size_t hash1 = std::hash<Index>{}(p.first);
    size_t hash2 = std::hash<Index>{}(p.second);
    return hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
}

Index Binom::operator()(Index n, Index k)
{
    if (k > n) return 0;

    Pair p = {n, k};

    auto it = memo.find(p);

    if (it == memo.end())
    {
        Index a = 1, b = 1;

        for (Index i = n; i >= n-k+1; --i)
            a *= i;

        for (Index i = k; i >= 1; --i)
            b *= i;

        a /= b;

        memo.insert({p, a});

        return a;
    }
    else
    {
        return it->second;
    }
}

Index Binom::factorial(Index n) const
{
    if (n < 0) return 0;
    else if (n == 0 || n == 1) return 1;
    else
    {
        Index f = n;

        for (Index i = 2; i < n; ++i)
            f *= i;

        return f;
    }
}
