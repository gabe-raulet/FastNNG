#ifndef BINOM_H_
#define BINOM_H_

#include "utils.h"

class Binom
{
    public:

        using Pair = std::pair<Index, Index>;

        struct HashPair
        {
            size_t operator()(const Pair& p) const;
        };

        using Memo = std::unordered_map<Pair, Index, HashPair>;

        inline Binom();

        Index operator()(Index n, Index k);

    private:

        Memo memo;

        Index factorial(Index n) const;
};

#include "binom.hpp"

#endif
