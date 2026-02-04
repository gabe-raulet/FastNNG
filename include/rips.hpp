
Simplex::Simplex() : id(0), interior(0) {}

Simplex::Simplex(Index id) : id(id), interior(0) {}

Simplex::Simplex(const IndexVector& verts) : interior(0)
{
    IndexVector vertices(verts);
    std::sort(vertices.begin(), vertices.end());

    uint64_t p = vertices.size()-1;
    uint64_t uid = 0;

    for (Index i = p; i >= 0; --i)
    {
        uid += binom(vertices[i], i+1);
    }

    id = static_cast<Index>(uid | (p << 60));
}

Index Simplex::getid() const
{
    return id;
}

Index Simplex::getuid() const
{
    uint64_t _id = id;
    uint64_t _uid = _id & 0xFFFFFFFFFFFFFFF;
    return _uid;
}

Index Simplex::getdim() const
{
    uint64_t _dim;
    uint64_t _id = id;

    _dim = (_id >> 60) & 0xF;

    return _dim;
}

IndexVector Simplex::getverts(Index n) const
{
    auto get_max_vertex = [&n](size_t uid, size_t d)
    {
        int64_t left = 0;
        int64_t right = n;
        int64_t i;

        while (left < right)
        {
            i = left + ((right-left)>>1);

            if (binom(i,d+1) > uid)
                right = i;
            else
                left = i+1;
        }

        return right-1;
    };

    size_t uid = getuid();
    size_t dim = getdim();
    IndexVector vertices;

    for (Index i = dim; i >= 0; --i)
    {
        int64_t l = get_max_vertex(uid, i);
        vertices.push_back(l);
        uid -= binom(l, i+1);
    }

    if (vertices.size() > 1)
        std::sort(vertices.begin(), vertices.end());

    return vertices;
}

void Simplex::get_facets(std::vector<Simplex>& facets, Index n) const
{
    facets.clear();

    IndexVector vertices = getverts(n);
    Index dim = vertices.size()-1;

    for (Index i = 0; i <= dim; ++i)
    {
        IndexVector facet(vertices);

        for (Index j = i; j < dim; ++j)
            facet[j] = facet[j+1];

        facet.pop_back();
        facets.emplace_back(facet);
    }
}

void Simplex::get_facet_ids(IndexVector& ids, Index n) const
{
    ids.clear();

    std::vector<Simplex> facets;
    get_facets(facets, n);

    for (const auto& s : facets)
    {
        ids.push_back(s.getid());
    }
}

void Simplex::reindex(const IndexVector& indices, Index n)
{
    IndexVector verts = getverts(n);
    for (Index& v : verts) v = indices[v];

    std::sort(verts.begin(), verts.end());

    uint64_t p = verts.size()-1;
    uint64_t uid = 0;

    for (Index i = p; i >= 0; --i)
    {
        uid += binom(verts[i], i+1);
    }

    id = static_cast<Index>(uid | (p << 60));
}

/* void RipsComplex::bron_kerbosch(IndexVector& current, const IndexVector& cands, Index excluded, const NeighborListVector& graph, NeighborListVector& weights, Index maxdim) */
/* { */
    /* if (!current.empty()) */
    /* { */
        /* Index p = current.size()-1; */
        /* simplices.emplace_back(current); */

        /* const Simplex& sigma = simplices.back(); */

        /* if (p == 0) weights[0].insert({sigma.getid(), 0.}); */
        /* else if (p == 1) weights[1].insert({sigma.getid(), graph[current[0]].find(current[1])->second}); */
        /* else weights[p].insert({sigma.getid(), 0.}); */
    /* } */

    /* if (current.size() == static_cast<size_t>(maxdim) + 1) */
        /* return; */

    /* Index m = cands.size(); */

    /* for (Index j = excluded+1; j < m; ++j) */
    /* { */
        /* current.push_back(cands[j]); */

        /* IndexVector new_cands; */

        /* for (Index i = 0; i < j; ++i) */
        /* { */
            /* if (graph[cands[i]].find(cands[j]) != graph[cands[i]].end()) */
                /* new_cands.push_back(cands[i]); */
        /* } */

        /* Index ex = new_cands.size(); */

        /* for (Index i = j+1; i < m; ++i) */
        /* { */
            /* if (graph[cands[i]].find(cands[j]) != graph[cands[i]].end()) */
                /* new_cands.push_back(cands[i]); */
        /* } */

        /* excluded = ex-1; */

        /* bron_kerbosch(current, new_cands, excluded, graph, weights, maxdim); */
        /* current.pop_back(); */
    /* } */
/* } */

/* RipsComplex::RipsComplex(const Graph& skeleton, Index maxdim) : num_vertices(skeleton.num_vertices()) */
/* { */
    /* NeighborListVector graph(num_vertices); */
    /* Index num_edges = skeleton.my_num_edges(); */

    /* for (Index i = 0; i < num_edges; ++i) */
    /* { */
        /* const auto& [u, v, dist] = skeleton[i]; */
        /* graph[u].insert({v, dist}); */
    /* } */

    /* NeighborListVector weights(maxdim+1); */

    /* IndexVector current; */
    /* IndexVector candidates(num_vertices); */

    /* std::iota(candidates.begin(), candidates.end(), (Index)0); */

    /* bron_kerbosch(current, candidates, -1, graph, weights, maxdim); */

    /* for (Index p = 2; p <= maxdim; ++p) */
    /* { */
        /* for (auto& [id, weight] : weights[p]) */
        /* { */
            /* weight = 0; */
            /* Simplex sigma(id); */

            /* IndexVector facet_ids; */
            /* sigma.get_facet_ids(facet_ids, num_vertices); */

            /* for (Index fid : facet_ids) */
            /* { */
                /* weight = std::max(weight, weights[p-1][fid]); */
            /* } */
        /* } */
    /* } */

    /* for (auto& s : simplices) */
    /* { */
        /* Index id = s.getid(); */
        /* Index dim = s.getdim(); */

        /* s.value = weights[dim][id]; */
    /* } */

    /* std::sort(simplices.begin(), simplices.end()); */
/* } */

std::string Simplex::repr(Index n) const
{
    IndexVector verts = getverts(n);
    Index size = verts.size();

    std::stringstream ss;
    ss << "<";

    for (Index i = 0; i < size-1; ++i)
    {
        ss << verts[i] << ",";
    }

    ss << verts[size-1] << ">";
    return ss.str();
}
