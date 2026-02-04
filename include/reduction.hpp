RipsFiltration::RipsFiltration() {}

RipsFiltration::RipsFiltration(const std::vector<Simplex>& simplices, Index num_verts) : simplices(simplices), num_verts(num_verts) {}

void RipsFiltration::read_file(const char *fname)
{
    std::ifstream is;
    std::string line;

    Index id;
    Real value;
    int interior;
    Index n = -1;

    is.open(fname, std::ios::in);

    while (std::getline(is, line))
    {
        Index verts;
        sscanf(line.c_str(), "%f\t%lld\t%lld\t%d", &value, &id, &verts, &interior);
        simplices.emplace_back(id, value, interior);

        if (n != -1)
        {
            assert((n == verts));
        }
        else
        {
            n = verts;
        }
    }

    is.close();

    num_verts = n;
}

BoundaryMatrix::BoundaryMatrix(const RipsFiltration& filt) :
    pivots(filt.size(), -1),
    columns(filt.size()),
    sorted_id_to_value(filt.size()),
    sorted_id_to_dim(filt.size())
{
    Index n = filt.size();

    for (Index i = 0; i < n; ++i)
    {
        Simplex simplex = filt[i];

        id_to_sorted_id.insert({simplex.getid(), i});
        sorted_id_to_value[i] = simplex.getvalue();
        sorted_id_to_dim[i] = simplex.getdim();
    }

    for (Index i = 0; i < n; ++i)
    {
        Simplex simplex = filt[i];
        IndexVector facet_ids;

        if (simplex.getdim() == 0)
            continue;

        simplex.get_facet_ids(facet_ids, filt.num_vertices());
        std::for_each(facet_ids.begin(), facet_ids.end(), [&](Index& id) { id = id_to_sorted_id.at(id); });
        std::sort(facet_ids.begin(), facet_ids.end());
        columns[i] = facet_ids;
    }
}

Index BoundaryMatrix::low(Index col) const
{
    return columns[col].empty() ? -1 : columns[col].back();
}

void BoundaryMatrix::add_column(Index left, Index right)
{
    IndexVector result;

    Index n = columns[left].size();
    Index m = columns[right].size();

    const Index *u = columns[left].data();
    const Index *v = columns[right].data();

    Index i = 0, j = 0;

    while (i < n && j < m)
    {
        if      (u[i] < v[j]) result.push_back(u[i++]);
        else if (u[i] > v[j]) result.push_back(v[j++]);
        else { i++, j++; }
    }

    while (i < n) result.push_back(u[i++]);
    while (j < m) result.push_back(v[j++]);

    std::swap(result, columns[right]);
}

void BoundaryMatrix::reduce()
{
    Index n = pivots.size();

    for (Index j = 0; j < n; ++j)
    {
        Index l;

        while ((l = low(j)) >= 0)
        {
            Index i = pivots[l];

            if (i == -1)
            {
                pivots[l] = j;
                break;
            }
            else
            {
                add_column(i, j);
            }
        }
    }
}

void BoundaryMatrix::write_homology_persistence(FILE *f) const
{
    Index n = pivots.size();

    for (Index i = 0; i < n; ++i)
    {
        if (low(i) == -1)
        {
            Index birth_index = i;
            Index death_index = pivots[birth_index];

            Real birth_value = sorted_id_to_value[birth_index];
            Index birth_dim = sorted_id_to_dim[birth_index];

            if (death_index < 0)
            {
                fprintf(f, "%lld: (%.6f, inf)\n", birth_dim, birth_value);
            }
            else
            {
                Real death_value = sorted_id_to_value[death_index];
                Index death_dim = sorted_id_to_dim[death_index];

                if (birth_value != death_value)
                {
                    fprintf(f, "%lld: (%.6f, %.6f)\n", birth_dim, birth_value, death_value);
                }
            }
        }
    }
}

void BoundaryMatrix::write_cohomology_persistence(FILE *f) const
{
    Index n = pivots.size();

    IndexSet paired;

    for (Index j = 0; j < n; ++j)
    {
        if (low(j) == -1)
            continue;

        Index birth_index = j;
        Index death_index = low(j);

        paired.insert(death_index);

        Real birth_value = sorted_id_to_value[n-1-birth_index];
        Real death_value = sorted_id_to_value[n-1-death_index];

        Index birth_dim = sorted_id_to_dim[n-1-birth_index];

        if (birth_value != death_value)
        {
            fprintf(f, "%lld: (%.6f, %.6f)\n", birth_dim, birth_value, death_value);
        }
    }

    for (Index j = 0; j < n; ++j)
    {
        if (low(j) == -1 && paired.count(j) == 0)
        {
            Real birth_value = sorted_id_to_value[n-1-j];
            Index birth_dim = sorted_id_to_dim[n-1-j];

            fprintf(f, "%lld: (%.6f, inf)\n", birth_dim, birth_value);
        }
    }
}

void BoundaryMatrix::antitranspose()
{
    Index n = pivots.size();
    std::vector<IndexVector> new_columns(n);

    for (Index j = 0; j < n; ++j)
    {
        const IndexVector& col = columns[j];

        for (Index i : col)
        {
            new_columns[n-1-i].push_back(n-1-j);
        }
    }

    for (auto& col : new_columns) std::sort(col.begin(), col.end());

    std::swap(columns, new_columns);
}
