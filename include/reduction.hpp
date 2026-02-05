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

    std::sort(simplices.begin(), simplices.end());
}

CoboundaryMatrix::CoboundaryMatrix(const RipsFiltration& filt) : simplices(filt.simplices), num_vertices(filt.num_vertices())
{
    Index num_columns = filt.size();

    columns.resize(num_columns);
    sorted_id_to_value.resize(num_columns);

    for (Index i = 0; i < num_columns; ++i)
    {
        id_to_sorted_id.insert({simplices[i].getid(), i});
        sorted_id_to_value[i] = simplices[i].getvalue();
    }

    for (Index i = 0; i < num_columns; ++i)
    {
        IndexVector facet_ids;

        if (simplices[i].getdim() == 0)
            continue;

        simplices[i].get_facet_ids(facet_ids, num_vertices);
        std::for_each(facet_ids.begin(), facet_ids.end(), [&](Index& id) { id = id_to_sorted_id.at(id); });
        std::sort(facet_ids.begin(), facet_ids.end());

        for (auto it = facet_ids.rbegin(); it != facet_ids.rend(); ++it)
        {
            Index j = num_columns - 1 - *it;
            columns[j].push_back(num_columns-1-i);
        }
    }

    std::reverse(simplices.begin(), simplices.end());

    for (auto& col : columns) std::reverse(col.begin(), col.end());
}

void CoboundaryMatrix::reduce()
{
    Index nrows = num_rows();
    Index ncols = num_cols();

    IndexVector pivots(nrows, -1);
    std::set<Index> addition_cache;

    for (Index j = 0; j < ncols; ++j)
    {
        IndexVector& reduced_col = columns[j];

        addition_cache.clear();
        addition_cache.insert(reduced_col.cbegin(), reduced_col.cend());

        while (!addition_cache.empty())
        {
            Index& pivot = pivots[*addition_cache.rbegin()];

            if (pivot == -1)
            {
                pivot = j;
                reduced_col.assign(addition_cache.begin(), addition_cache.end());
                break;
            }
            else
            {
                for (Index i : columns[pivot])
                {
                    auto insertion_result = addition_cache.insert(i);

                    if (!insertion_result.second)
                    {
                        addition_cache.erase(insertion_result.first);
                    }
                }
            }
        }

        if (addition_cache.empty())
        {
            reduced_col.clear();
        }
    }

}

void CoboundaryMatrix::print_persistence() const
{
    IndexSet paired;
    Index ncols = num_cols();

    for (Index j = 0; j < ncols; ++j)
    {
        if (columns[j].empty())
            continue;

        Index birth_index = j;
        Index death_index = low(j);

        paired.insert(death_index);

        Simplex birth_simplex = simplices[birth_index];
        Simplex death_simplex = simplices[death_index];

        if (birth_simplex.value != death_simplex.value)
        {
            std::string birth_string = birth_simplex.fullrepr(num_vertices);
            std::string death_string = death_simplex.fullrepr(num_vertices);
            printf("(%.5f, %.5f), simplices %s <-> %s\n", birth_simplex.value, death_simplex.value, birth_string.c_str(), death_string.c_str());
        }
    }

    for (Index j = 0; j < ncols; ++j)
    {
        if (columns[j].empty() && paired.count(j) == 0)
        {
            Index birth_index = j;
            Simplex birth_simplex = simplices[birth_index];
            std::string birth_string = birth_simplex.fullrepr(num_vertices);

            printf("(%.5f, inf), birth simplex %s\n", birth_simplex.value, birth_string.c_str());
        }
    }
}
