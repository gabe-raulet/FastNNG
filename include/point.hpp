#include "search.h"

template <class Atom_>
PointContainer<Atom_>::PointContainer() : offsets({0}) {}

template <class Atom_>
PointContainer<Atom_>::PointContainer(const AtomVector& atoms, const IndexVector& sizes) : data(atoms), offsets(sizes.size()+1)
{
    Index disp = 0;
    Index point_count = sizes.size();

    for (Index i = 0; i < point_count; ++i)
    {
        offsets[i] = disp;
        disp += sizes[i];
    }

    offsets[point_count] = disp;
}

template <class Atom_>
PointContainer<Atom_>::PointContainer(const AtomVector& atoms, Index size, Index dim) : data(atoms), offsets(size+1)
{
    Index disp = 0;

    for (Index i = 0; i <= size; ++i)
    {
        offsets[i] = disp;
        disp += dim;
    }
}

template <class Atom_>
PointContainer<Atom_>::PointContainer(const std::vector<const Atom*>& atoms, const IndexVector& sizes) : offsets(sizes.size()+1)
{
    Index atom_count = 0;
    Index size = sizes.size();

    for (Index i = 0; i < size; ++i)
    {
        offsets[i] = atom_count;
        atom_count += sizes[i];
    }

    offsets[size] = atom_count;

    data.reserve(atom_count);

    for (Index i = 0; i < size; ++i)
    {
        data.insert(data.end(), atoms[i], atoms[i]+sizes[i]);
    }
}

template <class Atom_>
PointContainer<Atom_>::PointContainer(const PointContainer& lhs, const PointContainer& rhs) : data(lhs.num_atoms() + rhs.num_atoms()), offsets(lhs.num_points() + rhs.num_points() + 1)
{
    auto it = data.begin();

    it = std::copy(lhs.data.begin(), lhs.data.end(), it);
    it = std::copy(rhs.data.begin(), rhs.data.end(), it);

    Index left_count = lhs.num_points();
    Index right_count = rhs.num_points();

    for (Index i = 0; i < left_count; ++i)
    {
        offsets[i] = lhs.offsets[i];
    }

    for (Index i = 0; i <= right_count; ++i)
    {
        offsets[i+left_count] = rhs.offsets[i] + lhs.offsets[left_count];
    }
}

template <class Atom_>
Index PointContainer<Atom_>::num_points() const
{
    return offsets.size()-1;
}

template <class Atom_>
Index PointContainer<Atom_>::num_atoms() const
{
    return data.size();
}

template <class Atom_>
const Atom_* PointContainer<Atom_>::mem(Index i) const
{
    return &data[offsets[i]];
}

template <class Atom_>
Index PointContainer<Atom_>::size(Index i) const
{
    return offsets[i+1]-offsets[i];
}

template <class Atom_>
Index PointContainer<Atom_>::read_fvecs(const char *fname)
{
    assert((std::same_as<Atom, float>));

    std::ifstream is;
    size_t filesize, total;
    std::vector<char> p;
    int d;

    is.open(fname, std::ios::binary | std::ios::in);

    is.seekg(0, is.end);
    filesize = is.tellg();
    is.seekg(0, is.beg);

    is.read((char*)&d, sizeof(int));
    is.seekg(0, is.beg);

    size_t point_size = sizeof(Atom)*d;
    size_t record_size = sizeof(int) + point_size;

    assert((filesize % record_size == 0));
    total = filesize / record_size;

    p.resize(record_size);

    data.resize(total*d);
    offsets.resize(total+1);

    Index disp = 0;

    for (size_t i = 0; i < total; ++i)
    {
        is.read(p.data(), record_size);

        const char *ds = p.data();
        const char *ps = p.data() + sizeof(int);

        int dt;
        memcpy(&dt, ds, sizeof(int)); assert((dt == d));

        char *dest = (char*)(&data[i*d]);
        memcpy(dest, ps, point_size);

        offsets[i] = disp;
        disp += d;
    }

    is.close();
    offsets[total] = disp;

    return total;
}

template <class Atom_>
Index PointContainer<Atom_>::read_fvecs(const char *fname, MPI_Comm comm)
{
    assert((std::same_as<Atom, float>));

    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    MPI_File fh;
    MPI_Aint extent;
    MPI_Offset filesize, filedisp;
    Index total, myleft, mysize;
    int dim;

    MPI_Datatype MPI_POINT;
    MPI_Datatype MPI_ATOM = MPI_FLOAT;

    MPI_File_open(comm, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    if (!myrank)
    {
        MPI_File_get_size(fh, &filesize);
        MPI_File_read(fh, &dim, 1, MPI_INT, MPI_STATUS_IGNORE);
    }

    MPI_Bcast(&dim, 1, MPI_INT, 0, comm);
    MPI_Bcast(&filesize, 1, MPI_OFFSET, 0, comm);

    extent = 4 * (dim + 1);
    total = filesize / extent;

    assert((filesize % extent == 0));

    mysize = total / nprocs;
    myleft = total % nprocs;

    if (myrank < myleft)
        mysize++;

    IndexVector sizes(nprocs);
    sizes[myrank] = mysize;

    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INDEX, sizes.data(), 1, MPI_INDEX, comm);

    Index totsize;
    Index myoffset;

    MPI_Allreduce(&mysize, &totsize, 1, MPI_INDEX, MPI_SUM, comm);
    MPI_Exscan(&mysize, &myoffset, 1, MPI_INDEX, MPI_SUM, comm);
    if (!myrank) myoffset = 0;

    data.resize(mysize*dim);
    offsets.resize(mysize+1);

    assert((dim >= 1));
    MPI_Type_contiguous(dim, MPI_ATOM, &MPI_POINT);
    MPI_Type_commit(&MPI_POINT);

    MPI_Datatype filetype;
    MPI_Type_create_resized(MPI_POINT, 0, extent, &filetype);
    MPI_Type_commit(&filetype);

    filedisp = myoffset*extent + sizeof(int);
    MPI_File_set_view(fh, filedisp, MPI_POINT, filetype, "native", MPI_INFO_NULL);

    MPI_File_read(fh, data.data(), (int)mysize, MPI_POINT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    MPI_Type_free(&filetype);
    MPI_Type_free(&MPI_POINT);

    Index disp = 0;

    for (Index i = 0; i <= mysize; ++i)
    {
        offsets[i] = disp;
        disp += dim;
    }

    return total;
}

template <class Atom_>
Index PointContainer<Atom_>::read_seqs(const char *fname)
{
    assert((std::same_as<Atom, char>));

    std::ifstream is;
    std::string line;

    Index id = 0;

    is.open(fname, std::ios::in);

    offsets.clear();
    data.clear();

    while (std::getline(is, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        if (line.empty())
            continue;

        offsets.push_back(data.size());
        std::copy(line.begin(), line.end(), std::back_inserter(data));
    }

    offsets.push_back(data.size());
    is.close();

    return offsets.size()-1;
}

template <class Atom_>
Index PointContainer<Atom_>::read_seqs(const char *fname, MPI_Comm comm)
{
    assert((std::same_as<Atom, char>));

    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    MPI_Datatype MPI_ATOM = MPI_CHAR;

    AtomVector alldata;
    IndexVector alloffsets;

    Index size, myoffset, mysize, myleft, atoms;

    if (!myrank)
    {
        PointContainer<Atom> allpoints;
        allpoints.read_seqs(fname);

        AtomVector& alldata_ref = allpoints.getdata();
        IndexVector& alloffsets_ref = allpoints.getoffsets();

        alldata.assign(alldata_ref.begin(), alldata_ref.end());
        alloffsets.assign(alloffsets_ref.begin(), alloffsets_ref.end());

        size = allpoints.num_points();
        atoms = allpoints.num_atoms();
    }

    MPI_Bcast(&size, 1, MPI_INDEX, 0, comm);
    MPI_Bcast(&atoms, 1, MPI_INDEX, 0, comm);

    if (myrank != 0)
    {
        alldata.resize(atoms);
        alloffsets.resize(size+1);
    }

    MPI_Bcast(alldata.data(), static_cast<int>(atoms), MPI_ATOM, 0, comm);
    MPI_Bcast(alloffsets.data(), static_cast<int>(size+1), MPI_INDEX, 0, comm);

    mysize = size/nprocs;
    myleft = size%nprocs;

    if (myrank < myleft)
        mysize++;

    MPI_Exscan(&mysize, &myoffset, 1, MPI_INDEX, MPI_SUM, comm);
    if (!myrank) myoffset = 0;

    data.clear();
    offsets.clear();

    for (Index i = myoffset; i < myoffset+mysize; ++i)
    {
        Index dataoffset = alloffsets[i];
        Index datasize = alloffsets[i+1] - dataoffset;

        auto first = alldata.begin() + dataoffset;
        auto last = first + datasize;

        offsets.push_back(data.size());
        std::copy(first, last, std::back_inserter(data));
    }

    offsets.push_back(data.size());

    return size;
}

template <class Atom_>
typename PointContainer<Atom_>::AtomVector& PointContainer<Atom_>::getdata()
{
    return data;
}

template <class Atom_>
IndexVector& PointContainer<Atom_>::getoffsets()
{
    return offsets;
}

template <class Atom_>
void PointContainer<Atom_>::localgather(const PointContainer& points, const IndexVector& local_indices)
{
    Index size = local_indices.size();
    Index atom_count = 0;

    offsets.resize(size+1);

    for (Index i = 0; i < size; ++i)
    {
        offsets[i] = atom_count;
        atom_count += points.size(local_indices[i]);
    }

    offsets[size] = atom_count;

    data.clear();
    data.reserve(atom_count);

    for (Index i = 0; i < size; ++i)
    {
        const Atom *mem = points.mem(local_indices[i]);
        Index dim = points.size(local_indices[i]);
        data.insert(data.end(), mem, mem+dim);
    }
}

template <class Atom_>
void PointContainer<Atom_>::allgather(const PointContainer& mypoints, MPI_Comm comm)
{
    MPI_Datatype MPI_ATOM = mpi_type<Atom>();

    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    data.clear();
    offsets.clear();

    const AtomVector& sendbuf_atoms = mypoints.data;
    AtomVector& recvbuf_atoms = data;

    Index mysize = mypoints.num_points();

    IndexVector sendbuf_sizes(mysize);
    IndexVector recvbuf_sizes;

    for (Index i = 0; i < mysize; ++i)
    {
        sendbuf_sizes[i] = mypoints.size(i);
    }

    std::vector<int> recvcounts(nprocs), rdispls(nprocs);
    std::vector<int> recvcounts_atoms(nprocs), rdispls_atoms(nprocs);

    recvcounts[myrank] = mysize;
    recvcounts_atoms[myrank] = sendbuf_atoms.size();

    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, recvcounts_atoms.data(), 1, MPI_INT, comm);

    std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), 0);
    std::exclusive_scan(recvcounts_atoms.begin(), recvcounts_atoms.end(), rdispls_atoms.begin(), 0);

    int totrecv = recvcounts.back() + rdispls.back();
    int totrecv_atoms = recvcounts_atoms.back() + rdispls_atoms.back();

    recvbuf_atoms.resize(totrecv_atoms);
    recvbuf_sizes.resize(totrecv);

    MPI_Allgatherv(sendbuf_sizes.data(), recvcounts[myrank], MPI_INDEX, recvbuf_sizes.data(), recvcounts.data(), rdispls.data(), MPI_INDEX, comm);
    MPI_Allgatherv(sendbuf_atoms.data(), recvcounts_atoms[myrank], MPI_ATOM, recvbuf_atoms.data(), recvcounts_atoms.data(), rdispls_atoms.data(), MPI_ATOM, comm);

    offsets.resize(totrecv);

    std::exclusive_scan(recvbuf_sizes.begin(), recvbuf_sizes.end(), offsets.begin(), (Index)0);
    offsets.push_back(offsets.back() + recvbuf_sizes.back());

    assert((offsets.back() == recvbuf_atoms.size()));
}

template <class Atom_>
void PointContainer<Atom_>::push_back(const Atom* point_mem, Index point_size)
{
    std::copy(point_mem, point_mem+point_size, std::back_inserter(data));
    offsets.push_back(data.size());
}

template <class Atom_>
VoronoiCell<Atom_>::VoronoiCell(const PointContainerType& points, const IndexVector& global_indices, const RealVector& dist_to_centers)
    : PointContainerType(points),
      global_indices(global_indices),
      dist_to_centers(dist_to_centers),
      interior(points.num_points(), false) {}

template <class Atom_>
Index VoronoiCell<Atom_>::index(Index i) const
{
    return global_indices[i];
}

template <class Atom_>
Real VoronoiCell<Atom_>::dist_to_center(Index i) const
{
    return dist_to_centers[i];
}

template <class Atom_>
Index VoronoiCell<Atom_>::num_ghosts() const
{
    return ghost_points.num_points();
}

template <class Atom_>
Index VoronoiCell<Atom_>::ghost_index(Index i) const
{
    return global_ghost_indices[i];
}

template <class Atom_>
Index VoronoiCell<Atom_>::ghost_size(Index i) const
{
    return ghost_points.size(i);
}

template <class Atom_>
const typename VoronoiCell<Atom_>::Atom* VoronoiCell<Atom_>::ghost_mem(Index i) const
{
    return ghost_points.mem(i);
}

template <class Atom_>
typename IndexVector::const_iterator VoronoiCell<Atom_>::ids_begin() const
{
    return global_indices.cbegin();
}

template <class Atom_>
typename IndexVector::const_iterator VoronoiCell<Atom_>::ids_end() const
{
    return global_indices.cend();
}

template <class Atom_>
void VoronoiCell<Atom_>::add_ghost_point(const Atom *point_mem, Index point_size, Index point_index)
{
    ghost_points.push_back(point_mem, point_size);
    global_ghost_indices.push_back(point_index);
}

template <class Atom_>
template <class Distance>
void VoronoiCell<Atom_>::find_neighbors(Real cover, Index leaf_size, Distance& distance, Real radius, EdgeVector& myedges) const
{
    CoverTree tree(cover, leaf_size);
    tree.build(*this, distance);

    auto functor = [&](Index neighbor, Index query, Real weight)
    {
        myedges.emplace_back(global_indices[neighbor], global_indices[query], weight);
    };

    auto ghost_functor = [&](Index neighbor, Index query, Real weight)
    {
        myedges.emplace_back(global_indices[neighbor], global_ghost_indices[query], weight);
    };

    tree.radius_query_batched(*this, distance, *this, radius, functor);
    tree.radius_query_batched(*this, distance, ghost_points, radius, ghost_functor);
}

template <class Atom_>
void VoronoiCell<Atom_>::set_interior(Index i)
{
    interior[i] = true;
}

template <class Atom_>
const typename VoronoiCell<Atom_>::PointContainerType VoronoiCell<Atom_>::ghosts() const
{
    return ghost_points;
}

template <class Atom_>
const std::vector<bool> VoronoiCell<Atom_>::interiors() const
{
    return interior;
}

template <class Atom_>
VoronoiComplex<Atom_>::VoronoiComplex(const VoronoiCellType& cell, Index universe_point_count) : points(cell, cell.ghosts()), indices(cell.num_points() + cell.num_ghosts()), interior(cell.interiors()), local(cell.num_points()), total(cell.num_points() + cell.num_ghosts()), universe_point_count(universe_point_count)
{
    Index point_count = cell.num_points();
    Index ghost_count = cell.num_ghosts();

    for (Index i = 0; i < point_count; ++i)
    {
        indices[i] = cell.index(i);
    }

    for (Index i = 0; i < ghost_count; ++i)
    {
        indices[i+point_count] = cell.ghost_index(i);
    }
}

template <class Atom_>
template <class Distance>
void VoronoiComplex<Atom_>::build_filtration(Distance& distance, Real radius, Index maxdim, Real cover, Index leaf_size)
{
    CoverTree tree(cover, leaf_size);
    tree.build(points, distance);

    NeighborListVector graph(total);
    NeighborListVector weights(maxdim+1);

    auto query_functor = [&](Index neighbor, Index query, Real weight)
    {
        graph[neighbor].insert({query, weight});
    };

    tree.radius_query_batched(points, distance, points, radius, query_functor);

    IndexVector current;
    IndexVector candidates(total);

    std::iota(candidates.begin(), candidates.end(), (Index)0);

    bron_kerbosch(current, candidates, -1, graph, weights, maxdim);

    for (Index p = 2; p <= maxdim; ++p)
    {
        for (auto& [id, weight] : weights[p])
        {
            weight = 0;
            Simplex sigma(id);

            IndexVector facet_ids;
            sigma.get_facet_ids(facet_ids, total);

            for (Index facet_id : facet_ids)
            {
                weight = std::max(weight, weights[p-1][facet_id]);
            }
        }
    }

    for (auto& s : simplices)
    {
        Index id = s.getid();
        Index dim = s.getdim();

        s.value = weights[dim][id];

        s.reindex(indices, universe_point_count);
    }

    std::sort(simplices.begin(), simplices.end());
}

template <class Atom_>
void VoronoiComplex<Atom_>::write_filtration_file(const char *fname, bool use_ids) const
{
    FILE *f;

    f = fopen(fname, "w");

    for (const auto& s : simplices)
    {
        if (use_ids)
        {
            fprintf(f, "%f\t%lld\t%d\n", s.value, s.getid(), static_cast<int>(s.interior));
        }
        else
        {
            std::string st = s.repr(universe_point_count);
            fprintf(f, "%f\t%s\t%d\n", s.value, st.c_str(), static_cast<int>(s.interior));
        }
    }

    fclose(f);
}

template <class Atom_>
void VoronoiComplex<Atom_>::bron_kerbosch(IndexVector& current, const IndexVector& cands, Index excluded, const NeighborListVector& graph, NeighborListVector& weights, Index maxdim)
{
    if (!current.empty())
    {
        bool is_interior = false;

        for (auto& v : current)
        {
            if (interior[v])
            {
                is_interior = true;
                break;
            }
        }

        Index p = current.size()-1;
        simplices.emplace_back(current);

        if (is_interior) simplices.back().interior = 1;

        const Simplex& sigma = simplices.back();

        if (p == 0) weights[0].insert({sigma.getid(), 0.});
        else if (p == 1) weights[1].insert({sigma.getid(), graph[current[0]].find(current[1])->second});
        else weights[p].insert({sigma.getid(), 0.});
    }

    if (current.size() == static_cast<size_t>(maxdim) + 1)
        return;

    Index m = cands.size();

    for (Index j = excluded+1; j < m; ++j)
    {
        current.push_back(cands[j]);

        IndexVector new_cands;

        for (Index i = 0; i < j; ++i)
        {
            if (graph[cands[i]].find(cands[j]) != graph[cands[i]].end())
                new_cands.push_back(cands[i]);
        }

        Index ex = new_cands.size();

        for (Index i = j+1; i < m; ++i)
        {
            if (graph[cands[i]].find(cands[j]) != graph[cands[i]].end())
                new_cands.push_back(cands[i]);
        }

        excluded = ex-1;

        bron_kerbosch(current, new_cands, excluded, graph, weights, maxdim);
        current.pop_back();
    }
}

template <class Atom_>
template <class Distance>
VoronoiDiagram<Atom_>::VoronoiDiagram(const PointContainerType& points, const PointContainerType& centers, const IndexVector& center_ids, Distance& distance) : centers(centers), center_ids(center_ids), cell_indices(points.num_points(), 0), dist_to_centers(points.num_points(), std::numeric_limits<Real>::max())
{
    Index size = points.num_points();
    Index num_centers = centers.num_points();

    for (Index i = 0; i < size; ++i)
    {
        for (Index cell_index = 0; cell_index < num_centers; ++cell_index)
        {
            Real dist = distance(centers.mem(cell_index), points.mem(i), centers.size(cell_index), points.size(i));

            if (dist <= dist_to_centers[i])
            {
                dist_to_centers[i] = dist;
                cell_indices[i] = cell_index;
            }
        }
    }
}

template <class Atom_>
void VoronoiDiagram<Atom_>::coalesce_cells(const PointContainerType& mypoints, std::vector<VoronoiCellType>& mycells, MPI_Comm comm) const
{
    using IndexPair = std::tuple<Index, Index>;
    using IndexPairVector = std::vector<IndexPair>;

    MPI_Datatype MPI_ATOM = mpi_type<Atom>();

    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    Index myoffset;
    Index mysize = mypoints.num_points();
    Index num_centers = centers.num_points();

    MPI_Exscan(&mysize, &myoffset, 1, MPI_INDEX, MPI_SUM, comm);
    if (!myrank) myoffset = 0;

    IndexVector cell_point_counts(num_centers), cell_atom_counts(num_centers);
    IndexVector my_cell_point_counts(num_centers, 0), my_cell_atom_counts(num_centers, 0);

    for (Index i = 0; i < mysize; ++i)
    {
        Index cell_index = cell_indices[i];
        my_cell_point_counts[cell_index]++;
        my_cell_atom_counts[cell_index] += mypoints.size(i);
    }

    MPI_Allreduce(my_cell_point_counts.data(), cell_point_counts.data(), (int)num_centers, MPI_INDEX, MPI_SUM, comm);
    MPI_Allreduce(my_cell_atom_counts.data(), cell_atom_counts.data(), (int)num_centers, MPI_INDEX, MPI_SUM, comm);

    std::vector<int> dests(num_centers);
    IndexPairVector pairs;

    for (Index cell_index = 0; cell_index < num_centers; ++cell_index)
    {
        pairs.emplace_back(cell_atom_counts[cell_index], cell_index);
    }

    std::sort(pairs.rbegin(), pairs.rend());

    IndexVector bins(nprocs, 0);

    for (const auto& [size, cell_index] : pairs)
    {
        int dest = std::min_element(bins.begin(), bins.end()) - bins.begin();
        bins[dest] += size;
        dests[cell_index] = dest;
    }

    Index my_assigned_cells = 0;
    IndexVector global_to_local_cell_index_map(num_centers);

    {
        IndexVector fillcounts(nprocs, 0);

        for (Index cell_index = 0; cell_index < num_centers; ++cell_index)
        {
            int dest = dests[cell_index];
            global_to_local_cell_index_map[cell_index] = fillcounts[dest]++;
            if (dest == myrank) my_assigned_cells++;
        }
    }

    std::vector<int> sendcounts(nprocs,0), recvcounts(nprocs), sdispls(nprocs), rdispls(nprocs);
    std::vector<int> sendcounts_atoms(nprocs,0), recvcounts_atoms(nprocs), sdispls_atoms(nprocs), rdispls_atoms(nprocs);

    struct PointEnvelope
    {
        Index id;
        Index cell;
        Index size;
        Real dist;

        PointEnvelope() {}
    };

    using PointEnvelopeVector = std::vector<PointEnvelope>;

    MPI_Datatype MPI_POINT_ENVELOPE;
    MPI_Type_contiguous(sizeof(PointEnvelope), MPI_CHAR, &MPI_POINT_ENVELOPE);
    MPI_Type_commit(&MPI_POINT_ENVELOPE);

    Index totsend, totrecv, totsend_atoms, totrecv_atoms;
    AtomVector sendbuf_atoms, recvbuf_atoms;
    PointEnvelopeVector sendbuf_envs, recvbuf_envs;

    for (Index cell_index = 0; cell_index < num_centers; ++cell_index)
    {
        int dest = dests[cell_index];
        sendcounts[dest] += my_cell_point_counts[cell_index];
        sendcounts_atoms[dest] += my_cell_atom_counts[cell_index];
    }

    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);
    MPI_Alltoall(sendcounts_atoms.data(), 1, MPI_INT, recvcounts_atoms.data(), 1, MPI_INT, comm);

    std::exclusive_scan(sendcounts.begin(), sendcounts.end(), sdispls.begin(), 0);
    std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), 0);

    std::exclusive_scan(sendcounts_atoms.begin(), sendcounts_atoms.end(), sdispls_atoms.begin(), 0);
    std::exclusive_scan(recvcounts_atoms.begin(), recvcounts_atoms.end(), rdispls_atoms.begin(), 0);

    totsend = sendcounts.back() + sdispls.back();
    totrecv = recvcounts.back() + rdispls.back();

    totsend_atoms = sendcounts_atoms.back() + sdispls_atoms.back();
    totrecv_atoms = recvcounts_atoms.back() + rdispls_atoms.back();

    sendbuf_atoms.resize(totsend_atoms), recvbuf_atoms.resize(totrecv_atoms);
    sendbuf_envs.resize(totsend), recvbuf_envs.resize(totrecv);

    auto sptrs = sdispls;

    for (Index i = 0; i < totsend; ++i)
    {
        Index cell_index = cell_indices[i];
        int dest = dests[cell_index];
        Index loc = sptrs[dest]++;

        sendbuf_envs[loc].id = i+myoffset;
        sendbuf_envs[loc].cell = cell_index;
        sendbuf_envs[loc].size = mypoints.size(i);
        sendbuf_envs[loc].dist = dist_to_centers[i];
    }

    auto it = sendbuf_atoms.begin();

    for (Index i = 0; i < totsend; ++i)
    {
        Index point_index = sendbuf_envs[i].id - myoffset;
        Index point_size = mypoints.size(point_index);
        const Atom *point_mem = mypoints.mem(point_index);

        it = std::copy(point_mem, point_mem+point_size, it);
        assert((point_size == sendbuf_envs[i].size));
    }

    MPI_Request reqs[2];

    MPI_Ialltoallv(sendbuf_envs.data(), sendcounts.data(), sdispls.data(), MPI_POINT_ENVELOPE,
                   recvbuf_envs.data(), recvcounts.data(), rdispls.data(), MPI_POINT_ENVELOPE, comm, &reqs[0]);

    MPI_Ialltoallv(sendbuf_atoms.data(), sendcounts_atoms.data(), sdispls_atoms.data(), MPI_ATOM,
                   recvbuf_atoms.data(), recvcounts_atoms.data(), rdispls_atoms.data(), MPI_ATOM, comm, &reqs[1]);

    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    MPI_Type_free(&MPI_POINT_ENVELOPE);

    IndexVector recv_offsets(totrecv);

    Index disp = 0;

    for (Index i = 0; i < totrecv; ++i)
    {
        Index cell = recvbuf_envs[i].cell;
        Index size = recvbuf_envs[i].size;

        recv_offsets[i] = disp;
        disp += size;
    }

    std::vector<std::vector<const Atom*>> cell_point_mems(my_assigned_cells);
    std::vector<IndexVector> cell_point_sizes(my_assigned_cells);
    std::vector<IndexVector> cell_indices(my_assigned_cells);
    std::vector<RealVector> cell_dist_to_centers(my_assigned_cells);
    IndexVector cell_center_offsets(my_assigned_cells);

    for (Index i = 0; i < totrecv; ++i)
    {
        Index cell_index = global_to_local_cell_index_map[recvbuf_envs[i].cell];
        Index size = recvbuf_envs[i].size;

        if (center_ids[recvbuf_envs[i].cell] == recvbuf_envs[i].id)
            cell_center_offsets[cell_index] = cell_point_mems[cell_index].size();

        const Atom *mem = &recvbuf_atoms[recv_offsets[i]];
        cell_point_mems[cell_index].push_back(mem);
        cell_point_sizes[cell_index].push_back(size);
        cell_indices[cell_index].push_back(recvbuf_envs[i].id);
        cell_dist_to_centers[cell_index].push_back(recvbuf_envs[i].dist);
    }

    mycells.clear();
    mycells.reserve(my_assigned_cells);

    for (Index cell = 0; cell < my_assigned_cells; ++cell)
    {
        std::vector<const Atom*>& pts = cell_point_mems[cell];
        RealVector& dists = cell_dist_to_centers[cell];
        IndexVector& indices = cell_indices[cell];
        IndexVector& point_sizes = cell_point_sizes[cell];

        if (!pts.empty())
        {
            std::swap(pts[0], pts[cell_center_offsets[cell]]);
            std::swap(indices[0], indices[cell_center_offsets[cell]]);
            std::swap(point_sizes[0], point_sizes[cell_center_offsets[cell]]);
            std::swap(dists[0], dists[cell_center_offsets[cell]]);
        }

        mycells.emplace_back(PointContainerType(pts, point_sizes), indices, dists);
    }
}

template <class Atom_>
template <class Distance>
void VoronoiDiagram<Atom_>::add_ghost_points_systolic(std::vector<VoronoiCellType>& mycells, Distance& distance, Real radius, Real cover, Index leaf_size, MPI_Comm comm) const
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    MPI_Datatype MPI_ATOM = mpi_type<Atom>();

    struct PointEnvelope
    {
        Index id;
        Index size;
        Real dist;

        PointEnvelope() {}
        PointEnvelope(Index id, Index size, Real dist) : id(id), size(size), dist(dist) {}
    };

    using PointEnvelopeVector = std::vector<PointEnvelope>;

    MPI_Datatype MPI_POINT_ENVELOPE;
    MPI_Type_contiguous(sizeof(PointEnvelope), MPI_CHAR, &MPI_POINT_ENVELOPE);
    MPI_Type_commit(&MPI_POINT_ENVELOPE);

    AtomVector sendbuf_atoms;
    AtomVector recvbuf_atoms;

    PointEnvelopeVector sendbuf_envs;
    PointEnvelopeVector recvbuf_envs;

    Index my_assigned_cells = mycells.size();
    Index my_assigned_points = 0;
    Index my_assigned_atoms = 0;

    for (const VoronoiCellType& cell : mycells)
    {
        my_assigned_points += cell.num_points();
        my_assigned_atoms += cell.num_atoms();
    }

    sendbuf_atoms.reserve(my_assigned_atoms);
    sendbuf_envs.reserve(my_assigned_points);

    std::vector<const Atom*> mycenters_points;
    IndexVector mycenters_sizes;
    IndexVector mycenters_ids;

    std::vector<CoverTree> trees;

    for (const VoronoiCellType& cell : mycells)
    {
        Index cell_point_count = cell.num_points();

        for (Index i = 0; i < cell_point_count; ++i)
        {
            const Atom *point_mem = cell.mem(i);
            Index point_size = cell.size(i);
            Index point_index = cell.index(i);
            Real dist_to_center = cell.dist_to_center(i);

            sendbuf_envs.emplace_back(point_index, point_size, dist_to_center);
            sendbuf_atoms.insert(sendbuf_atoms.end(), point_mem, point_mem+point_size);
        }

        if (cell_point_count >= 1)
        {
            const Atom *point_mem = cell.mem(0);
            Index point_size = cell.size(0);
            Index point_index = cell.index(0);

            mycenters_ids.push_back(point_index);
            mycenters_sizes.push_back(point_size);
            mycenters_points.push_back(point_mem);
        }

        trees.emplace_back(cover, leaf_size);
        trees.back().build(cell, distance);
    }

    PointContainerType mycenters(mycenters_points, mycenters_sizes);

    std::vector<IndexSet> treeids_set(my_assigned_cells);

    for (Index cell_index = 0; cell_index < my_assigned_cells; ++cell_index)
    {
        treeids_set[cell_index].insert(mycells[cell_index].ids_begin(), mycells[cell_index].ids_end());
    }

    CoverTree mycentertree(cover, 1);
    mycentertree.build(mycenters, distance);

    MPI_Request reqs[8];

    int sendcount, sendcount_atoms;
    int recvcount, recvcount_atoms;

    int sendtarg = myrank;
    int recvtarg;

    int recvrank = (myrank+1)%nprocs;
    int sendrank = (myrank-1+nprocs)%nprocs;

    int sendcount_buf[2], recvcount_buf[2];

    IndexVector ghostcells;

    auto functor = [&](Index neighbor, Real dist)
    {
        ghostcells.push_back(neighbor);
    };

    for (int step = 0; step <= nprocs/2; ++step)
    {
        recvtarg = (sendtarg+1)%nprocs;
        sendcount = sendbuf_envs.size();
        sendcount_atoms = sendbuf_atoms.size();

        sendcount_buf[0] = sendcount;
        sendcount_buf[1] = sendcount_atoms;

        double t = -MPI_Wtime();
        MPI_Irecv(recvcount_buf, 2, MPI_INT, recvrank, myrank,   comm, &reqs[0]);
        MPI_Isend(sendcount_buf, 2, MPI_INT, sendrank, sendrank, comm, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        t += MPI_Wtime();

        recvcount = recvcount_buf[0];
        recvcount_atoms = recvcount_buf[1];

        recvbuf_atoms.resize(recvcount_atoms);
        recvbuf_envs.resize(recvcount);

        MPI_Irecv(recvbuf_atoms.data(), recvcount_atoms, MPI_ATOM, recvrank, myrank+nprocs, comm, &reqs[0]);
        MPI_Isend(sendbuf_atoms.data(), sendcount_atoms, MPI_ATOM, sendrank, sendrank+nprocs, comm, &reqs[1]);

        MPI_Irecv(recvbuf_envs.data(), recvcount, MPI_POINT_ENVELOPE, recvrank, myrank+2*nprocs, comm, &reqs[2]);
        MPI_Isend(sendbuf_envs.data(), sendcount, MPI_POINT_ENVELOPE, sendrank, sendrank+2*nprocs, comm, &reqs[3]);

        Index targsize = sendcount;

        const Atom* mem = sendbuf_atoms.data();

        for (Index i = 0; i < targsize; ++i)
        {
            Index dim = sendbuf_envs[i].size;
            Index index = sendbuf_envs[i].id;
            Real dist = sendbuf_envs[i].dist;

            const Atom *query = mem;
            mem += dim;

            ghostcells.clear();
            mycentertree.radius_query(mycenters, distance, query, dim, dist + 2*radius, functor);

            if (ghostcells.empty())
                continue;

            for (Index cell : ghostcells)
                if (mycells[cell].num_points() != 0 && !treeids_set[cell].contains(index))
                {
                    if (trees[cell].has_radius_neighbor(mycells[cell], distance, query, dim, radius))
                    {
                        mycells[cell].add_ghost_point(query, dim, index);
                    }

                    treeids_set[cell].insert(index);
                }
        }

        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        sendtarg = recvtarg;
        sendbuf_atoms.swap(recvbuf_atoms);
        sendbuf_envs.swap(recvbuf_envs);
    }

    MPI_Type_free(&MPI_POINT_ENVELOPE);
}

template <class Atom_>
template <class Distance>
void VoronoiDiagram<Atom_>::add_ghost_points_systolic_rips(std::vector<VoronoiCellType>& mycells, Distance& distance, Real radius, Real cover, Index leaf_size, MPI_Comm comm) const
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    MPI_Datatype MPI_ATOM = mpi_type<Atom>();

    struct PointEnvelope
    {
        Index id;
        Index size;
        Real dist;

        PointEnvelope() {}
        PointEnvelope(Index id, Index size, Real dist) : id(id), size(size), dist(dist) {}
    };

    using PointEnvelopeVector = std::vector<PointEnvelope>;

    MPI_Datatype MPI_POINT_ENVELOPE;
    MPI_Type_contiguous(sizeof(PointEnvelope), MPI_CHAR, &MPI_POINT_ENVELOPE);
    MPI_Type_commit(&MPI_POINT_ENVELOPE);

    AtomVector sendbuf_atoms;
    AtomVector recvbuf_atoms;

    PointEnvelopeVector sendbuf_envs;
    PointEnvelopeVector recvbuf_envs;

    IndexVector sendbuf_shared;
    IndexVector recvbuf_shared;

    Index my_assigned_cells = mycells.size();
    Index my_assigned_points = 0;
    Index my_assigned_atoms = 0;

    for (const VoronoiCellType& cell : mycells)
    {
        my_assigned_points += cell.num_points();
        my_assigned_atoms += cell.num_atoms();
    }

    sendbuf_atoms.reserve(my_assigned_atoms);
    sendbuf_envs.reserve(my_assigned_points);
    sendbuf_shared.resize(my_assigned_points, 0);

    std::vector<const Atom*> mycenters_points;
    IndexVector mycenters_sizes;
    IndexVector mycenters_ids;

    std::vector<CoverTree> trees;

    for (const VoronoiCellType& cell : mycells)
    {
        Index cell_point_count = cell.num_points();

        for (Index i = 0; i < cell_point_count; ++i)
        {
            const Atom *point_mem = cell.mem(i);
            Index point_size = cell.size(i);
            Index point_index = cell.index(i);
            Real dist_to_center = cell.dist_to_center(i);

            sendbuf_envs.emplace_back(point_index, point_size, dist_to_center);
            sendbuf_atoms.insert(sendbuf_atoms.end(), point_mem, point_mem+point_size);
        }

        if (cell_point_count >= 1)
        {
            const Atom *point_mem = cell.mem(0);
            Index point_size = cell.size(0);
            Index point_index = cell.index(0);

            mycenters_ids.push_back(point_index);
            mycenters_sizes.push_back(point_size);
            mycenters_points.push_back(point_mem);
        }

        trees.emplace_back(cover, leaf_size);
        trees.back().build(cell, distance);
    }

    PointContainerType mycenters(mycenters_points, mycenters_sizes);

    std::vector<IndexSet> treeids_set(my_assigned_cells);

    for (Index cell_index = 0; cell_index < my_assigned_cells; ++cell_index)
    {
        treeids_set[cell_index].insert(mycells[cell_index].ids_begin(), mycells[cell_index].ids_end());
    }

    CoverTree mycentertree(cover, 1);
    mycentertree.build(mycenters, distance);

    MPI_Request reqs[8];

    int sendcount, sendcount_atoms;
    int recvcount, recvcount_atoms;

    int sendtarg = myrank;
    int recvtarg;

    int recvrank = (myrank+1)%nprocs;
    int sendrank = (myrank-1+nprocs)%nprocs;

    int sendcount_buf[2], recvcount_buf[2];

    IndexVector ghostcells;

    auto functor = [&](Index neighbor, Real dist)
    {
        ghostcells.push_back(neighbor);
    };

    for (int step = 0; step <= nprocs; ++step)
    {
        recvtarg = (sendtarg+1)%nprocs;
        sendcount = sendbuf_envs.size();
        sendcount_atoms = sendbuf_atoms.size();

        sendcount_buf[0] = sendcount;
        sendcount_buf[1] = sendcount_atoms;

        double t = -MPI_Wtime();
        MPI_Irecv(recvcount_buf, 2, MPI_INT, recvrank, myrank,   comm, &reqs[0]);
        MPI_Isend(sendcount_buf, 2, MPI_INT, sendrank, sendrank, comm, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        t += MPI_Wtime();

        recvcount = recvcount_buf[0];
        recvcount_atoms = recvcount_buf[1];

        recvbuf_atoms.resize(recvcount_atoms);
        recvbuf_envs.resize(recvcount);

        MPI_Irecv(recvbuf_atoms.data(), recvcount_atoms, MPI_ATOM, recvrank, myrank+nprocs, comm, &reqs[0]);
        MPI_Isend(sendbuf_atoms.data(), sendcount_atoms, MPI_ATOM, sendrank, sendrank+nprocs, comm, &reqs[1]);

        MPI_Irecv(recvbuf_envs.data(), recvcount, MPI_POINT_ENVELOPE, recvrank, myrank+2*nprocs, comm, &reqs[2]);
        MPI_Isend(sendbuf_envs.data(), sendcount, MPI_POINT_ENVELOPE, sendrank, sendrank+2*nprocs, comm, &reqs[3]);

        Index targsize = sendcount;

        const Atom* mem = sendbuf_atoms.data();

        for (Index i = 0; i < targsize; ++i)
        {
            Index dim = sendbuf_envs[i].size;
            Index index = sendbuf_envs[i].id;
            Real dist = sendbuf_envs[i].dist;

            const Atom *query = mem;
            mem += dim;

            ghostcells.clear();
            mycentertree.radius_query(mycenters, distance, query, dim, dist + 2*radius, functor);

            if (ghostcells.empty())
                continue;

            for (Index cell : ghostcells)
                if (mycells[cell].num_points() != 0 && !treeids_set[cell].contains(index))
                {
                    if (trees[cell].has_radius_neighbor(mycells[cell], distance, query, dim, radius))
                    {
                        mycells[cell].add_ghost_point(query, dim, index);
                        sendbuf_shared[i]++;
                    }

                    treeids_set[cell].insert(index);
                }
        }

        recvbuf_shared.resize(recvcount);

        MPI_Irecv(recvbuf_shared.data(), recvcount, MPI_INDEX, recvrank, myrank+3*nprocs, comm, &reqs[4]);
        MPI_Isend(sendbuf_shared.data(), sendcount, MPI_INDEX, sendrank, sendrank+3*nprocs, comm, &reqs[5]);

        MPI_Waitall(6, reqs, MPI_STATUSES_IGNORE);

        sendtarg = recvtarg;
        sendbuf_atoms.swap(recvbuf_atoms);
        sendbuf_envs.swap(recvbuf_envs);
        sendbuf_shared.swap(recvbuf_shared);
    }

    MPI_Type_free(&MPI_POINT_ENVELOPE);

    Index p = 0;

    for (Index cell = 0; cell < my_assigned_cells; ++cell)
    {
        Index n = mycells[cell].num_points();

        for (Index i = 0; i < n; ++i, ++p)
        {
            if (recvbuf_shared[p] == 0)
            {
                mycells[cell].set_interior(i);
            }
        }
    }
}
