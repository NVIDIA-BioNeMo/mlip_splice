// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The Python surface of the MD kernels: `splice.md._ops`.
//
// Every array crosses as a raw device address (a CuPy `.data.ptr`), so nothing
// here owns or copies memory — a workspace is handed its pointers once by
// `setinterface` and then stepped by `process`.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "cell_neighbor.hpp"
#include "verlet.hpp"
#include "langevin.hpp"

using namespace splice_md;

// ------------------------------------------------------------------ neighbour list

static uintptr_t neighbor_create(float cutoff, float skin, float cell_width) {
    return (uintptr_t) new cell_neigh_workspace(cutoff, skin, cell_width);
}

static void neighbor_release(uintptr_t ws) { delete (cell_neigh_workspace *) ws; }

// The box crosses as three Python floats rather than a device pointer: sizing
// the grid needs it on the host, and reading it back per step would put a
// synchronisation inside the MD loop.
static int neighbor_setup(uintptr_t ws, float bx, float by, float bz, int natom, int page) {
    float box[3] = {bx, by, bz};
    cell_neigh_workspace *w = (cell_neigh_workspace *) ws;
    int n = w->Setup(box, natom, page);
    if (n < 0) throw std::runtime_error(w->last_error() ? w->last_error() : "setup failed");
    return n;
}

static void neighbor_setinterface(uintptr_t ws, int natom, uintptr_t box, uintptr_t pos,
                                  uintptr_t counts, uintptr_t offsets, uintptr_t senders,
                                  uintptr_t receivers, uintptr_t status, int edge_capacity) {
    ((cell_neigh_workspace *) ws)->SetInterface(natom, (float *) box, (float *) pos,
        (int *) counts, (int *) offsets, (int *) senders, (int *) receivers,
        (int *) status, edge_capacity);
}

static void neighbor_process(uintptr_t ws) {
    cell_neigh_workspace *w = (cell_neigh_workspace *) ws;
    w->process();
    // A launch that fails leaves the buffers untouched, which reads downstream
    // as an empty or stale neighbour list rather than as an error. Raising here
    // is the whole reason the workspace records the CUDA status at all.
    if (w->last_error()) throw std::runtime_error(w->last_error());
}

static py::dict neighbor_geometry(uintptr_t ws) {
    cell_neigh_workspace *w = (cell_neigh_workspace *) ws;
    py::dict z;
    z["cells"] = py::make_tuple(w->ncell(0), w->ncell(1), w->ncell(2));
    z["cell_size"] = py::make_tuple(w->cell_size(0), w->cell_size(1), w->cell_size(2));
    z["shell"] = w->shell_size();
    z["ncell"] = w->cell_total();
    z["page"] = w->page_size();
    z["cutoff"] = w->cutoff;
    z["skin"] = w->skin;
    return z;
}

// ------------------------------------------------------------------ integrator

static uintptr_t verlet_create() { return (uintptr_t) new verlet_workspace(); }
static void verlet_release(uintptr_t ws) { delete (verlet_workspace *) ws; }

static void verlet_setinterface(uintptr_t ws, int natoms, float dt, uintptr_t box,
                                uintptr_t masses, uintptr_t x, uintptr_t v, uintptr_t f) {
    ((verlet_workspace *) ws)->SetInterface(natoms, dt, (float *) box, (float *) masses,
        (float *) x, (float *) v, (float *) f);
}

static void verlet_vs(uintptr_t ws, uintptr_t w) { ((verlet_workspace *) ws)->set_vscale((workspace *) w); }
static void verlet_vc(uintptr_t ws, uintptr_t w) { ((verlet_workspace *) ws)->set_vcorrection((workspace *) w); }
static void verlet_process(uintptr_t ws) { ((verlet_workspace *) ws)->process(); }

// ------------------------------------------------------------------ thermostat

static uintptr_t langevin_create() { return (uintptr_t) new langevin_workspace(); }
static void langevin_release(uintptr_t ws) { delete (langevin_workspace *) ws; }

static void langevin_setup(uintptr_t ws, float dt, float temperature, float friction,
                           int natoms, uintptr_t masses) {
    ((langevin_workspace *) ws)->setup(dt, temperature, friction, natoms, (float *) masses);
}

static void langevin_setinterface(uintptr_t ws, uintptr_t pos, uintptr_t vel, uintptr_t frc) {
    ((langevin_workspace *) ws)->SetInterface((float *) pos, (float *) vel, (float *) frc);
}

static void langevin_process(uintptr_t ws) { ((langevin_workspace *) ws)->process(); }

// ------------------------------------------------------------------ module

PYBIND11_MODULE(_ops, m) {
    m.doc() = "CUDA kernels for MD: cell neighbour list, Verlet integrator, Langevin thermostat.";

    py::module n = m.def_submodule("neighbor", "Cell-list neighbour search");
    n.def("workspace", &neighbor_create, "Allocate a neighbour-list workspace",
          py::arg("cutoff"), py::arg("skin") = 0.0f, py::arg("cell_width") = 0.0f);
    n.def("release", &neighbor_release, "Free a neighbour-list workspace");
    n.def("setup", &neighbor_setup,
          "Size the cell grid for this box; returns the number of cells",
          py::arg("ws"), py::arg("box_x"), py::arg("box_y"), py::arg("box_z"),
          py::arg("natom"), py::arg("page") = 128);
    n.def("setinterface", &neighbor_setinterface, "Bind the device buffers",
          py::arg("ws"), py::arg("natom"), py::arg("box"), py::arg("pos"), py::arg("counts"),
          py::arg("offsets"), py::arg("senders"), py::arg("receivers"), py::arg("status"),
          py::arg("edge_capacity"));
    n.def("process", &neighbor_process, "Rebuild the compact symmetric edge list");
    n.def("geometry", &neighbor_geometry, "Grid dimensions, cell capacity and shell size");

    py::module v = m.def_submodule("verlet", "Velocity-Verlet integrator");
    v.def("workspace", &verlet_create, "Allocate an integrator workspace");
    v.def("release", &verlet_release, "Free an integrator workspace");
    v.def("setinterface", &verlet_setinterface, "Bind the device buffers",
          py::arg("ws"), py::arg("natoms"), py::arg("dt"), py::arg("box"), py::arg("masses"),
          py::arg("x"), py::arg("v"), py::arg("f"));
    v.def("vs", &verlet_vs, "Attach a velocity-scaling stage (thermostat)");
    v.def("vc", &verlet_vc, "Attach a velocity-correction stage");
    v.def("process", &verlet_process, "Advance one step");

    py::module l = m.def_submodule("langevin", "Langevin thermostat");
    l.def("workspace", &langevin_create, "Allocate a thermostat workspace");
    l.def("release", &langevin_release, "Free a thermostat workspace");
    l.def("setup", &langevin_setup, "Precompute the per-atom prefactors",
          py::arg("ws"), py::arg("dt"), py::arg("temperature"), py::arg("friction"),
          py::arg("natoms"), py::arg("masses"));
    l.def("setinterface", &langevin_setinterface, "Bind the device buffers",
          py::arg("ws"), py::arg("pos"), py::arg("vel"), py::arg("frc"));
    l.def("process", &langevin_process, "Apply one thermostat update");
}
