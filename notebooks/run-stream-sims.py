import pathlib

import astropy.units as u
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
import matplotlib.pyplot as plt
import numpy as np
import streamsubhalosim as sss
from astropy.cosmology import Planck18
from gala.units import galactic
from scipy.spatial.transform import Rotation


def run_epicycles(paths, df, pot, prog_w0, sim_time, mockstream_kwargs, R):
    name = "epicycles"

    gen = gd.MockStreamGenerator(df, pot)
    stream, prog = gen.run(prog_w0, t1=0, t2=sim_time, **mockstream_kwargs)
    # TODO: save stream particle data to paths["data"]

    fig = stream.plot()
    prog.plot(axes=fig.axes, color="tab:red")
    fig.savefig(paths["plot_path"] / f"stream-xyz-{name}.png", dpi=250)


def run_bar(paths, df, pot, prog_w0, sim_time, mockstream_kwargs, R):
    name = "bar"

    Omega = 41 * u.km / u.s / u.kpc
    sign = -1.0
    bar_frame = gp.ConstantRotatingFrame(Omega * [0, 0, sign], units=galactic)

    mw_barred = gp.CCompositePotential()
    mw_barred["bar"] = gp.LongMuraliBarPotential(
        m=1e10 * u.Msun,
        a=3.0 * u.kpc,
        b=0.75 * u.kpc,
        c=0.5 * u.kpc,
        alpha=-27 * u.deg,
        units=galactic,
    )
    mw_barred["disk"] = pot["disk"].replicate(m=4.155e10 * u.Msun)
    mw_barred["halo"] = pot["halo"]
    mw_barred["nucleus"] = pot["nucleus"]
    assert u.allclose(
        mw_barred.circular_velocity([8.0, 0, 0]),
        pot.circular_velocity([8.0, 0, 0]),
        atol=0.5 * u.km / u.s,
    )

    H = gp.Hamiltonian(mw_barred, bar_frame)
    gen = gd.MockStreamGenerator(df, H)
    stream, prog = gen.run(prog_w0, t1=0, t2=sim_time, **mockstream_kwargs)
    # TODO: save stream particle data to paths["data"]

    fig = stream.plot()
    prog.plot(axes=fig.axes, color="tab:red")
    fig.savefig(paths["plot_path"] / f"stream-xyz-{name}.png", dpi=250)


def run_subhalo(paths, df, pot, prog_w0, sim_time, mockstream_kwargs, R):
    name = "subhalo"

    # Using the Moliné et al. 2017 fitting formula for the concentration-mass relation
    def c200(M200, xsub):
        c0 = 19.9
        a = [-0.195, 0.089, 0.089]
        b = -0.54
        h = Planck18.h
        return (
            c0
            * (
                1
                + np.sum(
                    [
                        (a[i] * np.log10(M200 / (1e8 * u.Msun) * h)) ** (i + 1)
                        for i in range(3)
                    ]
                )
            )
            * (1 + b * np.log10(xsub))
        )

    # TODO: multiple subhalos?
    M200 = 5e6 * u.Msun
    subhalo_pot = gp.NFWPotential.from_M200_c(
        M200, c=c200(M200, xsub=15.0 / 250), units=galactic
    )

    t_post_impact = 0.4 * u.Gyr
    sim = sss.StreamSubhaloSimulation(
        pot,
        prog_w0,
        M_stream=mockstream_kwargs["prog_mass"],
        t_pre_impact=sim_time - t_post_impact,
        t_post_impact=t_post_impact,
    )

    (init_stream, init_prog), _ = sim.run_init_stream()

    # Find an impact site at the final stream time:
    impact_site = sim.get_impact_site(init_stream, init_prog, prog_dist=5 * u.kpc)

    stream_style = dict(
        marker="o", ms=1.0, ls="none", alpha=0.2, plot_function=plt.plot
    )
    fig = init_stream.plot(**stream_style)
    impact_site.plot(color="r", axes=fig.axes, autolim=False, zorder=100)
    fig.savefig(paths["plot_path"] / f"stream-xyz-{name}-init.png", dpi=250)

    # Rewind the impact site to the impact time:
    impact_site_at_impact = sim.H.integrate_orbit(
        impact_site,
        dt=-sim.dt,
        t1=sim.t_pre_impact + sim.t_post_impact,
        t2=sim.t_pre_impact,
        Integrator=gi.DOPRI853Integrator,
    )[-1]

    # TODO: direct impact, arbitrary angles
    subhalo_w0 = sss.get_subhalo_w0(
        impact_site_at_impact,
        b=0.0 * u.pc,
        phi=0.0 * u.deg,
        vphi=50 * u.km / u.s,
        vz=50 * u.km / u.s,
    )

    # Compute "buffer" time duration and timestep
    # Buffer time is 32 times the crossing time:
    BUFFER_N = 32
    subhalo_dv = np.linalg.norm(subhalo_w0.v_xyz - impact_site.v_xyz)
    subhalo_dx = subhalo_pot.parameters["r_s"]

    # Minimum buffer time = 20 Myr
    t_buffer_impact = np.round(
        (BUFFER_N * subhalo_dx / subhalo_dv).to(u.Myr), decimals=0
    )
    t_buffer_impact = np.max(u.Quantity([t_buffer_impact, 20 * u.Myr]))

    # Minimum buffer time step = 0.05 Myr
    impact_dt = np.round((t_buffer_impact / 256).to(u.Myr), decimals=1)
    impact_dt = np.max(u.Quantity([impact_dt, 0.05 * u.Myr]))

    stream, _, final_prog, final_t = sim.run_perturbed_stream(
        subhalo_w0, subhalo_pot, t_buffer_impact, impact_dt
    )

    fig = stream.plot()
    final_prog.plot(axes=fig.axes, color="tab:red")
    fig.savefig(paths["plot_path"] / f"stream-xyz-{name}.png", dpi=250)


def main(pool, paths):
    sim_T = 6 * u.Gyr
    prog_w_final = gd.PhaseSpacePosition(
        [6.0, 0, 12] * u.kpc, [0, -140, -12] * u.km / u.s
    )
    mw_pot = gp.MilkyWayPotential2022()
    prog_orbit = mw_pot.integrate_orbit(
        prog_w_final, dt=-0.2, t1=0, t2=-sim_T, Integrator=gi.DOPRI853Integrator
    )
    prog_w_init = prog_orbit[-1]

    fig = prog_orbit.plot()
    fig.savefig(plot_path / "prog-orbit.png", dpi=250)

    # Rotation matrix so progenitor is along x axis at final time:
    ang = np.arctan2(prog_w_final.z, prog_w_final.x)[0]
    R = Rotation.from_euler("y", ang.to_value(u.deg), degrees=True).as_matrix()

    # Mock stream DF:
    ms_kwargs = {
        "prog_mass": 5e4 * u.Msun,
        # TODO: high resolution case
        # "dt": -0.1 * u.Myr,
        # "release_every": 1,
        # "n_particles": 4,
        # TODO: for testing
        "dt": -2 * u.Myr,
        "release_every": 2,
        "n_particles": 1,
    }

    for func in [run_epicycles, run_bar, run_subhalo]:
        rng = np.random.default_rng(seed=42)
        df = gd.FardalStreamDF(gala_modified=False, random_state=rng)
        func(paths, df, mw_pot, prog_w_init, sim_T, ms_kwargs, R)


if __name__ == "__main__":
    import argparse

    import schwimmbad

    parser = argparse.ArgumentParser(description="Run stream simulations")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mpi", action="store_true", default=False)
    group.add_argument("--ncores", type=int, default=1)
    args = parser.parse_args()

    plot_path = pathlib.Path(__file__).parent.parent / "plots" / "stream-sims"
    plot_path.mkdir(exist_ok=True, parents=True)

    data_path = pathlib.Path(__file__).parent.parent / "data" / "stream-sims"
    data_path.mkdir(exist_ok=True, parents=True)

    with schwimmbad.choose_pool(mpi=args.mpi, processes=args.ncores) as pool:
        main(pool, paths={"data_path": data_path, "plot_path": plot_path})
