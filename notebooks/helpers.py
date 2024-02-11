# mypy: disable-error-code="import-untyped, import-not-found"
"""Helper functions and centrally-defined assumptions / definitions"""

import pathlib

import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import mad_std
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import minimize

data_path = (pathlib.Path(".").parent / ".." / "data").absolute()

galcen_frame = coord.Galactocentric(
    galcen_distance=8.275 * u.kpc, galcen_v_sun=[8.4, 251, 8.4] * u.km / u.s
)
mw = gp.MilkyWayPotential2022()


def get_isochrone(age=12e9, feh=-2.2):
    # First, try to load from a pre-cached file:
    filename = data_path / f"iso_{age / 1e9:.0f}Gyr_{feh:.1f}feh.csv"
    if filename.exists():
        return at.Table.read(filename)

    from ezpadova import parsec

    Z = 10 ** (feh + np.log10(0.0207))
    iso = parsec.get_one_isochrone(age, Z, model="parsec12s", phot="panstarrs1")
    iso = iso.to_astropy_table()

    iso_gaia = parsec.get_one_isochrone(age, Z, model="parsec12s", phot="gaia")
    iso_gaia = iso_gaia.to_astropy_table()
    iso_gaia.remove_columns([x for x in iso_gaia.colnames if x in iso.colnames])

    iso = at.hstack((iso, iso_gaia))
    iso.write(filename)

    return iso


##############################################################################
# For stream orbit fitting:
#
def get_frame_from_points(c):
    """
    Parameters
    ----------
    c : astropy.coordinates.SkyCoord
        Must have sky coordinates for a set of stars.
    """
    u_xyz = (
        c.data.represent_as(coord.UnitSphericalRepresentation).to_cartesian().xyz.value
    )

    # Initial guess of the direction that the stream spreads in, based on eigenvectors:
    eig_res = np.linalg.eig(np.cov(u_xyz))

    proj = u_xyz.T @ eig_res.eigenvectors[:, 0]
    idx = np.argsort(proj)
    frame = gc.GreatCircleICRSFrame.from_endpoints(c[idx[0]], c[idx[-1]])

    # Now we will make fine adjustments to the orientation of the frame to get the
    # stream to lie near phi2=0

    # The initial rotation matrix from the endpoints above:
    R = np.zeros((3, 3))
    R[0] = frame.origin.cartesian.xyz
    R[2] = frame.pole.cartesian.xyz
    R[1] = np.cross(R[:, 0], R[:, 2])

    def frame_adjust(rot_x, rot_y):
        Rx = coord.matrix_utilities.rotation_matrix(rot_x * u.deg, "x")
        Ry = coord.matrix_utilities.rotation_matrix(rot_y * u.deg, "y")
        new_R = Rx @ Ry @ R
        return gc.GreatCircleICRSFrame.from_R(new_R)

    def objective(p):
        new_frame = frame_adjust(p[0], p[1])
        c_fr = c.transform_to(new_frame)
        return np.sum(c_fr.phi2.degree**2)

    res = minimize(objective, x0=[0.0, 0.0], bounds=[[-30, 30], [-20, 20]])
    return frame_adjust(*res.x)


def get_w0_from_p(p, frame):
    # At phi1 = 0.
    orbit_w0_fr = coord.SkyCoord(
        phi1=0 * u.deg,
        phi2=p["phi2"] * u.deg,
        distance=p["distance"] * u.kpc,
        pm_phi1_cosphi2=p["pmphi1"] * u.mas / u.yr,
        pm_phi2=p["pmphi2"] * u.mas / u.yr,
        radial_velocity=p["rv"] * u.km / u.s,
        frame=frame,
    )
    orbit_w0_galcen = orbit_w0_fr.transform_to(galcen_frame)
    return gd.PhaseSpacePosition(orbit_w0_galcen.data)


def p_arr_to_dict(p):
    return {
        "phi2": p[0],
        "distance": p[1],
        "pmphi1": p[2],
        "pmphi2": p[3],
        "rv": p[4],
    }


def get_orbit(mw, orbit_w0, int_time):
    orbit1 = mw.integrate_orbit(orbit_w0, dt=-0.5, t1=0, t2=-2 * int_time)
    orbit2 = mw.integrate_orbit(orbit_w0, dt=0.5, t1=0, t2=2 * int_time)

    orbit_xyz = np.hstack((orbit1[::-1].xyz, orbit2[1:].xyz))
    orbit_vxyz = np.hstack((orbit1[::-1].v_xyz, orbit2[1:].v_xyz))
    orbit_t = np.concatenate((orbit1[::-1].t, orbit2.t[1:]))
    orbit = gd.Orbit(orbit_xyz, orbit_vxyz, t=orbit_t)[::-1]
    return orbit


def get_orbit_from_p(mw, p, c_fr):
    phi1_size = c_fr.phi1.radian.max() - c_fr.phi1.radian.min()
    orbit_w0 = get_w0_from_p(p, c_fr.frame)

    # Old way:
    # int_time = (
    #     (phi1_size * p["distance"] * u.kpc) / np.linalg.norm(orbit_w0.v_xyz)
    # ).to(u.Myr)
    int_time = (
        (phi1_size * u.radian)
        / (np.sqrt(p["pmphi1"] ** 2 + p["pmphi2"] ** 2) * u.mas / u.yr)
    ).to_value(u.Myr)
    int_time = np.max([int_time, 100.0]) * u.Myr

    orbit = get_orbit(mw, orbit_w0, int_time)
    return orbit


def cut_wrapped_orbit(orbit, orbit_fr):
    idx = np.where(np.abs(np.diff(orbit_fr.phi1.wrap_at(180 * u.deg).degree)) > 180)[0]

    orbit_mask = np.ones(orbit.ntimes, dtype=bool)
    for i in idx:
        wrap_time = orbit.t[i]
        if wrap_time > 0:
            orbit_mask &= orbit.t < wrap_time
        else:
            orbit_mask &= orbit.t > wrap_time
    return orbit[orbit_mask], orbit_fr[orbit_mask]


def ln_likelihood(p, c_fr, data, mw):
    orbit = get_orbit_from_p(mw, p, c_fr)
    orbit_fr = orbit.to_coord_frame(c_fr.frame, galactocentric_frame=galcen_frame)

    # if the orbit wraps, need to cut it off at +/- 180º!
    orbit, orbit_fr = cut_wrapped_orbit(orbit, orbit_fr)

    # fig, ax = plt.subplots()
    # ax.plot(orbit.t.value[1:], np.diff(orbit_fr.phi1.degree))

    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.scatter(c_fr.phi1.degree, c_fr.phi2.degree, s=3)
    # ax.plot(orbit_fr.phi1.degree, orbit_fr.phi2.degree, marker='')

    interps = {}
    units = {}
    for name in ["phi2", "distance", "pm_phi1_cosphi2", "pm_phi2", "radial_velocity"]:
        comp = getattr(orbit_fr, name)
        x = orbit_fr.phi1.wrap_at(180 * u.deg).degree
        idx = np.argsort(x)
        interps[name] = InterpolatedUnivariateSpline(x[idx], comp[idx], k=3)
        units[name] = comp.unit

    ll = 0.0
    for name in interps:
        if name == "phi2":
            err = 0.05
        elif name == "distance":
            err = data["parallax_error"]
        else:
            err = data[f"{name}_error"]

        model_y = interps[name](c_fr.phi1.wrap_at(180 * u.deg).degree)

        if name == "distance":
            y = data["parallax"]
            model_y = c_fr.distance.parallax.to_value(u.mas)
        else:
            y = getattr(c_fr, name).to_value(units[name])
        ll += -0.5 * (model_y - y) ** 2 / err**2

    return ll


def objective(p, *args, **kwargs):
    p = p_arr_to_dict(p)
    lls = ln_likelihood(p, *args, **kwargs)
    return -np.sum(lls) / len(lls)


def run_orbit_fit(stream_id, gaia_data, c, N_init_dist=6):
    assert len(gaia_data) == len(c)
    frame = get_frame_from_points(c)
    c_fr = c.transform_to(frame)

    C, _ = gaia_data.get_cov(coords=["pmra", "pmdec"])
    C_pm_fr = gc.transform_pm_cov(c, C, frame)
    pm1_err = np.sqrt(C_pm_fr[:, 0, 0])
    pm2_err = np.sqrt(C_pm_fr[:, 1, 1])
    obj_data = {
        "parallax": gaia_data.parallax.value,
        "parallax_error": gaia_data.parallax_error.value,
        "pm_phi1_cosphi2_error": pm1_err,
        "pm_phi2_error": pm2_err,
        "radial_velocity_error": gaia_data.evh,
    }

    idx = np.argsort(np.abs(c_fr.phi1.wrap_at(180 * u.deg).degree))
    phi2 = np.nanmedian(c_fr.phi2.to_value(u.deg)[idx][:8])
    pm_phi1 = np.nanmean(c_fr.pm_phi1_cosphi2[idx][:8])
    pm_phi2 = np.nanmean(c_fr.pm_phi2[idx][:8])
    rv = np.nanmean(gaia_data["vh"][gaia_data["vh"] != 0.0])

    # Try orbit-fitting optimization from N_init_dist different initial distance values,
    # then pick the best one after optimization:
    init_dists = np.geomspace(1, 40, N_init_dist)
    reses = []
    for d0 in init_dists:
        p0 = {
            "phi2": phi2,
            "distance": d0,
            "pmphi1": pm_phi1.value,
            "pmphi2": pm_phi2.value,
            "rv": rv.value,
        }
        try:
            res = minimize(
                objective,
                list(p0.values()),
                bounds=[[-5, 5], [0.5, 50], [-100, 100], [-100, 100], [-500, 500]],
                args=(c_fr, obj_data, mw),
                method="L-BFGS-B",
                options={"ftol": 1e-10, "gtol": 1e-10},
            )
        except Exception as err:
            print(f"{stream_id} failed with an exception: ")
            print(err)
            continue

        if not res.success:
            continue

        reses.append(res)

    if len(reses) == 0:
        print(f"{stream_id} failed: {res.nit}")
        return None
    elif len(reses) == 1:
        res = reses[0]
    else:
        i = np.argmin([r.fun for r in reses])
        res = reses[i]

    res_p = p_arr_to_dict(res.x)
    orbit = get_orbit_from_p(mw, res_p, c_fr)
    orbit_fr = orbit.to_coord_frame(c_fr.frame, galactocentric_frame=galcen_frame)

    # Store stuff:
    this_data = {}
    this_data["c_fr"] = c_fr
    this_data["obj_data"] = obj_data
    this_data["p"] = res_p
    this_data["orbit"] = orbit
    this_data["orbit_fr"] = orbit_fr
    return this_data


def make_components_plot(orbit, orbit_fr, c_fr, obj_data):
    orbit, orbit_fr = cut_wrapped_orbit(orbit, orbit_fr)

    comps = ["phi2", "parallax", "pm_phi1_cosphi2", "pm_phi2", "radial_velocity"]
    fig, axes = plt.subplots(
        len(comps), 1, figsize=(8, len(comps) * 2.5), sharex=True, layout="constrained"
    )
    err_style = dict(ls="none", marker="o", ms=3, color="k")
    for i, name in enumerate(comps):
        ax = axes[i]
        if name == "parallax":
            ax.plot(
                orbit_fr.phi1.degree,
                orbit_fr.distance.parallax.to_value(u.mas),
                marker="",
                color="tab:blue",
            )
            ax.errorbar(
                c_fr.phi1.degree,
                obj_data["parallax"],
                obj_data["parallax_error"],
                **err_style,
            )

        else:
            ax.plot(
                orbit_fr.phi1.degree,
                getattr(orbit_fr, name).value,
                marker="",
                color="tab:blue",
            )

            y = getattr(c_fr, name).value
            y_mask = ~np.isclose(y, 0.0)
            if f"{name}_error" in obj_data:
                ax.errorbar(
                    c_fr.phi1.degree[y_mask],
                    y[y_mask],
                    obj_data[f"{name}_error"][y_mask],
                    **err_style,
                )
            else:
                ax.scatter(c_fr.phi1.degree[y_mask], y[y_mask], s=3)

        ax.set_ylabel(name)
    ax.set_xlim(-100, 100)
    axes[-1].set_xlabel("phi1")
    return fig, axes


def make_ibata_poly_nodes(c_fr, n_nodes_per_segment=128, n_sigma=1, poly_deg=2):
    """
    Given the Ibata+2023 stream members, fit a polynomial to the phi1 vs. phi2
    distribution and measure the width of the stream using the Median Absolute Deviation
    of the members from the polynomial. Then, create a polygon that traces the stream,
    with nodes at the +/- n_sigma lines.
    """
    N = n_nodes_per_segment

    poly = np.polynomial.Polynomial.fit(
        c_fr.phi1.degree, c_fr.phi2.degree, deg=poly_deg
    )
    xgrid = np.linspace(c_fr.phi1.degree.min(), c_fr.phi1.degree.max(), N)
    mu = poly(xgrid)
    sigma = mad_std(c_fr.phi2.degree - poly(c_fr.phi1.degree))
    track_sc = coord.SkyCoord(
        phi1=xgrid * u.deg,
        phi2=mu * u.deg,
        frame=c_fr.frame.replicate_without_data(),
    )

    nodes = np.concatenate(
        (
            np.stack((xgrid, mu - n_sigma * sigma)).T,
            np.stack(
                (
                    np.full(N, xgrid[-1]),
                    np.linspace(mu[-1] - n_sigma * sigma, mu[-1] + n_sigma * sigma, N),
                )
            ).T,
            np.stack((xgrid, mu + n_sigma * sigma)).T[::-1],
            np.stack(
                (
                    np.full(N, xgrid[0]),
                    np.linspace(mu[0] - n_sigma * sigma, mu[0] + n_sigma * sigma, N),
                )
            ).T[::-1],
        )
    )
    poly_sc = coord.SkyCoord(
        phi1=nodes[:, 0] * u.deg,
        phi2=nodes[:, 1] * u.deg,
        frame=c_fr.frame.replicate_without_data(),
    )

    return track_sc, poly_sc


##############################################################################


def get_full_galstreams_poly(poly_sc):
    """
    Galstreams stream sky polygons only have 2 segments, varying phi1 along the +/- phi2
    extent. This function fills in the phi2 segments along the +/- phi1 edges, which is
    needed for visualizing the polygons with matplotlib and cartopy.
    """
    split_i = np.where(np.abs(np.diff(poly_sc.phi2.degree)) > 0.5)[0][0] + 1

    nodes = np.concatenate(
        (
            np.stack((poly_sc.phi1.degree[:split_i], poly_sc.phi2.degree[:split_i])).T,
            np.stack(
                (
                    np.full(128, poly_sc.phi1.degree[split_i]),
                    np.linspace(
                        poly_sc.phi2.degree[split_i - 1],
                        poly_sc.phi2.degree[split_i],
                        128,
                    ),
                )
            ).T,
            np.stack(
                (
                    poly_sc.phi1.degree[split_i + 1 : -1],
                    poly_sc.phi2.degree[split_i + 1 : -1],
                )
            ).T,
            np.stack(
                (
                    np.full(128, poly_sc.phi1.degree[0]),
                    np.linspace(
                        poly_sc.phi2.degree[-2],
                        poly_sc.phi2.degree[0],
                        128,
                    ),
                )
            ).T,
        )
    )

    # reverse order below to make the points wind counter-clockwise = convex
    poly_sc = coord.SkyCoord(
        phi1=nodes[::-1, 0] * u.deg,
        phi2=nodes[::-1, 1] * u.deg,
        frame=poly_sc.frame.replicate_without_data(),
    )
    return poly_sc


def get_default_track_for_stream(stream_name):
    if getattr(get_default_track_for_stream, "mwstreams", None) is None:
        import galstreams

        get_default_track_for_stream.mwstreams = galstreams.MWStreams()

    mwstreams = get_default_track_for_stream.mwstreams
    galstreams_default_tracks = np.array(list(mwstreams.keys()))
    track_names = np.array(mwstreams.get_track_names_for_stream(stream_name))
    try:
        track_name = track_names[np.in1d(track_names, galstreams_default_tracks)][0]
        return track_name, mwstreams[track_name]
    except IndexError:
        return None, None
