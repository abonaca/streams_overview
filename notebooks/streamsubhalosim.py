import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from gala.dynamics import mockstream as ms
from gala.units import galactic
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.stats import binned_statistic


class StreamSubhaloSimulation:
    @u.quantity_input(
        M_stream=u.Msun, t_pre_impact=u.Myr, t_post_impact=u.Myr, dt=u.Myr
    )
    def __init__(
        self,
        mw_potential,
        final_prog_w,
        M_stream,
        t_pre_impact,
        t_post_impact,
        dt=1 * u.Myr,
        progenitor_potential=None,
        release_every=1,
        n_particles=1,
        df=None,
    ) -> None:
        self.mw_potential = mw_potential
        self.final_prog_w = final_prog_w
        if not isinstance(self.mw_potential, gp.PotentialBase):
            raise TypeError("Input potential must be a gala potential object")
        if not isinstance(self.final_prog_w, gd.PhaseSpacePosition):
            raise TypeError(
                "Input final_prog_w must be a gala PhaseSpacePosition object"
            )

        # Mockstream and simulation parameters validated by the quantity_input decorator
        self.M_stream = M_stream
        self.t_pre_impact = t_pre_impact
        self.t_post_impact = t_post_impact
        self.dt = dt
        self._mockstream_kw = dict(release_every=release_every, n_particles=n_particles)

        self.H = gp.Hamiltonian(mw_potential)

        if df is None:
            df = ms.FardalStreamDF()
        self._df = df
        self._prog_pot = progenitor_potential

        self._prog_w0 = self.H.integrate_orbit(
            self.final_prog_w,
            dt=-dt,
            t1=t_pre_impact + t_post_impact,
            t2=0,
            Integrator=gi.Ruth4Integrator,
            store_all=False,
        )[0]

    def _make_ms_gen(self):
        # TODO: add option to reset rng to initial state
        return ms.MockStreamGenerator(
            df=self._df, hamiltonian=self.H, progenitor_potential=self._prog_pot
        )

    def run_init_stream(self):
        """
        Generate the initial, unperturbed stream model. This is used to pick a random
        star particle to then define the impact site of the perturber.
        """
        mockstream_gen = self._make_ms_gen()
        at_impact = mockstream_gen.run(
            self._prog_w0,
            self.M_stream,
            dt=self.dt,
            t1=0,
            t2=self.t_pre_impact,
            **self._mockstream_kw,
        )

        # reset seed in DF
        mockstream_gen = self._make_ms_gen()
        final = mockstream_gen.run(
            self._prog_w0,
            self.M_stream,
            dt=self.dt,
            t1=0,
            t2=self.t_pre_impact + self.t_post_impact,
            **self._mockstream_kw,
        )
        return at_impact, final

    def get_impact_site(
        self, init_stream, init_prog, prog_dist=8 * u.kpc, leading=True
    ):
        """
        Given a simulation of the unperturbed stream, pick a random star particle at
        some distance away from the progenitor in either the leading or trailing tail.
        This is used to define the impact site of the perturber. The input stream and
        progenitor position are assumed to be at the final timestep of the eventual
        impact simulation (i.e. the impact site is defined at the end timestep of the
        unperturbed stream simulation).
        """
        if leading:
            lead_trail_sign = 1.0
        else:
            lead_trail_sign = -1.0

        dx = init_stream.xyz - init_prog.xyz
        prog_dists = np.linalg.norm(dx, axis=0)

        # pick zeroth star that matches - MAGIC NUMBER
        rando_star_idx = np.where(
            np.isclose(prog_dists, prog_dist, atol=0.1 * u.kpc)
            & (np.sign(dx[1]) == np.sign(lead_trail_sign * init_prog.v_y))
        )[0][0]

        nearby_mask = (
            np.linalg.norm(
                init_stream.xyz - init_stream[rando_star_idx].xyz[:, None], axis=0
            )
            < 1 * u.kpc
        )  # MAGIC NUMBER

        return gd.PhaseSpacePosition(
            np.mean(init_stream.xyz[:, nearby_mask], axis=1),
            np.mean(init_stream.v_xyz[:, nearby_mask], axis=1),
        )

    @u.quantity_input(t_buffer_impact=u.Myr, impact_dt=u.Myr)
    def run_perturbed_stream(
        self,
        subhalo_at_impact,
        subhalo_potential,
        t_buffer_impact=None,
        impact_dt=None,
    ):
        final_time = self.t_pre_impact + self.t_post_impact

        subhalo_v = np.linalg.norm(subhalo_at_impact.v_xyz)
        if t_buffer_impact is None:
            t_buffer_impact = np.round((1 * u.kpc / subhalo_v).to(u.Myr), decimals=1)
        if impact_dt is None:
            impact_dt = (1.0 * u.pc / subhalo_v).to(u.Myr)

        # Integrate the subhalo orbit from time of impact back to the buffer time
        subhalo_buffer = self.H.integrate_orbit(
            subhalo_at_impact,
            dt=-self.dt / 10,
            t1=self.t_pre_impact,
            t2=self.t_pre_impact - t_buffer_impact,
            Integrator=gi.DOPRI853Integrator,
            store_all=False,
        )[0]

        # Generate the mock stream up to the buffer time relative to the impact
        mockstream_gen = self._make_ms_gen()
        stream_buffer_pre, prog_w_buffer_pre = mockstream_gen.run(
            self._prog_w0,
            self.M_stream,
            dt=self.dt,
            t1=0,
            t2=self.t_pre_impact - t_buffer_impact,
            **self._mockstream_kw,
        )

        # Extract mock stream particle initial conditions to prepare to forward
        # integrate through the impact using N-body forces
        tmp = gd.PhaseSpacePosition(
            stream_buffer_pre.pos, stream_buffer_pre.vel, frame=stream_buffer_pre.frame
        )
        nbody_w0 = gd.combine((subhalo_buffer, tmp))

        null_potential = gp.NullPotential(units=galactic)
        nbody = gd.DirectNBody(
            w0=nbody_w0,
            particle_potentials=[subhalo_potential] + [null_potential] * tmp.shape[0],
            external_potential=self.H.potential,
            frame=self.H.frame,
            save_all=False,
        )

        if (self.t_pre_impact + t_buffer_impact) > final_time:
            buffer_t2 = final_time
        else:
            buffer_t2 = self.t_pre_impact + t_buffer_impact

        stream_impact = nbody.integrate_orbit(
            dt=impact_dt,
            t1=self.t_pre_impact - t_buffer_impact,
            t2=buffer_t2,
        )[1:]  # remove subhalo particle
        if buffer_t2 != final_time:
            stream_after_impact = self.H.integrate_orbit(
                stream_impact,
                dt=self.dt,
                t1=buffer_t2,
                t2=final_time,
                Integrator=gi.Ruth4Integrator,
                store_all=False,
            )[0]
        else:
            stream_after_impact = stream_impact

        stream_after_impact = gd.mockstream.MockStream(
            pos=stream_after_impact.pos,
            vel=stream_after_impact.vel,
            frame=stream_after_impact.frame,
            release_time=stream_buffer_pre.release_time,
        )

        # Needed to make sure the later part of the stream is also affected by the
        # subhalo
        nbody = gd.DirectNBody(
            w0=nbody_w0,
            particle_potentials=[subhalo_potential] + [null_potential] * tmp.shape[0],
            external_potential=self.H.potential,
            frame=self.H.frame,
            save_all=False,
        )
        unpert_stream_post, final_prog = mockstream_gen.run(
            prog_w_buffer_pre[0],
            self.M_stream,
            dt=self.dt,
            t1=self.t_pre_impact - t_buffer_impact,
            t2=final_time,
            nbody=nbody,
            **self._mockstream_kw,
        )

        # merge the two sections of the simulated stream:
        merged_stream = gd.mockstream.MockStream(
            pos=np.hstack((stream_after_impact.xyz, unpert_stream_post.xyz)),
            vel=np.hstack((stream_after_impact.v_xyz, unpert_stream_post.v_xyz)),
            release_time=np.concatenate(
                (stream_after_impact.release_time, unpert_stream_post.release_time)
            ),
        )

        return merged_stream, final_prog[0], final_time


def get_new_basis(impact_xyz, new_zhat_xyz):
    """
    Stream basis is defined so that z points along stream, x and y point orthogonal
    """
    new_zhat = new_zhat_xyz / np.linalg.norm(new_zhat_xyz)

    tmp_yhat = impact_xyz / np.linalg.norm(impact_xyz)
    new_yhat = tmp_yhat - (tmp_yhat @ new_zhat) * new_zhat
    new_yhat = new_yhat / np.linalg.norm(new_yhat)

    new_xhat = np.cross(new_yhat, new_zhat)
    R = np.stack((new_xhat, new_yhat, new_zhat)).T
    return R


@u.quantity_input(b=u.kpc, phi=u.rad, vphi=u.km / u.s, vz=u.km / u.s)
def get_subhalo_w0(impact_site, b, phi, vphi, vz):
    """
    Given an impact site and encounter parameters, return the subhalo phase-space
    position at the time of impact.
    """
    R = get_new_basis(impact_site.xyz.value, impact_site.v_xyz.value)

    # z is along stream
    # x, y other coords
    x = b * np.cos(phi)
    y = b * np.sin(phi)
    z = 0.0 * b.unit
    xyz = np.stack((x, y, z))

    vx = -vphi * np.sin(phi)
    vy = vphi * np.cos(phi)
    vxyz = np.stack((vx, vy, vz))

    return gd.PhaseSpacePosition(
        R @ xyz + impact_site.xyz, R @ vxyz + impact_site.v_xyz
    )


def get_in_stream_frame(stream, prog=None, impact=None, stream_frame=None):
    stream_galcen = coord.Galactocentric(stream.data)
    stream_icrs = stream_galcen.transform_to(coord.ICRS())

    if stream_frame is None:
        if impact is None or prog is None:
            raise ValueError("Must provide impact and prog to get stream frame")

        impact_icrs = impact.to_coord_frame(coord.ICRS())
        other_icrs = prog.to_coord_frame(coord.ICRS())

        stream_frame = gc.GreatCircleICRSFrame.from_xyz(
            xnew=impact_icrs.data.without_differentials(),
            ynew=impact_icrs.data.without_differentials()
            - other_icrs.data.without_differentials(),
        )

    stream_sfr = stream_icrs.transform_to(stream_frame)
    return stream_sfr


def get_stream_track(stream_sfr, lon_lim=None, plot_debug=False):
    """
    Given a stream simulation in a rotated stream frame, compute the smoothed stream
    track in each coordinate component.
    """
    x = stream_sfr.phi1.wrap_at(180 * u.deg).degree
    if lon_lim is None:
        lon_lim = (x.min(), x.max())
    lon_mask = (x >= lon_lim[0]) & (x <= lon_lim[1])

    # longitude bins
    bins = np.percentile(x[lon_mask], np.linspace(5, 95, 81))
    dlon = 2.0
    xtend1 = (lon_lim[0] - 2 * dlon, bins[0])
    N1 = int((xtend1[1] - xtend1[0]) / dlon)
    xtend2 = (bins[-1], lon_lim[1] + 2 * dlon)
    N2 = int((xtend2[1] - xtend2[0]) / dlon)
    bins = np.concatenate(
        (np.linspace(*xtend1, N1), bins[1:-1], np.linspace(*xtend2, N2))
    )

    comps = ["phi2", "distance", "pm_phi1_cosphi2", "pm_phi2", "radial_velocity"]

    tracks = {}
    for comp in comps:
        y = getattr(stream_sfr, comp)
        stat = binned_statistic(x, y.value, bins=bins, statistic=np.nanmedian)
        xc = 0.5 * (stat.bin_edges[1:] + stat.bin_edges[:-1])

        track_mask = np.isfinite(stat.statistic)
        x_track = xc[track_mask]
        y_track = gaussian_filter1d(stat.statistic[track_mask], 2.0, mode="nearest")
        y_track = stat.statistic[track_mask]
        tracks[comp] = InterpolatedUnivariateSpline(x_track, y_track, k=3)

        if plot_debug:
            plt.figure(figsize=(12, 3))
            plt.hist2d(
                x[lon_mask],
                y.value[lon_mask],
                bins=(np.linspace(*lon_lim, 512), 151),
                norm=mpl.colors.LogNorm(vmin=0.1),
                cmap="Greys",
            )
            plt.plot(x_track, y_track, marker="o", lw=1, color="tab:red")

            # ---

            plt.figure(figsize=(12, 3))
            plt.hist2d(
                x[lon_mask],
                y.value[lon_mask] - tracks[comp](x[lon_mask]),
                bins=(np.linspace(*lon_lim, 512), 151),
                norm=mpl.colors.LogNorm(vmin=0.1),
                cmap="Greys",
            )

    return tracks
