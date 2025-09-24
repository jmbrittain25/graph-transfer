import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from astropy.constants import G, M_earth
from astropy.coordinates import (
    GCRS, ITRS, CartesianRepresentation, CartesianDifferential
)
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from functools import partial
import multiprocessing
from scipy.integrate import solve_ivp
from tqdm import tqdm


# Constants
mu = (G * M_earth).to(u.km**3 / u.s**2).value
r_earth = 6371.0


def generate_random_orbit(epoch, orbit_type="Random", two_d=False):
    """
    Your existing random orbit generator. 
    Returns a poliastro.twobody.Orbit object.
    """
    if orbit_type == "Random":
        orbit_type = np.random.choice(["LEO", "MEO", "HEO", "GEO"])

    # Define altitude ranges for each orbital regime
    if orbit_type == "LEO":
        periapsis_alt_km_min, periapsis_alt_km_max = 160, 2000
        apoapsis_alt_km_min, apoapsis_alt_km_max = 160, 2000
    elif orbit_type == "MEO":
        periapsis_alt_km_min, periapsis_alt_km_max = 2000, 35786
        apoapsis_alt_km_min, apoapsis_alt_km_max = 2000, 35786
    elif orbit_type == "HEO":
        periapsis_alt_km_min, periapsis_alt_km_max = 160, 2000
        apoapsis_alt_km_min, apoapsis_alt_km_max = 2000, 35786
    elif orbit_type == "GEO":
        periapsis_alt_km_min, periapsis_alt_km_max = 35786, 35786
        apoapsis_alt_km_min, apoapsis_alt_km_max = 35786, 35786
    else:
        raise ValueError("Invalid orbit type. Must be 'LEO', 'MEO', 'HEO', 'GEO' or 'Random'.")

    # Random altitudes
    periapsis_alt = np.random.uniform(periapsis_alt_km_min, periapsis_alt_km_max)
    apoapsis_alt = np.random.uniform(apoapsis_alt_km_min, apoapsis_alt_km_max)

    # Ensure apoapsis >= periapsis
    while apoapsis_alt < periapsis_alt:
        apoapsis_alt = np.random.uniform(apoapsis_alt_km_min, apoapsis_alt_km_max)

    # Calculate distances from Earth's center
    periapsis = 6371 + periapsis_alt
    apoapsis = 6371 + apoapsis_alt

    # Semi-major axis and eccentricity
    a = (periapsis + apoapsis) / 2 * u.km
    ecc = (apoapsis - periapsis) / (apoapsis + periapsis) * u.one

    # Random angles
    inc = np.random.uniform(0, 180) * u.deg if not two_d else 0 * u.deg
    raan = np.random.uniform(0, 360) * u.deg if not two_d else 0 * u.deg
    argp = np.random.uniform(0, 360) * u.deg
    nu = np.random.uniform(0, 360) * u.deg

    return Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, epoch=epoch), orbit_type


def two_body_2d(t, state):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    if r < 1e-6:
        return [0, 0, 0, 0]
    ax = -mu * x / r**3
    ay = -mu * y / r**3
    return [vx, vy, ax, ay]


def propagate_orbit(state0, period, time_step=60):
    t_max = period
    num_steps = int(t_max // time_step) + 1
    t_eval = np.linspace(0, t_max, num_steps)
    sol = solve_ivp(two_body_2d, [0, t_max], state0, t_eval=t_eval, rtol=1e-8)
    states = sol.y.T  # [num_steps, 4]
    t_norm = t_eval / t_max
    return np.concatenate([states, t_norm[:, np.newaxis]], axis=1)  # [num_steps, 5]


def generate_dataset(n_orbits=1000, time_step=60, out_file=None):
    results = []
    types = []
    for i in tqdm(range(n_orbits), desc="Generating orbits"):
        np.random.seed(i)
        orbit, orbit_type = generate_random_orbit()
        state0, period = orbit.r.to(u.km).value, orbit.period.to(u.s).value
        seq = propagate_orbit(state0, period, time_step)
        results.append(seq)
        types.append(orbit_type)
    np.savez(out_file, trajectories=np.array(results, dtype=object), types=np.array(types))
    print(f"Dataset saved: {out_file} with {n_orbits} trajectories (variable lengths)")
    return results


def propagate_segment(state, t_start, t_end, time_step=60):
    t_eval = np.arange(t_start, t_end + time_step, time_step)
    sol = solve_ivp(two_body_2d, [t_start, t_end], state, t_eval=t_eval, rtol=1e-8)
    states = sol.y.T  # [len(t_eval), 4]
    t_norm = (t_eval - t_start) / (t_end - t_start)  # Relative norm for segment
    return np.concatenate([states, t_norm[:, np.newaxis]], axis=1)

def apply_tangential_burn(state, dv):
    # Apply impulsive dv along velocity direction (pro/retrograde)
    vx, vy = state[2], state[3]
    v_mag = np.sqrt(vx**2 + vy**2)
    if v_mag < 1e-6:
        return state
    unit_v = np.array([vx, vy]) / v_mag
    new_vx = vx + dv * unit_v[0]
    new_vy = vy + dv * unit_v[1]
    return [state[0], state[1], new_vx, new_vy]

def generate_perturbed_trajectory(params, time_step=60):
    state0, period, orbit_type = generate_random_orbit()
    max_dv_per, max_num_burns, max_total_dv, max_time_btwn, max_total_time = params

    num_burns = np.random.randint(1, max_num_burns + 1)
    total_dv_used = 0.0
    current_time = 0.0
    current_state = state0
    segments = []

    for i in range(num_burns):
        time_to_next = np.random.uniform(time_step, max_time_btwn)
        if current_time + time_to_next > max_total_time:
            time_to_next = max_total_time - current_time
        seg_end = current_time + time_to_next
        seg = propagate_segment(current_state, current_time, seg_end, time_step)
        segments.append(seg[:-1])  # Exclude last to avoid duplicate at burn

        # Apply burn at seg_end
        dv = np.random.uniform(-max_dv_per, max_dv_per)  # Random sign/mag
        if abs(total_dv_used + dv) > max_total_dv:
            dv = np.sign(dv) * (max_total_dv - abs(total_dv_used))  # Cap
        current_state = apply_tangential_burn(seg[-1, :4], dv)  # Apply to last state
        total_dv_used += abs(dv)
        current_time = seg_end

        if current_time >= max_total_time or total_dv_used >= max_total_dv:
            break

    # Final coast to max_total_time if needed
    if current_time < max_total_time:
        final_seg = propagate_segment(current_state, current_time, max_total_time, time_step)
        segments.append(final_seg)

    full_traj = np.vstack(segments)
    global_t_norm = np.linspace(0, 1, len(full_traj))[:, np.newaxis]  # Normalize over whole traj
    full_traj[:, -1] = global_t_norm.squeeze()
    return full_traj, orbit_type  # [num_points, 5], same as passive

def generate_perturbed_dataset(n_orbits=1000, max_dv_per=0.5, max_num_burns=3, 
                               max_total_dv=1.0, max_time_btwn=3600, max_total_time=86400, 
                               time_step=60, out_file="perturbed_orbits_2d.npz"):
    results = []
    types = []
    params = (max_dv_per, max_num_burns, max_total_dv, max_time_btwn, max_total_time)
    for i in tqdm(range(n_orbits), desc="Generating perturbed orbits"):
        np.random.seed(i)
        traj, orbit_type = generate_perturbed_trajectory(params, time_step)
        results.append(traj)
        types.append(orbit_type)
    np.savez(out_file, trajectories=np.array(results, dtype=object), types=np.array(types))
    print(f"Perturbed dataset saved: {out_file} with {n_orbits} trajectories")
    return results


def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates (km) to spherical (r, theta, phi).
    - r (km)
    - theta (deg) = polar angle from z-axis (colatitude)
    - phi (deg)   = azimuth angle in x-y plane
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero
    if r < 1e-12:
        return (0.0, 0.0, 0.0)
    theta = np.degrees(np.arccos(z / r))  # 0 to 180
    phi = np.degrees(np.arctan2(y, x))    # -180 to 180
    return (r, theta, phi)


def propagate_orbit_to_df(orbit_obj, orbit_id, orbit_regime, time_step=60*u.s, num_steps=None):
    """
    Propagate a single poliastro Orbit at fixed time intervals.
    Collect position & velocity in both ECI (GCRS) and ECEF (ITRS).
    Return a pandas DataFrame with one row per timestep.

    Parameters
    ----------
    orbit_obj : poliastro.twobody.Orbit
    orbit_id : str
        A unique identifier string for the orbit
    orbit_regime : str
        e.g. 'LEO', 'MEO', 'HEO', 'GEO', or 'Random'
    time_step : astropy.units.Quantity
        Timestep, e.g. 60*u.s

    Returns
    -------
    df : pandas.DataFrame
        Columns: orbit_id, orbit_regime, epoch, period_s, time_s,
                 x_eci_km, y_eci_km, z_eci_km, vx_eci_km_s, vy_eci_km_s, vz_eci_km_s,
                 r_eci_km, theta_eci_deg, phi_eci_deg,
                 x_ecef_km, y_ecef_km, z_ecef_km, vx_ecef_km_s, vy_ecef_km_s, vz_ecef_km_s,
                 r_ecef_km, theta_ecef_deg, phi_ecef_deg,
                 sma_km, ecc, inc_deg, raan_deg, argp_deg, nu_deg
    """
    rows = []
    period_s = orbit_obj.period.to_value(u.s)
    epoch_str = orbit_obj.epoch.isot  # epoch in ISO format

    if num_steps is None:
        num_steps = int(period_s // time_step.to_value(u.s)) - 1
    else:
        time_step = period_s / num_steps * u.s

    for step in range(num_steps):
        dt = step * time_step
        # Propagate the orbit to the desired offset from epoch
        new_orbit = orbit_obj.propagate(dt)

        # ECI coordinates (poliastro's representation is effectively GCRS)
        r_eci = new_orbit.r.to(u.km).value  # [x, y, z]
        v_eci = new_orbit.v.to(u.km/u.s).value  # [vx, vy, vz]

        x_eci, y_eci, z_eci = r_eci
        vx_eci, vy_eci, vz_eci = v_eci

        # Convert ECI -> ECEF using GCRS -> ITRS transform
        # We attach obstime = new_orbit.epoch + dt
        current_time = new_orbit.epoch + dt
        gcrs_coord = GCRS(
            x=x_eci*u.km, y=y_eci*u.km, z=z_eci*u.km,
            v_x=vx_eci*(u.km/u.s), v_y=vy_eci*(u.km/u.s), v_z=vz_eci*(u.km/u.s),
            representation_type=CartesianRepresentation,
            differential_type=CartesianDifferential,
            obstime=current_time
        )
        itrs_coord = gcrs_coord.transform_to(ITRS(obstime=current_time))

        x_ecef = itrs_coord.x.to_value(u.km)
        y_ecef = itrs_coord.y.to_value(u.km)
        z_ecef = itrs_coord.z.to_value(u.km)
        vx_ecef = itrs_coord.v_x.to_value(u.km/u.s)
        vy_ecef = itrs_coord.v_y.to_value(u.km/u.s)
        vz_ecef = itrs_coord.v_z.to_value(u.km/u.s)

        # Spherical coords in ECI
        r_eci_val, theta_eci_val, phi_eci_val = cartesian_to_spherical(x_eci, y_eci, z_eci)
        # Spherical coords in ECEF
        r_ecef_val, theta_ecef_val, phi_ecef_val = cartesian_to_spherical(x_ecef, y_ecef, z_ecef)

        # Classical orbital elements (use new_orbit)
        sma_km   = new_orbit.a.to_value(u.km)
        ecc_val  = new_orbit.ecc.value
        inc_deg  = new_orbit.inc.to_value(u.deg)
        raan_deg = new_orbit.raan.to_value(u.deg)
        argp_deg = new_orbit.argp.to_value(u.deg)
        nu_deg   = new_orbit.nu.to_value(u.deg)

        row = {
            "orbit_id": orbit_id,
            "orbit_regime": orbit_regime,
            "epoch": epoch_str,
            "period_s": period_s,
            "time_s": dt.to_value(u.s),

            # ECI (GCRS) cartesian
            "x_eci_km": x_eci,
            "y_eci_km": y_eci,
            "z_eci_km": z_eci,
            "vx_eci_km_s": vx_eci,
            "vy_eci_km_s": vy_eci,
            "vz_eci_km_s": vz_eci,

            # ECI spherical
            "r_eci_km": r_eci_val,
            "theta_eci_deg": theta_eci_val,
            "phi_eci_deg": phi_eci_val,

            # ECEF (ITRS) cartesian
            "x_ecef_km": x_ecef,
            "y_ecef_km": y_ecef,
            "z_ecef_km": z_ecef,
            "vx_ecef_km_s": vx_ecef,
            "vy_ecef_km_s": vy_ecef,
            "vz_ecef_km_s": vz_ecef,

            # ECEF spherical
            "r_ecef_km": r_ecef_val,
            "theta_ecef_deg": theta_ecef_val,
            "phi_ecef_deg": phi_ecef_val,

            # Classical elements
            "sma_km": sma_km,
            "ecc": ecc_val,
            "inc_deg": inc_deg,
            "raan_deg": raan_deg,
            "argp_deg": argp_deg,
            "nu_deg": nu_deg,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def generate_single_orbit(i, orbit_types, time_step, base_epoch, two_d=False, num_steps=None):
    """
    Generate and propagate a single orbit.
    
    Args:
        i (int): Orbit index.
        orbit_types (tuple): Tuple of orbit types to choose from.
        time_step (astropy.units.quantity.Quantity): Time step for propagation.
        base_epoch (astropy.time.Time): Base epoch for the orbit.
    
    Returns:
        pd.DataFrame: DataFrame containing the propagated orbit data.
    """
    np.random.seed(i)  # Set unique seed for reproducibility
    chosen_type = np.random.choice(orbit_types)
    orb = generate_random_orbit(base_epoch, orbit_type=chosen_type, two_d=two_d)
    orbit_id = f"Orbit_{i+1}"
    df = propagate_orbit_to_df(
        orbit_obj=orb,
        orbit_id=orbit_id,
        orbit_regime=chosen_type,
        time_step=time_step,
        num_steps=num_steps
    )
    return df

def generate_orbits_dataset(n_orbits=2, orbit_types=("LEO", "MEO", "HEO", "GEO"), time_step=60*u.s,
                            out_csv=None, num_workers=None, two_d=False, num_steps=None):
    """
    Generate multiple random orbits in parallel, propagate each, and store data in a single CSV.
    
    Args:
        n_orbits (int): Number of orbits to generate.
        orbit_types (tuple): Tuple of orbit types to choose from.
        time_step (astropy.units.quantity.Quantity): Time step for propagation.
        out_csv (str): Output CSV file path.
        num_workers (int, optional): Number of worker processes. Defaults to CPU core count.
    
    Returns:
        pd.DataFrame: Concatenated DataFrame of all generated orbits.
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    base_epoch = Time("J2000", scale="tt")
    
    # Create a partial function with fixed arguments
    generate_func = partial(generate_single_orbit, orbit_types=orbit_types, time_step=time_step, base_epoch=base_epoch, two_d=two_d, num_steps=num_steps)
    
    # Use multiprocessing pool to generate orbits in parallel
    with multiprocessing.Pool(num_workers) as pool:
        results = []
        # Process orbits and show progress
        for df in tqdm(pool.imap_unordered(generate_func, range(n_orbits)), total=n_orbits, desc="Generating orbits", unit="orbit"):
            results.append(df)
    
    # Combine all DataFrames
    final_df = pd.concat(results, ignore_index=True)
    if out_csv is not None:
        final_df.to_csv(out_csv, index=False)
        print(f"Saved {len(final_df)} rows to {out_csv}")
    return final_df


def split_orbits_by_id(df, train_ratio=0.8, val_ratio=0.1):
    unique_ids = df["orbit_id"].unique()
    np.random.shuffle(unique_ids)

    n_total = len(unique_ids)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    train_ids = unique_ids[:n_train]
    val_ids   = unique_ids[n_train:n_train+n_val]
    test_ids  = unique_ids[n_train+n_val:]

    df_train = df[df["orbit_id"].isin(train_ids)].copy()
    df_val   = df[df["orbit_id"].isin(val_ids)].copy()
    df_test  = df[df["orbit_id"].isin(test_ids)].copy()

    return df_train, df_val, df_test
