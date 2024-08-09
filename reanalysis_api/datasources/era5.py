import os
from pathlib import Path
import pandas as pd
import xarray as xr
import pint_xarray  # type: ignore
import metpy.calc as mpcalc
import metpy.constants as mpconst
from metpy.units import units

from reanalysis_api.models import Profile, ProfileData, ProfileUnits


def arco_era5_native() -> xr.Dataset:
    return xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1",
        chunks=None,
        storage_options=dict(token="anon"),
    )


def arco_era5_isobaric() -> xr.Dataset:
    return xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        chunks=None,
        storage_options=dict(token="anon"),
    )


def era5_level_coefficients() -> pd.DataFrame:
    parent = Path(__name__).parents[0].absolute()
    datadir = os.path.join(parent, "data")
    file = os.path.join(datadir, "era5_vertical.txt")
    cols = [
        "a",
        "b",
        "ph",
        "pf",
        "geopotential_altitude",
        "geometric_altitude",
        "temperature",
        "density",
    ]
    return pd.read_csv(file, sep="\t", names=cols)


def surface_pressure(ds: xr.Dataset) -> float:
    sfcp = (ds.surface_pressure * units(ds.surface_pressure.units)).pint.to("Pa")
    return float(sfcp)


def transform_era5_isobaric(ds: xr.Dataset) -> Profile:
    lat = float(ds.coords["latitude"])
    lon = float(ds.coords["longitude"])

    # sfc
    sfc_pressure = (
        (float(ds["surface_pressure"]) * units(ds["surface_pressure"].units))
        .to(units("hPa"))
        .magnitude
    )
    sfc_height = 0
    sfc_temp = ds["2m_temperature"]
    sfc_dewpoint = ds["2m_dewpoint_temperature"]
    sfc_u_wind = ds["10m_u_component_of_wind"]
    sfc_v_wind = ds["10m_v_component_of_wind"]
    above_sfc_mask = ds["level"] < sfc_pressure

    # aloft, masked for above ground
    pressure = ds["level"].where(above_sfc_mask, drop=True)
    height = ds["geopotential"].where(above_sfc_mask, drop=True) / mpconst.g
    temperature = ds["temperature"].where(above_sfc_mask, drop=True)

    specific_humidity = ds["specific_humidity"].where(above_sfc_mask, drop=True)
    dewpoint = mpcalc.dewpoint_from_specific_humidity(
        pressure * units.hectopascal,
        temperature * units(temperature.units),
        specific_humidity * units("kg/kg"),
    )
    dewpoint = dewpoint.pint.to(temperature.units)

    u_wind = ds["u_component_of_wind"].where(above_sfc_mask, drop=True)
    v_wind = ds["v_component_of_wind"].where(above_sfc_mask, drop=True)
    omega = ds["vertical_velocity"].where(above_sfc_mask, drop=True)

    profile_data = ProfileData(
        pressure=_merge(sfc_pressure, pressure),
        height=_merge(sfc_height, height),
        temperature=_merge(sfc_temp, temperature),
        dewpoint=_merge(sfc_dewpoint, dewpoint),
        u_wind=_merge(sfc_u_wind, u_wind),
        v_wind=_merge(sfc_v_wind, v_wind),
        omega=_merge(None, omega),
    )

    # units
    pressure_units = "hPa"
    height_units = "m"
    temperature_units = temperature.units
    dewpoint_units = temperature.units
    u_wind_units = u_wind.units
    v_wind_units = v_wind.units
    omega_units = omega.units

    profile_units = ProfileUnits(
        pressure=pressure_units,
        height=height_units,
        temperature=temperature_units,
        dewpoint=dewpoint_units,
        u_wind=u_wind_units,
        v_wind=v_wind_units,
        omega=omega_units,
    )

    return Profile(
        lat=lat,
        lon=lon,
        timestamp=pd.Timestamp(ds["time"].values).to_pydatetime(),
        data=profile_data,
        units=profile_units,
    )


def transform_era5_native(ds: xr.Dataset, sfc_pres_pa: float) -> Profile:
    lat = float(ds.coords["latitude"])
    lon = float(ds.coords["longitude"])

    vert_levels = era5_level_coefficients()
    pressure = (vert_levels.a + vert_levels.b * sfc_pres_pa)[1:] / 10
    # hacky here - convert to hPa
    # pressure /= 100

    height = ds["geopotential"] / mpconst.g
    temperature = ds["temperature"]

    specific_humidity = ds["specific_humidity"]
    dewpoint = mpcalc.dewpoint_from_specific_humidity(
        # conform the dimensions
        xr.DataArray(pressure, dims=["hybrid"]) * units.hectopascal,
        temperature * units(temperature.units),
        specific_humidity * units("kg/kg"),
    )
    dewpoint = dewpoint.pint.to(temperature.units)

    u_wind = ds["u_component_of_wind"]
    v_wind = ds["v_component_of_wind"]
    omega = ds["vertical_velocity"]

    profile_data = ProfileData(
        pressure=_reversed(pressure),
        height=_reversed(height),
        temperature=_reversed(temperature),
        dewpoint=_reversed(dewpoint),
        u_wind=_reversed(u_wind),
        v_wind=_reversed(v_wind),
        omega=_reversed(omega),
    )

    # units
    pressure_units = "hPa"
    height_units = "m"
    temperature_units = temperature.units
    dewpoint_units = temperature.units
    u_wind_units = u_wind.units
    v_wind_units = v_wind.units
    omega_units = omega.units

    profile_units = ProfileUnits(
        pressure=pressure_units,
        height=height_units,
        temperature=temperature_units,
        dewpoint=dewpoint_units,
        u_wind=u_wind_units,
        v_wind=v_wind_units,
        omega=omega_units,
    )

    return Profile(
        lat=lat,
        lon=lon,
        timestamp=pd.Timestamp(ds["time"].values).to_pydatetime(),
        data=profile_data,
        units=profile_units,
    )


def _merge(sfc: xr.DataArray | float | None, da: xr.DataArray) -> list[float | None]:
    return [None if sfc is None else float(sfc)] + _reversed(da)


def _reversed(da: xr.DataArray) -> list[float]:
    return list(reversed(da.values.tolist()))
