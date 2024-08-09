from datetime import datetime
from fastapi import FastAPI, HTTPException

from reanalysis_api.datasources.era5 import (
    surface_pressure,
    transform_era5_isobaric,
    transform_era5_native,
)
from reanalysis_api.lifespan import lifespan, era5_isobaric_pool, era5_native_pool
from reanalysis_api.models import Profile

app = FastAPI(lifespan=lifespan)


@app.get("/era5_isobaric")
def era5_isobaric_profile(t: datetime, lat: float, lon: float) -> Profile:
    try:
        ds = era5_isobaric_pool.get_at(t, lat, lon)
        return transform_era5_isobaric(ds)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/era5_native")
def era5_native_profile(t: datetime, lat: float, lon: float) -> Profile:
    try:
        ds_isobaric = era5_isobaric_pool.get_at(t, lat, lon)
        ds_native = era5_native_pool.get_at(t, lat, lon)
        sfc_pressure = surface_pressure(ds_isobaric)
        return transform_era5_native(ds_native, sfc_pressure)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
