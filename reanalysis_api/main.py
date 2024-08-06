from datetime import datetime
from fastapi import FastAPI, HTTPException

from reanalysis_api.datasources.era5 import transform_era5_isobaric
from reanalysis_api.lifespan import lifespan, era5_isobaric_pool
from reanalysis_api.models import Profile

app = FastAPI(lifespan=lifespan)


@app.get("/era5/isobaric")
def era5_isobaric_profile(t: datetime, lat: float, lon: float) -> Profile:
    try:
        ds = era5_isobaric_pool.get_at(t, lat, lon)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return transform_era5_isobaric(ds)
